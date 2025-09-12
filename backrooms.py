import anthropic
import openai
import json
import datetime
import os
import argparse
import dotenv
import sys
import colorsys
import requests
import re
import signal
from pathlib import Path
from model_config import get_model_choices, get_model_info
from typing import Optional

# Local imports for optional media agent
try:
    from media_agent import load_media_config, run_media_agent, parse_result_for_image_ref
except Exception:
    load_media_config = None  # type: ignore
    run_media_agent = None  # type: ignore
    parse_result_for_image_ref = None  # type: ignore

# Local imports for optional discord agent
try:
    from discord_agent import load_discord_config, run_discord_agent
except Exception:
    load_discord_config = None  # type: ignore
    run_discord_agent = None  # type: ignore

# Attempt to load from .env file, but don't override existing env vars
dotenv.load_dotenv(override=False)

# API clients (lazily initialized if not set by main())
anthropic_client = None
openai_client = None
openrouter_client = None
# Optional global override set from CLI; do not use env vars
EXPLICIT_MAX_TOKENS: Optional[int] = None

MODEL_INFO = get_model_info()


class ManualStop(Exception):
    pass

_SAVE_WARNED = False  # print missing-env warning once per run


def claude_conversation(actor, model, context, system_prompt=None):
    messages = [{"role": m["role"], "content": m["content"]} for m in context]

    # If Claude is the first model in the conversation, it must have a user message
    # Anthropic requires max_tokens; use CLI override if provided, else a higher default
    max_toks = EXPLICIT_MAX_TOKENS if EXPLICIT_MAX_TOKENS is not None else 4096
    kwargs = {
        "model": model,
        "max_tokens": max_toks,
        "temperature": 1.0,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    # Lazy init for Anthropic client (used by media agent too)
    global anthropic_client
    if anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY must be set to call Anthropic models (e.g., for the media agent)."
            )
        anthropic_client = anthropic.Client(api_key=api_key)
    message = anthropic_client.messages.create(**kwargs)
    return message.content[0].text


def gpt4_conversation(actor, model, context, system_prompt=None):
    messages = [{"role": m["role"], "content": m["content"]} for m in context]

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 1.0,
    }

    if EXPLICIT_MAX_TOKENS is not None:
        kwargs["max_tokens"] = EXPLICIT_MAX_TOKENS
    else:
        if model == "o1-preview" or model == "o1-mini":
            kwargs["max_tokens"] = 4000
        else:
            kwargs["max_tokens"] = 1024

    # Lazy init for OpenAI client (used by media agent too)
    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY must be set to call OpenAI models (e.g., for the media agent)."
            )
        openai_client = openai.OpenAI(api_key=api_key)
    response = openai_client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def openrouter_conversation(actor, model, context, system_prompt=None):
    messages = [{"role": m["role"], "content": m["content"]} for m in context]

    # OpenRouter is OpenAI-compatible; we can include a system message directly
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    # Allow a local suffix marker to denote reasoning-enabled variant
    reasoning_flag = False
    api_model = model
    if isinstance(model, str) and model.endswith("#reasoning"):
        reasoning_flag = True
        api_model = model.split("#", 1)[0]

    kwargs = {
        "model": api_model,
        "messages": messages,
        "temperature": 1.0,
    }
    # Only set max_tokens for OpenRouter when explicitly requested via CLI
    if EXPLICIT_MAX_TOKENS is not None:
        kwargs["max_tokens"] = EXPLICIT_MAX_TOKENS
    # Enable Hermes 4 internal reasoning traces when requested via vendor extension
    if reasoning_flag:
        kwargs["extra_body"] = {"reasoning": {"enabled": True, "exclude": False}, "include_reasoning": True}
    # Lazy init for OpenRouter client (used by media agent too)
    global openrouter_client
    if openrouter_client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY must be set to call OpenRouter models (e.g., for the media agent)."
            )
        openrouter_client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    response = openrouter_client.chat.completions.create(**kwargs)
    # Prefer normal content; if empty, fall back to reasoning text if present
    try:
        msg = response.choices[0].message
        content = getattr(msg, "content", None) or ""
        if content and str(content).strip():
            return content

        # Attempt to read unified reasoning fields for a fallback
        reason_txt = None
        # Direct attribute access
        r = getattr(msg, "reasoning", None)
        if isinstance(r, str) and r.strip():
            reason_txt = r.strip()
        elif isinstance(r, dict):
            reason_txt = r.get("text") or r.get("summary")
        # Try reasoning_details
        if not reason_txt:
            rd = getattr(msg, "reasoning_details", None)
            if isinstance(rd, list) and rd:
                parts = []
                for item in rd:
                    if isinstance(item, dict):
                        t = item.get("text") or item.get("summary") or item.get("data")
                        if t:
                            parts.append(t)
                if parts:
                    reason_txt = "\n".join(parts)
        if reason_txt and str(reason_txt).strip():
            return str(reason_txt)
    except Exception:
        pass
    return response.choices[0].message.content or ""


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _parse_history_markdown(path: str):
    """Parse markdown chat history into a list of {role, content}.

    A message begins with a markdown heading whose text is 'user' or 'assistant'
    (case-insensitive), e.g.:
      ## user
      Content...
      ## assistant
      Reply...
    Content continues until the next heading or EOF.
    """
    text = _read_text_file(path)
    if not text.strip():
        return []

    lines = text.splitlines()
    msgs = []
    current_role = None
    current_lines = []
    role_header_re = re.compile(r"^\s{0,3}#{1,6}\s*(user|assistant)\s*$", re.IGNORECASE)

    def flush():
        if current_role is not None:
            content = "\n".join(current_lines).rstrip("\n")
            msgs.append({"role": current_role, "content": content})

    for line in lines:
        m = role_header_re.match(line)
        if m:
            # new section starts
            flush()
            current_role = m.group(1).lower()
            current_lines = []
        else:
            if current_role is not None:
                current_lines.append(line)
            else:
                # ignore preamble lines before first heading
                continue
    flush()
    return msgs


def _parse_folder_template(template_name: str):
    base = Path("templates") / template_name
    spec_path = base / "template.json"
    with spec_path.open("r", encoding="utf-8") as f:
        spec = json.load(f)
    if not isinstance(spec, dict) or not isinstance(spec.get("agents"), list):
        raise ValueError("Template folder spec must contain an 'agents' list")
    configs = []
    for agent in spec["agents"]:
        sp_rel = agent.get("system")
        hist_rel = agent.get("history")
        sp_text = ""
        if isinstance(sp_rel, str) and sp_rel:
            sp_text = _read_text_file(str(base / sp_rel)).strip()
        ctx = []
        if isinstance(hist_rel, str) and hist_rel:
            ctx = _parse_history_markdown(str(base / hist_rel))
        configs.append({"system_prompt": sp_text, "context": ctx})
    return configs


def load_template(template_name, models, cli_vars: Optional[dict[str, str]] = None):
    try:
        # Prefer folder-based template: templates/<name>/template.json
        folder_spec = Path("templates") / template_name / "template.json"
        if folder_spec.exists():
            configs = _parse_folder_template(template_name)
        else:
            # Legacy JSON spec at templates/<name>.json
            json_path = f"templates/{template_name}.json"
            with open(json_path, "r", encoding="utf-8") as f:
                spec = json.load(f)

            if not isinstance(spec, dict) or not isinstance(spec.get("agents"), list):
                raise ValueError("Template JSON must contain an 'agents' list")

            configs = []
            for agent in spec["agents"]:
                sp_path = agent.get("system_prompt_file", "")
                hist_path = agent.get("history_file", "")
                _system_raw = _read_text_file(sp_path)
                system_prompt = _system_raw if _system_raw.strip() else ""
                context = _parse_history_markdown(hist_path)
                configs.append({"system_prompt": system_prompt, "context": context})

        # Build model metadata used for templated placeholders
        # Expose {modelN_company} and {modelN_display_name} for N starting at 1
        companies = []
        display_names = []
        # Optional template variables from vars.json inside the template folder
        extra_vars = {}
        try:
            vars_path = Path("templates") / template_name / "vars.json"
            if vars_path.exists():
                with vars_path.open("r", encoding="utf-8") as vf:
                    raw_vars = json.load(vf)
                    if isinstance(raw_vars, dict):
                        # Escape braces to avoid .format placeholder issues in user-provided strings
                        extra_vars = {
                            k: (v.replace("{", "{{").replace("}", "}}") if isinstance(v, str) else v)
                            for k, v in raw_vars.items()
                        }
        except Exception:
            extra_vars = {}

        # Merge CLI-provided vars, with CLI taking precedence
        if cli_vars:
            for k, v in cli_vars.items():
                if isinstance(v, str):
                    extra_vars[k] = v.replace("{", "{{").replace("}", "}}")
                else:
                    extra_vars[k] = v

        for i, model in enumerate(models):
            if model.lower() == "cli":
                companies.append("CLI")
                display_names.append("CLI")
            else:
                base_company = MODEL_INFO[model]["company"]
                # For OpenRouter models, expose the vendor prefix from api_name (e.g., 'nousresearch')
                if base_company == "openrouter":
                    api_name = MODEL_INFO[model].get("api_name", "")
                    vendor = api_name.split("/", 1)[0] if "/" in api_name else base_company
                    companies.append(vendor)
                else:
                    companies.append(base_company)
                display_names.append(MODEL_INFO[model]["display_name"])

        for i, config in enumerate(configs):
            if models[i].lower() == "cli":
                config["cli"] = True
                continue

            # Format placeholders in system prompt with new keys
            _fmt_args = {
                **{f"model{j+1}_company": companies[j] for j in range(len(companies))},
                **{f"model{j+1}_display_name": display_names[j] for j in range(len(display_names))},
                **extra_vars,
            }
            config["system_prompt"] = config["system_prompt"].format(**_fmt_args)
            for message in config["context"]:
                message["content"] = message["content"].format(**_fmt_args)

            if (
                models[i] in MODEL_INFO
                and MODEL_INFO[models[i]]["company"] == "openai"
                and config["system_prompt"]
            ):
                system_prompt_added = False
                for message in config["context"]:
                    if message["role"] == "user":
                        message["content"] = (
                            f"<SYSTEM>{config['system_prompt']}</SYSTEM>\n\n{message['content']}"
                        )
                        system_prompt_added = True
                        break
                if not system_prompt_added:
                    config["context"].append(
                        {
                            "role": "user",
                            "content": f"<SYSTEM>{config['system_prompt']}</SYSTEM>",
                        }
                    )
            config["cli"] = config.get("cli", False)
        return configs
    except FileNotFoundError:
        print(f"Error: Template '{template_name}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in template '{template_name}'.")
        exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)


def get_available_templates():
    template_dir = Path("./templates")
    names = set()
    # Folder-based templates
    for p in template_dir.iterdir():
        if p.is_dir() and (p / "template.json").exists():
            names.add(p.name)
    # File-based templates (legacy)
    for file in template_dir.iterdir():
        if file.is_file() and file.suffix == ".json" and not file.name.endswith(".media.json"):
            names.add(file.stem)
    return sorted(names)


def main():
    global anthropic_client
    global openai_client
    global openrouter_client
    parser = argparse.ArgumentParser(
        description="Run conversation between two or more AI language models."
    )
    parser.add_argument(
        "--lm",
        choices=get_model_choices(include_cli=True),
        nargs="+",
        default=["opus", "opus"],
        help="Choose model aliases from model_config (or 'cli' for the world interface)",
    )

    available_templates = get_available_templates()
    parser.add_argument(
        "--template",
        choices=available_templates,
        default="cli" if "cli" in available_templates else available_templates[0],
        help=f"Choose a conversation template (available: {', '.join(available_templates)})",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=float("inf"),
        help="Maximum number of turns in the conversation (default: infinity)",
    )
    parser.add_argument(
        "--max-context-frac",
        type=float,
        default=0.0,
        help="Early-stop when estimated prompt tokens exceed this fraction of the context window (0 disables).",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=128000,
        help="Assumed context window size in tokens for the limiting model (default: 128000).",
    )
    parser.add_argument(
        "--discord",
        action="append",
        default=None,
        help="Enable one or more Discord profiles (repeatable or comma-separated), from ./discord/<name>.json.",
    )
    parser.add_argument(
        "--media",
        action="append",
        default=None,
        help="Enable one or more media presets (repeatable or comma-separated). If omitted, no media agent runs.",
    )
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        help="Template variable override NAME=VALUE. Repeatable.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Convenience text variable. Sets both QUERY and DREAM_TEXT for templates.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max output tokens per response. For OpenRouter, only applied when provided.",
    )
    args = parser.parse_args()
    # Set global explicit max tokens if provided
    if args.max_tokens is not None and args.max_tokens > 0:
        global EXPLICIT_MAX_TOKENS
        EXPLICIT_MAX_TOKENS = args.max_tokens

    # Build CLI vars map from --var NAME=VALUE and --query
    cli_vars: dict[str, str] = {}
    # --var can be repeated
    if isinstance(args.var, list):
        for item in args.var:
            if not isinstance(item, str):
                continue
            if "=" in item:
                name, value = item.split("=", 1)
                name = name.strip()
                if name:
                    cli_vars[name] = value
    # --query sets both QUERY and DREAM_TEXT for convenience
    if args.query:
        if "QUERY" not in cli_vars:
            cli_vars["QUERY"] = args.query
        if "DREAM_TEXT" not in cli_vars:
            cli_vars["DREAM_TEXT"] = args.query

    models = args.lm
    lm_models = []
    lm_display_names = []

    companies = []
    actors = []

    for i, model in enumerate(models):
        if model.lower() == "cli":
            lm_display_names.append("CLI")
            lm_models.append("cli")
            companies.append("CLI")
            actors.append("CLI")
        else:
            if model in MODEL_INFO:
                lm_display_names.append(f"{MODEL_INFO[model]['display_name']} {i+1}")
                api_name = MODEL_INFO[model]["api_name"]
                # For OpenRouter Hermes variants, encode reasoning preference in the model string
                if (
                    MODEL_INFO[model].get("company") == "openrouter"
                    and MODEL_INFO[model].get("reasoning_enabled") is True
                ):
                    lm_models.append(f"{api_name}#reasoning")
                else:
                    lm_models.append(api_name)
                companies.append(MODEL_INFO[model]["company"])
                actors.append(f"{MODEL_INFO[model]['display_name']} {i+1}")
            else:
                print(f"Error: Model '{model}' not found in MODEL_INFO.")
                sys.exit(1)

    # Filter out models not in MODEL_INFO (like 'cli')
    anthropic_models = [
        model
        for model in models
        if model in MODEL_INFO and MODEL_INFO[model]["company"] == "anthropic"
    ]
    if anthropic_models:
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            print(
                "Error: ANTHROPIC_API_KEY must be set in the environment or in a .env file."
            )
            sys.exit(1)
        anthropic_client = anthropic.Client(api_key=anthropic_api_key)

    openai_models = [
        model
        for model in models
        if model in MODEL_INFO and MODEL_INFO[model]["company"] == "openai"
    ]
    if openai_models:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print(
                "Error: OPENAI_API_KEY must be set in the environment or in a .env file."
            )
            sys.exit(1)
        openai_client = openai.OpenAI(api_key=openai_api_key)

    # Initialize OpenRouter client only if selected
    openrouter_models = [
        model
        for model in models
        if model in MODEL_INFO and MODEL_INFO[model]["company"] == "openrouter"
    ]
    if openrouter_models:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            print(
                "Error: OPENROUTER_API_KEY must be set in the environment or in a .env file."
            )
            sys.exit(1)
        # OpenRouter is OpenAI-compatible; just set base_url and api_key
        openrouter_client = openai.OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    # (moved) Template loading and context setup occurs later to allow CLI var overlays

    logs_folder = "BackroomsLogs"
    # Group logs by template for easier organization
    template_logs_folder = os.path.join(logs_folder, args.template)
    if not os.path.exists(template_logs_folder):
        os.makedirs(template_logs_folder, exist_ok=True)

    # Track run start (UTC) for DB metadata
    run_start = datetime.datetime.now(datetime.timezone.utc)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{template_logs_folder}/{'_'.join(models)}_{args.template}_{timestamp}.txt"
    # Write a concise run header for easier forensics/search
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write("### Backrooms Run ###\n")
            f.write(f"Template: {args.template}\n")
            f.write(f"Models: {', '.join(models)}\n")
            f.write(f"Started: {run_start.isoformat()}\n")
    except Exception:
        pass

    # Optional media agent configs: support multiple presets
    media_cfgs = []
    # Read global per-run overrides for media presets from env (JSON)
    media_overrides_env = os.getenv("BACKROOMS_MEDIA_OVERRIDES")
    media_overrides: dict[str, object] = {}
    if media_overrides_env:
        try:
            _mov = json.loads(media_overrides_env)
            if isinstance(_mov, dict):
                media_overrides = _mov
        except Exception:
            pass
    if args.media and load_media_config:
        # Flatten possible comma-separated values across repeated flags
        raw_media: list[str] = []
        for m in (args.media or []):
            if isinstance(m, str) and "," in m:
                raw_media.extend([x.strip() for x in m.split(",") if x.strip()])
            elif isinstance(m, str):
                raw_media.append(m.strip())
        seen = set()
        for name in raw_media:
            if not name or name in seen:
                continue
            seen.add(name)
            cfg = load_media_config(name)
            if cfg:
                cfg.setdefault("__name__", name)
                # Apply shallow overrides from env if provided
                if media_overrides:
                    try:
                        for k, v in media_overrides.items():
                            cfg[k] = v
                    except Exception:
                        pass
                media_cfgs.append(cfg)
            else:
                try:
                    media_dir = Path("media")
                    choices = []
                    if media_dir.exists():
                        for p in media_dir.iterdir():
                            if p.is_file() and p.suffix == ".json":
                                choices.append(p.stem)
                    print(
                        f"Warning: media preset '{name}' not found. Available: {', '.join(sorted(choices)) or 'none'}"
                    )
                except Exception:
                    print(f"Warning: media preset '{name}' not found.")

    # Optional Discord agent configs: support multiple profiles
    discord_cfgs = []
    # Maintain per-profile message history (current run) for Discord bots
    discord_bot_history: dict[str, list[str]] = {}
    if args.discord and load_discord_config:
        raw_dc: list[str] = []
        for d in (args.discord or []):
            if isinstance(d, str) and "," in d:
                raw_dc.extend([x.strip() for x in d.split(",") if x.strip()])
            elif isinstance(d, str):
                raw_dc.append(d.strip())
        seen_dc = set()
        # Read global per-run overrides for Discord profiles from env (JSON)
        discord_overrides_env = os.getenv("BACKROOMS_DISCORD_OVERRIDES")
        discord_overrides: dict[str, object] = {}
        if discord_overrides_env:
            try:
                import json as _json
                parsed = _json.loads(discord_overrides_env)
                if isinstance(parsed, dict):
                    discord_overrides = parsed  # shallow override map
            except Exception:
                pass
        for name in raw_dc:
            if not name or name in seen_dc:
                continue
            seen_dc.add(name)
            cfg = load_discord_config(name)
            if cfg:
                cfg.setdefault("__name__", name)
                # Apply shallow overrides from env if provided (config-level control)
                if discord_overrides:
                    try:
                        for k, v in discord_overrides.items():
                            cfg[k] = v
                    except Exception:
                        pass
                discord_cfgs.append(cfg)
                discord_bot_history[name] = []
            else:
                try:
                    dd = Path("discord")
                    choices = []
                    if dd.exists():
                        for p in dd.iterdir():
                            if p.is_file() and p.suffix == ".json":
                                choices.append(p.stem)
                    print(
                        f"Warning: discord preset '{name}' not found. Available: {', '.join(sorted(choices)) or 'none'}"
                    )
                except Exception:
                    print(f"Warning: discord preset '{name}' not found.")

    if discord_cfgs:
        try:
            loaded_names = ", ".join([cfg.get("__name__", "?") for cfg in discord_cfgs])
            print(f"[backrooms] Discord profiles loaded: {loaded_names}")
        except Exception:
            pass

    def media_generate_text_fn(system_prompt: str, api_model: str, user_message: str) -> str:
        # Reuse existing model call path, branching by provider name in api_model
        # api_model is the provider API string (e.g., 'claude-3-...', 'gpt-4o-...')
        context = [{"role": "user", "content": user_message}]
        if api_model.startswith("claude-"):
            return claude_conversation("Media Agent", api_model, context, system_prompt)
        elif "/" in api_model:
            # Heuristic: OpenRouter models typically include a provider prefix like "org/model"
            return openrouter_conversation("Media Agent", api_model, context, system_prompt)
        else:
            return gpt4_conversation("Media Agent", api_model, context, system_prompt)

    # Heuristic token estimation (approx. 4 chars per token)
    def estimate_tokens_for_agent(i: int) -> int:
        text_parts = []
        sp = system_prompts[i] or ""
        if sp:
            text_parts.append(sp)
        for msg in contexts[i]:
            text_parts.append(str(msg.get("content", "")))
        chars = sum(len(t) for t in text_parts)
        # Avoid zero; add small overhead for roles/formatting
        return max(1, (chars // 4) + 8)

    def next_max_tokens_for_model(api_model: str) -> int:
        # Use CLI override if provided
        if EXPLICIT_MAX_TOKENS is not None:
            return EXPLICIT_MAX_TOKENS
        # Heuristics per provider/model string
        if api_model == "o1-preview" or api_model == "o1-mini":
            return 4000
        # Anthropic models typically start with 'claude-'
        if isinstance(api_model, str) and api_model.startswith("claude-"):
            return 4096
        # OpenRouter (contains '/') — leave modest budget unless overridden
        return 1024

    # Load template with CLI vars overlays
    configs = load_template(args.template, models, cli_vars=cli_vars)

    assert len(models) == len(
        configs
    ), f"Number of LMs ({len(models)}) does not match the number of elements in the template ({len(configs)})"

    system_prompts = [config.get("system_prompt", "") for config in configs]
    contexts = [config.get("context", []) for config in configs]
    # Snapshot initial contexts for metadata (before mutation by conversation)
    initial_contexts = [list(ctx) for ctx in contexts]

    # Validate starting state: if all histories are empty, abort with a helpful error
    if all((not ctx) for ctx in contexts):
        print(
            "Error: All agents have empty chat_history. Provide a conversation starter (e.g., a user message) in at least one agent's history file."
        )
        sys.exit(1)

    turn = 0
    transcript: list[dict[str, str]] = []

    # Persist run details + transcript into Supabase (best-effort)
    # Disabled by default. Set BACKROOMS_SAVE_ENABLED=1 to re-enable.
    def _save_run_to_supabase(exit_reason: str = "max_turns") -> None:
        # Respect explicit opt-in only; default is no-op so transcripts are
        # pushed via scripts/sync_backrooms.py after runs complete.
        if os.getenv("BACKROOMS_SAVE_ENABLED") != "1":
            return
        global _SAVE_WARNED
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = (
                os.getenv("SUPABASE_KEY")
                or os.getenv("SUPABASE_ANON_KEY")
                or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            )
            if not supabase_url or not supabase_key or os.getenv("BACKROOMS_SAVE_DISABLED"):
                if not _SAVE_WARNED and not os.getenv("BACKROOMS_SAVE_SILENT"):
                    print("[backrooms] Supabase save disabled or missing SUPABASE_URL/KEY; skipping.")
                    _SAVE_WARNED = True
                return
            # Build prompt from initial user messages if present
            init_user_msgs = []
            for ctx in initial_contexts:
                for msg in ctx:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        c = str(msg.get("content", ""))
                        if c.strip():
                            init_user_msgs.append(c)
            prompt_text = "\n\n".join(init_user_msgs) if init_user_msgs else None
            # Read transcript file content
            try:
                with open(filename, "r", encoding="utf-8") as fh:
                    transcript_text = fh.read()
            except Exception:
                transcript_text = ""
            # End timestamp + duration
            end_ts = datetime.datetime.now(datetime.timezone.utc)
            duration = (end_ts - run_start).total_seconds()
            payload = [{
                "models": models,
                "template": args.template,
                "max_turns": args.max_turns,
                "created_at": run_start.isoformat(),
                "duration_sec": duration,
                "log_file": filename,
                "exit_reason": exit_reason,
                "prompt": prompt_text,
                "transcript": transcript_text,
            }]
            # Additional fields like dream_id/source are attached during sync
            # via scripts/sync_backrooms.py to avoid coupling runtime to env vars.
            headers = {
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=representation",
            }
            url = f"{supabase_url}/rest/v1/backrooms?on_conflict=log_file"
            try:
                requests.post(url, headers=headers, json=payload, timeout=10)
            except Exception:
                pass
        except Exception:
            # Never interrupt the run on analytics failure
            pass

    # Register signal handlers to persist partial progress on interrupt/terminate
    def _handle_signal(signum, frame):  # type: ignore[no-redef]
        try:
            _save_run_to_supabase(exit_reason="interrupted")
        finally:
            # Exit with standard codes: SIGINT -> 130, SIGTERM -> 143
            code = 130 if signum == signal.SIGINT else 143
            sys.exit(code)

    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception:
        pass

    # Database save disabled by default; no early row creation.
    # Track Discord profiles that have already posted at start to avoid duplicates
    _discord_posted_start: set[str] = set()

    while turn < args.max_turns:
        # Announce round progress in terminal and append to log file
        try:
            round_no = turn + 1
            header = f"\n--- Round {round_no}/{args.max_turns} ---"
            print(header)
            with open(filename, "a") as f:
                f.write(f"\n### Round {round_no}/{args.max_turns}\n")
        except Exception:
            pass
        try:
            pass
        except Exception:
            pass
        # loop body continues below
        # Optional early stop based on estimated context budget
        if args.max_context_frac and args.context_window and args.context_window > 0:
            # Ensure the next responses for all agents would fit within the fraction
            would_exceed = False
            for i in range(len(models)):
                used = estimate_tokens_for_agent(i)
                budget = int(args.max_context_frac * args.context_window)
                # Determine model string used for generation path
                api_model = lm_models[i]
                # Strip any vendor flags like #reasoning
                if isinstance(api_model, str) and "#" in api_model:
                    api_model = api_model.split("#", 1)[0]
                nxt = next_max_tokens_for_model(api_model)
                if used + nxt >= budget:
                    would_exceed = True
                    break
            if would_exceed:
                msg = (
                    f"\nContext budget limit reached (>= {int(args.max_context_frac*100)}% of {args.context_window} tokens). Conversation ended."
                )
                print(msg)
                with open(filename, "a") as f:
                    f.write(msg + "\n")
                # Persist early stop (no-op unless BACKROOMS_SAVE_ENABLED=1)
                _save_run_to_supabase(exit_reason="early_stop")
                break

        round_entries = []
        manual_stopped = False
        for i in range(len(models)):
            if models[i].lower() == "cli":
                lm_response = cli_conversation(contexts[i])
            else:
                lm_response = generate_model_response(
                    lm_models[i],
                    lm_display_names[i],
                    contexts[i],
                    system_prompts[i],
                )
            try:
                process_and_log_response(
                    lm_response,
                    lm_display_names[i],
                    filename,
                    contexts,
                    i,
                )
            except ManualStop:
                manual_stopped = True
                # Append the partial entry to transcript before stopping
                round_entries.append({"actor": lm_display_names[i], "text": lm_response})
                break
            round_entries.append({"actor": lm_display_names[i], "text": lm_response})

        # If manual stop occurred, persist and end outer loop
        if manual_stopped:
            transcript.extend(round_entries)
            _save_run_to_supabase(exit_reason="manual_stop")
            break

        # First, for Discord profiles that should post only at the very start,
        # run them before media so headers land first in the channel.
        if discord_cfgs and run_discord_agent:
            for dcfg in discord_cfgs:
                try:
                    run_on = (dcfg.get("run_on") or "every").strip().lower()
                except Exception:
                    run_on = "every"
                post_once_at_start = bool(dcfg.get("post_once_at_start", False))
                name = dcfg.get("__name__", "")
                # Only fire on the very first round, once
                if turn == 0 and (post_once_at_start or run_on in ("first", "first_only", "start", "start_only")):
                    if name and name in _discord_posted_start:
                        continue
                    try:
                        res = run_discord_agent(
                            discord_cfg=dcfg,
                            selected_models=models,
                            round_entries=round_entries,
                            transcript=transcript,
                            generate_text_fn=media_generate_text_fn,
                            model_info=MODEL_INFO,
                            media_url=None,
                            bot_history=discord_bot_history.get(name, []),
                        )
                        # Track posted summary in bot history and mark as posted-at-start
                        try:
                            if name and isinstance(res, dict):
                                summary_text = res.get("composed_summary")
                                if isinstance(summary_text, str) and summary_text.strip():
                                    discord_bot_history.setdefault(name, []).append(summary_text.strip())
                            if name:
                                _discord_posted_start.add(name)
                        except Exception:
                            pass
                    except Exception as e:
                        err = f"\nDiscord Agent error: {e}"
                        print(err)
                        with open(filename, "a") as f:
                            f.write(err + "\n")

        # After the round, invoke media agents (one or many); media posts images on its own
        if media_cfgs and run_media_agent:
            for mcfg in media_cfgs:
                # Optional: only run media every N rounds (default 1 = every round)
                try:
                    every = int(mcfg.get("run_every_n_rounds", 1))
                except Exception:
                    every = 1
                if every < 1:
                    every = 1
                # 'turn' is 0-based; run on rounds where turn % every == 0
                if (turn % every) != 0:
                    continue
                try:
                    run_media_agent(
                        media_cfg=mcfg,
                        selected_models=models,
                        round_entries=round_entries,
                        transcript=transcript,
                        filename=filename,
                        generate_text_fn=media_generate_text_fn,
                        model_info=MODEL_INFO,
                    )
                except Exception as e:
                    err = f"\nMedia Agent error: {e}"
                    print(err)
                    with open(filename, "a") as f:
                        f.write(err + "\n")
        # After the round, post Discord updates (text-only) for each profile
        if discord_cfgs and run_discord_agent:
            for dcfg in discord_cfgs:
                # Optional: allow certain Discord presets to post only at the start of a dream
                try:
                    run_on = (dcfg.get("run_on") or "every").strip().lower()
                except Exception:
                    run_on = "every"
                post_once_at_start = bool(dcfg.get("post_once_at_start", False))
                # 'turn' is 0-based and incremented at end of loop; first round occurs when turn == 0
                if post_once_at_start or run_on in ("first", "first_only", "start", "start_only"):
                    # Skip here if not first turn, or if we already posted in the pre-media stage
                    name = dcfg.get("__name__", "")
                    if turn != 0 or (name and name in _discord_posted_start):
                        continue
                try:
                    res = run_discord_agent(
                        discord_cfg=dcfg,
                        selected_models=models,
                        round_entries=round_entries,
                        transcript=transcript,
                        generate_text_fn=media_generate_text_fn,
                        model_info=MODEL_INFO,
                        media_url=None,
                        bot_history=discord_bot_history.get(dcfg.get("__name__", ""), []),
                    )
                    # Append the composed summary to this bot's history
                    try:
                        name = dcfg.get("__name__", "")
                        if name and isinstance(res, dict):
                            summary_text = res.get("composed_summary")
                            if isinstance(summary_text, str) and summary_text.strip():
                                discord_bot_history.setdefault(name, []).append(summary_text.strip())
                    except Exception:
                        pass
                except Exception as e:
                    err = f"\nDiscord Agent error: {e}"
                    print(err)
                    with open(filename, "a") as f:
                        f.write(err + "\n")
        # Append this round to running transcript
        transcript.extend(round_entries)
        turn += 1
        # Checkpoint after each round (no-op unless BACKROOMS_SAVE_ENABLED=1)
        _save_run_to_supabase(exit_reason="in_progress")

    print(f"\nReached maximum number of turns ({args.max_turns}). Conversation ended.")
    with open(filename, "a") as f:
        f.write(
            f"\nReached maximum number of turns ({args.max_turns}). Conversation ended.\n"
        )
    # Persist run completion (no-op unless BACKROOMS_SAVE_ENABLED=1)
    _save_run_to_supabase(exit_reason="max_turns")


def generate_model_response(model, actor, context, system_prompt):
    if model.startswith("claude-"):
        return claude_conversation(
            actor, model, context, system_prompt if system_prompt else None
        )
    elif "/" in model:
        return openrouter_conversation(
            actor, model, context, system_prompt if system_prompt else None
        )
    else:
        return gpt4_conversation(
            actor, model, context, system_prompt if system_prompt else None
        )


def generate_distinct_colors():
    hue = 0
    golden_ratio_conjugate = 0.618033988749895
    while True:
        hue += golden_ratio_conjugate
        hue %= 1
        rgb = colorsys.hsv_to_rgb(hue, 0.95, 0.95)
        yield tuple(int(x * 255) for x in rgb)


color_generator = generate_distinct_colors()
actor_colors = {}


def get_ansi_color(rgb):
    return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def process_and_log_response(response, actor, filename, contexts, current_model_index):
    global actor_colors

    # Get or generate a color for this actor
    if actor not in actor_colors:
        actor_colors[actor] = get_ansi_color(next(color_generator))

    color = actor_colors[actor]
    bold = "\033[1m"
    reset = "\033[0m"

    # Create a visually distinct header for each actor
    console_header = f"\n{bold}{color}{actor}:{reset}"
    file_header = f"\n### {actor} ###\n"

    print(console_header)
    print(response)

    with open(filename, "a") as f:
        f.write(file_header)
        f.write(response + "\n")

    if "^C^C" in response:
        end_message = f"\n{actor} has ended the conversation with ^C^C."
        print(end_message)
        with open(filename, "a") as f:
            f.write(end_message + "\n")
        # Signal to main loop to handle graceful termination and persistence
        raise ManualStop()

    # Add the response to all contexts
    for i, context in enumerate(contexts):
        role = "assistant" if i == current_model_index else "user"
        context.append({"role": role, "content": response})


def cli_conversation(context):
    # Extract the last user message
    last_message = context[-1]["content"]
    # Prepare the payload
    payload = {"messages": [{"role": "user", "content": last_message}]}
    headers = {
        "Authorization": f"Bearer {os.getenv('WORLD_INTERFACE_KEY')}",
        "Content-Type": "application/json",
    }
    # Send POST request to the world-interface
    response = requests.post(
        "http://localhost:3000/v1/chat/completions",
        json=payload,
        headers=headers,
    )
    response.raise_for_status()
    response_data = response.json()
    cli_response = response_data["choices"][0]["message"]["content"]
    return cli_response


if __name__ == "__main__":
    main()
