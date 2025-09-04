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
from pathlib import Path
from model_config import get_model_choices, get_model_info

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

MODEL_INFO = get_model_info()


def claude_conversation(actor, model, context, system_prompt=None):
    messages = [{"role": m["role"], "content": m["content"]} for m in context]

    # If Claude is the first model in the conversation, it must have a user message
    kwargs = {
        "model": model,
        "max_tokens": 1024,
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
        "max_tokens": 1024,
    }
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


def load_template(template_name, models):
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
            config["system_prompt"] = config["system_prompt"].format(
                **{f"model{j+1}_company": companies[j] for j in range(len(companies))},
                **{f"model{j+1}_display_name": display_names[j] for j in range(len(display_names))},
            )
            for message in config["context"]:
                message["content"] = message["content"].format(
                    **{f"model{j+1}_company": companies[j] for j in range(len(companies))},
                    **{f"model{j+1}_display_name": display_names[j] for j in range(len(display_names))},
                )

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
        "--discord",
        type=str,
        default=None,
        help="Enable Discord posting with a profile name from ./discord (e.g., 'chronicle').",
    )
    parser.add_argument(
        "--media",
        type=str,
        default=None,
        help="Enable media agent with a preset from ./media (e.g., 'cli'). Omit to disable.",
    )
    args = parser.parse_args()

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

    configs = load_template(args.template, models)

    assert len(models) == len(
        configs
    ), f"Number of LMs ({len(models)}) does not match the number of elements in the template ({len(configs)})"

    system_prompts = [config.get("system_prompt", "") for config in configs]
    contexts = [config.get("context", []) for config in configs]

    # Validate starting state: if all histories are empty, abort with a helpful error
    if all((not ctx) for ctx in contexts):
        print(
            "Error: All agents have empty chat_history. Provide a conversation starter (e.g., a user message) in at least one agent's history file."
        )
        sys.exit(1)

    logs_folder = "BackroomsLogs"
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{logs_folder}/{'_'.join(models)}_{args.template}_{timestamp}.txt"

    # Optional media agent config (explicit only; disabled when omitted)
    media_cfg = (
        load_media_config(args.media) if (load_media_config and args.media) else None
    )

    # Optional discord agent config
    discord_cfg = load_discord_config(args.discord) if load_discord_config else None

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

    turn = 0
    transcript: list[dict[str, str]] = []
    while turn < args.max_turns:
        round_entries = []
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
            process_and_log_response(
                lm_response,
                lm_display_names[i],
                filename,
                contexts,
                i,
            )
            round_entries.append({"actor": lm_display_names[i], "text": lm_response})

        # After both actors in a round, fold the round into the transcript
        transcript.extend(round_entries)

        # After that, optionally invoke media agent once
        media_url: Optional[str] = None
        if media_cfg and run_media_agent:
            try:
                media_result = run_media_agent(
                    media_cfg=media_cfg,
                    selected_models=models,
                    round_entries=round_entries,
                    transcript=transcript,
                    filename=filename,
                    generate_text_fn=media_generate_text_fn,
                    model_info=MODEL_INFO,
                )
                if parse_result_for_image_ref and isinstance(media_result, dict):
                    media_url = parse_result_for_image_ref(media_result)
            except Exception as e:
                err = f"\nMedia Agent error: {e}"
                print(err)
                with open(filename, "a") as f:
                    f.write(err + "\n")
        # After the round, optionally post a Discord update
        if discord_cfg and run_discord_agent:
            try:
                discord_result = run_discord_agent(
                    discord_cfg=discord_cfg,
                    selected_models=models,
                    round_entries=round_entries,
                    transcript=transcript,
                    generate_text_fn=media_generate_text_fn,
                    model_info=MODEL_INFO,
                    media_url=media_url,
                    filename=filename,
                )
                # Helpful terminal + file logs of what was posted
                if isinstance(discord_result, dict) and "posted" in discord_result:
                    posted = discord_result.get("posted", {})
                    ch = posted.get("channel", "?")
                    sv = posted.get("server") or "default"
                    msg = posted.get("message", "")
                    murl = posted.get("mediaUrl")
                    header = "\n\033[1m\033[38;2;120;180;255mDiscord Agent\033[0m"
                    print(header)
                    print(f"Channel: {ch} (server: {sv})")
                    print(f"Message: {msg}")
                    if murl:
                        print(f"Media:   {murl}")
                    with open(filename, "a") as f:
                        f.write("\n### Discord Agent ###\n")
                        f.write(f"Channel: {ch} (server: {sv})\n")
                        f.write(f"Message: {msg}\n")
                        if murl:
                            f.write(f"Media: {murl}\n")
            except Exception as e:
                err = f"\nDiscord Agent error: {e}"
                print(err)
                with open(filename, "a") as f:
                    f.write(err + "\n")
        turn += 1

    print(f"\nReached maximum number of turns ({args.max_turns}). Conversation ended.")
    with open(filename, "a") as f:
        f.write(
            f"\nReached maximum number of turns ({args.max_turns}). Conversation ended.\n"
        )


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
        exit()

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
