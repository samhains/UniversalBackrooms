import anthropic
import openai
import json
import datetime
import os
import argparse
import dotenv
import sys

# Attempt to load from .env file, but don't override existing env vars
dotenv.load_dotenv(override=False)

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not anthropic_api_key or not openai_api_key:
    print(
        "Error: ANTHROPIC_API_KEY and OPENAI_API_KEY must be set in the environment or in a .env file."
    )
    sys.exit(1)

anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
openai_client = openai.OpenAI(api_key=openai_api_key)

MODEL_INFO = {
    "sonnet": {
        "api_name": "claude-3-5-sonnet-20240620",
        "display_name": "Claude",
        "company": "anthropic",
    },
    "opus": {
        "api_name": "claude-3-opus-20240229",
        "display_name": "Claude",
        "company": "anthropic",
    },
    "gpt4o": {
        "api_name": "gpt-4o-2024-08-06",
        "display_name": "GPT4o",
        "company": "openai",
    },
    "o1-preview": {"api_name": "o1-preview", "display_name": "O1", "company": "openai"},
    "o1-mini": {"api_name": "o1-mini", "display_name": "Mini", "company": "openai"},
}


def claude_conversation(actor, model, context, system_prompt=None):
    messages = [{"role": m["role"], "content": m["content"]} for m in context]
    kwargs = {
        "model": model,
        "max_tokens": 1024,
        "temperature": 1.0,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
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
        kwargs["max_completion_tokens"] = 4000
    else:
        kwargs["max_tokens"] = 1024

    response = openai_client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def load_template(template_name, models):
    try:
        with open(f"templates/{template_name}.jsonl", "r") as f:
            configs = [json.loads(line) for line in f]

        companies = [MODEL_INFO[model]["company"] for model in models]
        actors = [
            f"{MODEL_INFO[model]['display_name']} {i+1}"
            for i, model in enumerate(models)
        ]

        for i, config in enumerate(configs):
            config["system_prompt"] = config["system_prompt"].format(
                **{f"lm{j+1}_company": company for j, company in enumerate(companies)},
                **{f"lm{j+1}_actor": actor for j, actor in enumerate(actors)},
            )
            for message in config["context"]:
                message["content"] = message["content"].format(
                    **{
                        f"lm{j+1}_company": company
                        for j, company in enumerate(companies)
                    },
                    **{f"lm{j+1}_actor": actor for j, actor in enumerate(actors)},
                )

            if MODEL_INFO[models[i]]["company"] == "openai" and config["system_prompt"]:
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
        return configs
    except FileNotFoundError:
        print(f"Error: Template '{template_name}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in template '{template_name}'.")
        exit(1)


def get_available_templates():
    template_dir = "./templates"
    templates = []
    for file in os.listdir(template_dir):
        if file.endswith(".jsonl"):
            templates.append(os.path.splitext(file)[0])
    return templates


def main():
    parser = argparse.ArgumentParser(
        description="Run conversation between two or more AI language models."
    )
    parser.add_argument(
        "--lm",
        choices=["sonnet", "opus", "gpt4o", "o1-preview", "o1-mini"],
        nargs="+",
        default=["opus", "opus"],
        help="Choose the models for LMs (default: opus opus)",
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
        default=10,
        help="Maximum number of turns in the conversation (default: 10)",
    )
    args = parser.parse_args()

    models = args.lm
    configs = load_template(args.template, models)

    assert len(models) == len(
        configs
    ), f"Number of LMs ({len(models)}) does not match the number of elements in the template ({len(configs)})"

    lm_models = [MODEL_INFO[model]["api_name"] for model in models]
    lm_display_names = [
        f"{MODEL_INFO[model]['display_name']} {i+1}" for i, model in enumerate(models)
    ]

    system_prompts = [config["system_prompt"] for config in configs]
    contexts = [config["context"] for config in configs]

    logs_folder = "BackroomsLogs"
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{logs_folder}/{'_'.join(models)}_{args.template}_{timestamp}.txt"

    turn = 0
    while turn < args.max_turns:
        for i in range(len(models)):
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
    else:
        return gpt4_conversation(
            actor, model, context, system_prompt if system_prompt else None
        )


def process_and_log_response(response, actor, filename, contexts, current_model_index):
    print(f"\n{actor}:")
    print(response)
    with open(filename, "a") as f:
        f.write(f"\n{actor}:\n")
        f.write(response)
        f.write("\n")

    if "^C^C" in response:
        print(f"\n{actor} has ended the conversation with ^C^C.")
        with open(filename, "a") as f:
            f.write(f"\n{actor} has ended the conversation with ^C^C.\n")
        exit()

    # Add the response to all contexts
    for i, context in enumerate(contexts):
        if i == current_model_index:
            context.append({"role": "assistant", "content": response})
        else:
            context.append({"role": "user", "content": response})


if __name__ == "__main__":
    main()
