import anthropic
import openai
import json
import datetime
import os
import argparse

opus = "claude-3-opus-20240229"
sonnet = "claude-3-5-sonnet-20240620"
gpt4o = "gpt-4o-2024-08-06"
o1_preview = "o1-preview"  # New OpenAI model
o1_mini = "o1-mini"

explorer_temperature = 1.0
cli_temperature = 1.0

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic()

# Initialize OpenAI client
openai_client = openai.OpenAI()


def claude_conversation(actor, model, temperature, context, system_prompt=None):
    messages = [{"role": m["role"], "content": m["content"]} for m in context]
    kwargs = {
        "model": model,
        "max_tokens": 1024,
        "temperature": temperature,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    message = anthropic_client.messages.create(**kwargs)
    return message.content[0].text


def gpt4_conversation(actor, model, temperature, context, system_prompt=None):
    messages = [{"role": m["role"], "content": m["content"]} for m in context]

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if model == "o1-preview" or model == "o1-mini":
        kwargs["max_completion_tokens"] = 8192
    else:
        kwargs["max_tokens"] = 1024

    response = openai_client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def load_template(
    template_name, cli_company, explorer_company, cli_actor, explorer_actor
):
    try:
        with open(f"templates/{template_name}.json", "r") as f:
            template = json.load(f)

        # Replace placeholders in system prompt
        template["system_prompt"] = template["system_prompt"].format(
            cli_company=cli_company,
            explorer_company=explorer_company,
            cli_actor=cli_actor,
            explorer_actor=explorer_actor,
        )

        # Replace placeholders in context messages
        for context in ["agent_01_context", "agent_02_context"]:
            for message in template[context]:
                message["content"] = message["content"].format(
                    cli_company=cli_company,
                    explorer_company=explorer_company,
                    cli_actor=cli_actor,
                    explorer_actor=explorer_actor,
                )

        return template
    except FileNotFoundError:
        print(f"Error: Template '{template_name}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in template '{template_name}'.")
        exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run conversation between an explorer model and a CLI model."
    )
    parser.add_argument(
        "--cli-model",
        choices=["sonnet", "opus", "gpt4o", "o1-preview", "o1-mini"],
        default="opus",
        help="Choose the model for the CLI (default: opus)",
    )
    parser.add_argument(
        "--explorer-model",
        choices=["sonnet", "opus", "gpt4o", "o1-preview", "o1-mini"],
        default="opus",
        help="Choose the model for the explorer/prompter (default: opus)",
    )
    parser.add_argument(
        "--template",
        choices=[
            "cli",
            "eureka",
            "fugue",
            "thoughtX",
            "gallery",
            "jung-mckenna",
            "student",
            "ethics",
        ],  # Add more template names as needed
        default="cli",
        help="Choose a conversation template (default: cli)",
    )
    args = parser.parse_args()

    cli_model = get_model(args.cli_model)
    explorer_model = get_model(args.explorer_model)

    cli_actor = f"{get_actor_prefix(args.cli_model)} CLI"
    explorer_actor = f"{get_actor_prefix(args.explorer_model)} Explorer"

    cli_company = get_company(args.cli_model)
    explorer_company = get_company(args.explorer_model)

    template = load_template(
        args.template, cli_company, explorer_company, cli_actor, explorer_actor
    )
    cli_system_prompt = template["system_prompt"]
    explorer_context = template["agent_02_context"]
    cli_context = template["agent_01_context"]

    explorer_folder = f"{args.explorer_model.capitalize()}Explorations"
    if not os.path.exists(explorer_folder):
        os.makedirs(explorer_folder)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{explorer_folder}/{args.explorer_model}_exploring_{args.cli_model}_{timestamp}.txt"

    turn = 0
    while True:
        # Explorer turn
        explorer_response = handle_conversation(
            explorer_model, explorer_actor, explorer_temperature, explorer_context
        )
        handle_response(
            explorer_response, explorer_actor, filename, explorer_context, cli_context
        )

        # CLI turn
        cli_response = handle_conversation(
            cli_model, cli_actor, cli_temperature, cli_context, cli_system_prompt
        )
        handle_response(
            cli_response, cli_actor, filename, cli_context, explorer_context
        )

        turn += 1


def handle_conversation(model, actor, temperature, context, system_prompt=None):
    if model.startswith("claude-"):
        return claude_conversation(actor, model, temperature, context, system_prompt)
    else:
        return gpt4_conversation(actor, model, temperature, context)


def handle_response(response, actor, filename, own_context, other_context):
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

    own_context.append({"role": "assistant", "content": response})
    other_context.append({"role": "user", "content": response})


def get_model(model_name):
    model_map = {
        "sonnet": sonnet,
        "opus": opus,
        "gpt4o": gpt4o,
        "o1-preview": o1_preview,
        "o1-mini": o1_mini,
    }
    return model_map[model_name]


def get_actor_prefix(model_name):
    prefix_map = {
        "sonnet": "Claude",
        "opus": "Claude",
        "gpt4o": "GPT4o",
        "o1-preview": "O1",
        "o1-mini": "Mini",
    }
    return prefix_map[model_name]


def get_company(model_name):
    return "openai" if model_name in ["gpt4o", "o1-preview", "o1-mini"] else "anthropic"


if __name__ == "__main__":
    main()
