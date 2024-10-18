import anthropic
import openai
import json
import datetime
import os
import argparse

TEMPERATURES = [1.0, 1.0]

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic()

# Initialize OpenAI client
openai_client = openai.OpenAI()

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


def load_template(template_name, lm_display_names, lm_companies):
    try:
        with open(f"templates/{template_name}.json", "r") as f:
            template = json.load(f)

        # Replace placeholders in system prompts
        for system_prompt in ["system_prompt1", "system_prompt2"]:
            template[system_prompt] = template[system_prompt].format(
                lm1_company=lm_companies[0],
                lm2_company=lm_companies[1],
                lm1_actor=lm_display_names[0],
                lm2_actor=lm_display_names[1],
            )
        # Replace placeholders in context messages
        for context in ["lm1_context", "lm2_context"]:
            for message in template[context]:
                message["content"] = message["content"].format(
                    lm1_company=lm_companies[0],
                    lm2_company=lm_companies[1],
                    lm1_actor=lm_display_names[0],
                    lm2_actor=lm_display_names[1],
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
        description="Run conversation between two AI language models."
    )
    parser.add_argument(
        "--lm1",
        choices=["sonnet", "opus", "gpt4o", "o1-preview", "o1-mini"],
        default="opus",
        help="Choose the model for LM1 (default: opus)",
    )
    parser.add_argument(
        "--lm2",
        choices=["sonnet", "opus", "gpt4o", "o1-preview", "o1-mini"],
        default="opus",
        help="Choose the model for LM2 (default: opus)",
    )
    parser.add_argument(
        "--template",
        choices=[
            "cli",
            "science",
            "fugue",
            "gallery",
            "student",
            "ethics",
        ],
        default="cli",
        help="Choose a conversation template (default: cli)",
    )
    args = parser.parse_args()

    lm_models = [MODEL_INFO[args.lm1]["api_name"], MODEL_INFO[args.lm2]["api_name"]]
    lm_display_names = [
        f"{MODEL_INFO[args.lm1]['display_name']} 1",
        f"{MODEL_INFO[args.lm2]['display_name']} 2",
    ]
    lm_companies = [MODEL_INFO[args.lm1]["company"], MODEL_INFO[args.lm2]["company"]]

    template = load_template(args.template, lm_display_names, lm_companies)
    system_prompts = [template["system_prompt1"], template["system_prompt2"]]
    contexts = [template["lm1_context"], template["lm2_context"]]

    lm1_folder = f"BackroomsLogs/{args.lm1.capitalize()}Explorations"
    if not os.path.exists(lm1_folder):
        os.makedirs(lm1_folder)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{lm1_folder}/{args.lm1}_{args.lm2}_{args.template}_{timestamp}.txt"

    turn = 0
    while True:
        for i in range(2):
            lm_response = handle_conversation(
                lm_models[i],
                lm_display_names[i],
                TEMPERATURES[i],
                contexts[i],
                system_prompts[i],
            )
            handle_response(
                lm_response,
                lm_display_names[i],
                filename,
                contexts[i],
                contexts[1 - i],
            )

        turn += 1


def handle_conversation(model, actor, temperature, context, system_prompt):
    if model.startswith("claude-"):
        return claude_conversation(
            actor, model, temperature, context, system_prompt if system_prompt else None
        )
    else:
        return gpt4_conversation(
            actor, model, temperature, context, system_prompt if system_prompt else None
        )


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


if __name__ == "__main__":
    main()
