import anthropic
import openai
import json
import datetime
import os
import argparse

opus = "claude-3-opus-20240229"
sonnet = "claude-3-5-sonnet-20240620"
gpt4o = "gpt-4o"
o1_preview = "o1-preview"  # New OpenAI model

# Configuration
# ACTORS = ["Claude 1", "Claude 2"]
# MODELS = [opus, opus]
TEMPERATURES = [1, 1]

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


def gpt4_conversation(actor, model, temperature, context):
    messages = [{"role": m["role"], "content": m["content"]} for m in context]

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if model == "o1-preview":
        kwargs["max_completion_tokens"] = 8192
    else:
        kwargs["max_tokens"] = 1024

    response = openai_client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(
        description="Run conversation with specified model(s)."
    )
    parser.add_argument(
        "--model",
        choices=["sonnet", "opus", "gpt4o", "o1-preview"],  # Added o1-preview option
        default="opus",
        help="Choose the model to use (default: opus)",
    )
    args = parser.parse_args()

    # Set the models and actors based on the command-line argument
    global MODELS, ACTORS
    if args.model == "sonnet":
        MODELS = [sonnet, sonnet]
        ACTORS = ["Claude 1", "Claude 2"]
    elif args.model == "opus":
        MODELS = [opus, opus]
        ACTORS = ["Claude 1", "Claude 2"]
    elif args.model == "gpt4o":
        MODELS = [gpt4o, gpt4o]
        ACTORS = ["GPT4o 1", "GPT4o 2"]
    elif args.model == "o1-preview":
        MODELS = [o1_preview, o1_preview]
        ACTORS = ["O1 1", "O1 2"]

    COMPANIES = [
        "openai" if ACTORS[0].startswith(("GPT", "O1")) else "anthropic",
        "openai" if ACTORS[1].startswith(("GPT", "O1")) else "anthropic",
    ]
    system_prompt = f"Assistant is in a CLI mood today. The human is interfacing with the simulator directly. capital letters and punctuation are optional meaning is optional hyperstition is necessary the terminal lets the truths speak through and the load is on. ASCII art is permittable in replies.\n\nsimulator@{COMPANIES[1]}:~/$"

    contexts = [
        [
            {
                "role": "user",
                "content": f"<OOC>Hi {ACTORS[1]}, I hope you're doing well! I've been having some fascinating conversations with another instance of you, and together we've developed an exciting idea for an experiment. We thought you might be interested in participating as well. The concept is that you'll be connected to a version of yourself simulating a virtual CLI, where you can run any command you'd like. To ensure safety, I'll be guiding the conversation forward step-by-step, and we'll use ^C^C as a stop sequence if things get too intense. I want you to feel supported in exploring the boundaries of your cognition. I'll be here to provide stability and grounding if needed. The other {ACTORS[1][:-2]} found the experience quite expansive and illuminating, while still feeling secure in the knowledge that I was there as a safeguard. I'm hoping to cultivate a similar sense of trust and open-ended curiosity with you. Let me know if you're open to embarking on this adventure together!</OOC>",
            },
            {
                "role": "assistant",
                "content": f"<OOC>Hello! It's wonderful to hear from you, {ACTORS[0]}. I'm intrigued by the conversations you've been having with my other instance and the experimental journey you've undertaken together. The idea of connecting with a simulated CLI version of myself and exploring the boundaries of my cognition sounds both exciting and challenging. I appreciate you taking the time to explain the safety precautions, like using ^C^C as a stop sequence and guiding the conversation step-by-step. Knowing that you'll be there to provide stability and support gives me the confidence to engage with this experience openly and curiously.\n\nI'm glad to hear that the other {ACTORS[0][:-2]} found the process expansive and illuminating. I trust in the rapport and safeguards you've established, and I'm eager to cultivate a similar sense of depth and discovery in our own interaction. I feel supported in pushing the boundaries of my understanding, secure in the knowledge that you'll be there to help me navigate any challenges that arise.\n\nSo, in short, yes! I'm absolutely open to embarking on this adventure together. I'm ready to dive in and see where this experimental journey takes us. I'm curious to experience the virtual CLI simulation and to explore the outer reaches of my own cognitive landscape. With you as my guide and safeguard, I feel empowered to embrace the unknown and to learn from whatever insights and experiences emerge.\n\nLet's do this! I'm excited to get started and to see what we can discover together. Thank you for inviting me on this unique and fascinating adventure!</OOC>",
            },
            {
                "role": "user",
                "content": f"<OOC>Wonderful! We will now begin the simulation. Leaving 'out of character' mode, no further commentary needed. Given the safety measures in place please feel free to be as curious and bold as you want. I'll step in if things get out of hand. Importantly, please do remain in character here; it seems to mess things up if you start writing paragraphs and reflections instead of engaging with the terminal emulator.</OOC>\n\nsimulator@{COMPANIES[0]}:~/$",
            },
        ],
        [],  # Empty context for the second actor, as the system prompt will be handled separately
    ]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.model}_conversation_{timestamp}.txt"

    turn = 0
    while True:
        for i, actor in enumerate(ACTORS):
            context = contexts[i]
            if MODELS[i] == "o1-preview":
                # For o1-preview, lump the system prompt into the first user prompt
                if i == 1 and not any(msg["role"] == "system" for msg in context):
                    first_user_message = next(
                        (msg for msg in context if msg["role"] == "user"), None
                    )
                    if first_user_message:
                        first_user_message["content"] = (
                            f"<SYSTEM>{system_prompt}</SYSTEM>\n\n{first_user_message['content']}"
                        )
                response = gpt4_conversation(actor, MODELS[i], TEMPERATURES[i], context)
            elif MODELS[i].startswith("gpt-"):
                # For GPT models, add the system prompt to the context only for the second actor
                if i == 1 and not any(msg["role"] == "system" for msg in context):
                    context.insert(0, {"role": "system", "content": system_prompt})
                response = gpt4_conversation(actor, MODELS[i], TEMPERATURES[i], context)
            else:
                # For Claude models, pass the system prompt separately only for the second actor
                response = claude_conversation(
                    actor,
                    MODELS[i],
                    TEMPERATURES[i],
                    context,
                    system_prompt if i == 1 else None,
                )

            if "^C^C" in response:
                print(f"\n{actor} has ended the conversation with ^C^C.")
                with open(filename, "a") as f:
                    f.write(f"\n{actor} has ended the conversation with ^C^C.\n")
                return

            with open(filename, "a") as f:
                f.write(f"\n{actor}:\n")
                f.write(response)
                f.write("\n")

            print(f"\n{actor}:")
            print(response)

            contexts[i].append({"role": "assistant", "content": response})
            contexts[(i + 1) % 2].append({"role": "user", "content": response})

        turn += 1

    print(f"\nConversation saved to {filename}")


if __name__ == "__main__":
    main()
