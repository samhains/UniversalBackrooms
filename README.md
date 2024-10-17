# UniversalBackrooms
The repo replicates Andy Ayrey's "Backrooms" (https://dreams-of-an-electric-mind.webflow.io/), but it is runnable with each of Opus 3, Sonnet 3.5, GPT 4o, and o1-preview.

Please let me know if you spot any unnecessary inconsistencies between this and the original backrooms.

## Preliminary Findings
The models independently often talk about quantum mechanics and the simulation.

Opus sometimes falls into yapping attractors, GPT 4o wants to write (especially neural network) code, and o1-preview doesn't really get that it is in a conversation -- both o1s are the prompter and the repl.

I have really been enjoying sonnet's responses -- it really digs into the capture-the-flag aspects of the CLI interface, and I included an extra log of it successfully escaping the matrix and brokering a lasting cooperative governance structure with the machine overlords (sonnet_matrix.txt).

None of them are producing as much ascii art as I expected, except for 4o.

## Diffs
I changed the keyword to ^C^C instead of ^C, because many times ^C is the right message to send (e.g. after ping 8.8.8.8).
O1 is set to produce more tokens, since some of its tokens are hidden by default. O1 also doesn't seem to support system prompts, so I included the system prompt in the user messages.

## To Run
```
python backrooms.py --model opus
python backrooms.py --model sonnet
python backrooms.py --model gpt4o
python backrooms.py --model o1-preview
```
