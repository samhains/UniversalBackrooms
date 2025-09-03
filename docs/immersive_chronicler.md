# Immersive Chronicler + Media: Upgrade Notes

This note outlines two layers of changes to make the chronicler and media agents feel truly in‑world and first‑person, while fitting the current architecture.

## Lightweight Upgrades (drop‑in)

- Post intent gating: Have the discord preset output either `[POST] <message>` or `[HOLD] <reason>`. In `backrooms.py`, only call `run_discord_agent` when output begins with `[POST]`, stripping the tag before posting. This lets the agent decide when to speak.
- Actor alignment: Inject `{last_actor}` into the discord preset and nudge: “Assume ‘I’ is the actor most likely to continue after `{last_actor}` unless context strongly suggests otherwise.”
- Style continuity memory: Pass the last 1–2 dispatched messages (from Discord chronicler) back into the chronicler prompt as “voice memory” to stabilize tone, lexicon, and pacing.
- Media pairing (soft reference): If an image is generated this round (`media_url`), mention “what I see now” without links/meta; let the chronicler optionally allude to the subject in-world.
- Stricter POV constraints: Add negative constraints to the system prompt (“no summaries, no meta, no apologies, no ‘as an AI’”). Keep 1–3 sentences, present tense, sensory detail.
- Fail‑safe caps: Enforce a character limit and strip code blocks/markdown that might slip in.

Suggested minimal logic (pseudo):

```python
# after producing candidate text from discord preset
raw = candidate.strip()
if raw.startswith('[POST]'):
    post = raw[len('[POST]'):].strip()
    # post via run_discord_agent with post
else:
    # skip posting this round
```

## Deeper Immersion Ideas (architectural)

- Dedicated Chronicler agent (participant): Add a third agent to the conversation template that role‑plays the in‑world chronicler and emits tool calls by protocol (e.g., `[TOOL:discord.send-message]{json}`), which the orchestrator parses and executes. The chronicler truly “makes” the post.
- Inner/outer channels: Allow the chronicler to produce both an internal monologue (kept in transcript) and a public utterance (posted). The monologue conditions style and memory without leaking to Discord.
- World‑state affordances: Provide a compact state bundle each round (location, light, sound, threat, objective, injury). The chronicler references these as immediate sensations, improving continuity without meta.
- Media‑driven feedback: Pipe the generated image reference (URL or short description) back as “what I see now” context so the chronicler can anchor prose to visuals.
- Turn control + rate limiting: Let the chronicler request to skip rounds or batch events (e.g., `[HOLD n=2]`) to avoid over‑posting and preserve pacing.
- Persona memory: Persist short‑term “voice DNA” (idioms, fears, motifs) to a small file and feed it as part of the chronicler’s prompt window.
- Tool‑belief loop: Treat successful tool executions as world events the chronicler can feel (e.g., posting increases “signal noise,” opening doors changes environment). Feed a minimal event log back into context.

## Where to Hook (current code)

- Post gating: `backrooms.py` around the Discord agent call (look for `run_discord_agent(...)`). Intercept the candidate text from the discord preset before invoking the MCP tool.
- Actor alignment: Extend the discord preset `user_template` with `{last_actor}` and update `discord_agent.py` prompt assembly (it already exposes `last_actor`/`last_text`).
- Media pairing: This is already partially wired; `media_url` is passed into `run_discord_agent`. Nudge the preset system prompt to reference visuals without links/meta.
- Persona memory: Store last 1–2 posted lines to a small file in `BackroomsLogs/` and thread them into the chronicler preset as an extra field (e.g., `{voice_memory}`).

## Testing Checklist

- First‑person voice stays present‑tense and avoids meta.
- `[POST]/[HOLD]` logic prevents accidental empty or low‑confidence posts.
- Chronicler’s tone remains stable across 3–5 rounds.
- Media prompt references are POV/scene‑anchored, never meta.

