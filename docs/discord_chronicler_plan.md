# Discord Chronicler Posting System — Plan

## Overview

Goal: Post generated images to Discord with an in-universe caption from a Chronicler/Archivist persona, then evolve into a bot that can also respond and occasionally post autonomously. Start minimally (sidecar metadata next to images) and iterate toward richer behavior (threads, memory, approval).

## Approach Options

1) Keep Folder Watcher (+Caption)
- Continue writing images to a watched folder; add a sidecar metadata file for caption and posting instructions.
- Watcher assembles both and posts to Discord.

2) Post Queue Service
- Replace/augment watcher with a small local service/CLI that accepts a “post” (image + caption + channel/thread) and pushes to Discord.
- Adds retries, dedupe, and logging.

3) Discord Bot Persona
- A bot participates in channels as the Chronicler/Archivist.
- Reacts to events (new image, mentions, timers), writes captions, and decides when to post.

4) Hybrid (Recommended)
- Phase 1 uses sidecar metadata for speed.
- Later phases add a bot with threads, context memory, and scheduling.

## Phased Plan (Recommended)

- Phase 1 — Sidecar Metadata (quick win)
  - Add caption and context next to the image; update watcher to read both and post in one message.

- Phase 2 — Robust Posting
  - Introduce a “post” abstraction (JSON schema + queue). Add retries, move posted assets, and log message IDs.

- Phase 3 — Threads & Structure
  - Create/target Discord threads per “expedition/level/session” for tidy narrative streams.

- Phase 4 — Chronicler Agent
  - Persona that (a) captions new images, (b) posts unprompted occasionally, (c) responds to mentions.

- Phase 5 — Review & Safety
  - Optional staging channel + approval, rate limiting, and fallback text if caption fails.

## Data Model (Sidecar/Post)

Core fields
- image_path: path to the generated image file.
- caption: diegetic text to accompany the image.
- channel_id or webhook: Discord destination.
- thread_id (optional): For posting into a specific thread.
- author: e.g., "Archivist-7".
- created_at: ISO timestamp.

Context snapshot (optional, for richer captions)
- level, sector, hazards, entities_seen, objective, timestamp, generator_prompt/seed, location_tags.

Control flags
- post_as_reply_to, visibility (staging/public), priority, allow_autogen_caption.

State (managed by uploader)
- status (queued | posting | posted | failed), discord_message_id, error.

Example (conceptual)
- image: renders/level_3/hallway_001.png
- caption: "Field Log 7. The hum is louder…"
- context: { level: 3, hazards: ["electrical", "mold"], entities_seen: ["Hounds?"] }
- post: { channel_id: ..., thread_id: optional }

## Caption Generation Strategies

Template + Variables (deterministic)
- Short forms like: "Field Log {n}: Level {level} — {hazards}. Observation: {detail}."

LLM Prompting (rich voice)
- Few-shot prompt for Chronicler/Archivist with constraints: concise, sensory, diegetic, avoid meta/spoilers.
- Inputs: context snapshot + image prompt summary + last 3 captions for continuity.

Variant Generation
- Produce 2–3 candidates (Field Note, Archivist Note, Redacted Snippet); select automatically or via quick human reaction vote.

Style Guardrails
- Voice: observational, methodical, slightly weary, clinically curious.
- Tense: present or recent past; avoid omniscience and memes (unless intentional).
- Include small catalog hooks (Specimen IDs, Sector codes) sparingly.

## Posting Mechanics

Watcher upgrade (Phase 1)
- Write image; atomically write `*.post.json` (temp then rename) with caption and routing.
- Watcher debounces; when both exist, validate and post (caption + attachment).
- On success, move files to `posted/YYYY-MM-DD/`; log message ID.

Bot vs Webhook
- Webhook: simplest; limited for threads/replies.
- Bot token: enables threads, replies, reading context; preferred from Phase 3 onward.

Threading scheme
- One thread per run/session/level; name like: `L3 // Corridor Survey // 2025-09-03`.
- Store `thread_id` in sidecar to target specific threads.

Rate limiting & retries
- Exponential backoff, content hash dedupe, size caps.

## Chronicler/Archivist Persona

Voice & Structure
- Observational, measured, diegetic; hints not answers.
- Suggested format:
  - Header: "Field Log n — Level {level}, Sector {sector}"
  - Body: 1–3 sentences of sensory detail + cautious inference.
  - Tail: small catalog line, e.g., "Ref: A7-L{level}-{hash[:6]}".

Memory
- Rolling window of last N posts per thread; maintain: current level, hazards, open questions, seen entities, location tags.

Triggers
- Event-driven: on new image generated.
- Reactive: on mentions/keywords.
- Periodic: every X minutes if thread active.

Autonomy controls
- Max posts/hour, quiet hours, kill switch, dry-run/stage mode.

## Moderation & Safety

- Staging-first: Post to a private channel; react with ✅ to promote.
- Content checks: simple keyword or LLM pass; fallback caption if tripped.
- Audit log: append-only log of posts with source path, caption, context, and result.

## Incremental Deliverables

Phase 1
- Define `*.post.json` sidecar contract.
- Update watcher to read caption, attach image, and post.
- Add atomic writes and `posted/` and `failed/` folders.

Phase 2
- Introduce a small queue (directory queue is fine) + retry/backoff and a JSONL log.
- Add Discord bot mode (optional) while keeping webhook fallback.

Phase 3
- Thread routing, naming, storing `thread_id`.
- Lightweight state cache per thread for continuity.

Phase 4
- Chronicler caption generation (template first, then LLM).
- Triggers (event + mention) and autonomy controls.

Phase 5
- Staging/approval, rate limits, metrics, and error alerts.

## Open Questions

1) What’s the current state source for level/sector/hazards? Do we persist this, or infer from image prompt/filenames?
2) Threads per run, per level, or a single channel chronology?
3) Webhook-only acceptable for now, or jump to a bot for threads/mentions?
4) Should the Chronicler ever generate images proactively, or only caption generated ones?
5) Prefer staging channel approval, or direct-to-public?
6) Any constraints on post frequency, size, or tone?

## Next Steps (No Code Yet)

- If approved, I’ll draft the `*.post.json` schema and a Chronicler style guide, plus a stepwise change list for the watcher/uploader.

