"""Minimal Discord interaction webhook server for prompt control.

Provides two commands:
- system: update the active system prompt for the requesting channel.
- gen: enqueue a generation request using the current system prompt and the supplied scene prompt.

State is tracked per Discord channel under data/discord_sessions/state.json so history survives restarts.
Generation requests are written to data/discord_jobs as JSON files for downstream processing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

LOGGER = logging.getLogger("discord_webhook")
logging.basicConfig(level=logging.INFO)

DEFAULT_TEMPLATE = "hours"
DEFAULT_SYSTEM_PROMPT = (
    "You are the After Hours Operator, a collaborative system prompt architect who keeps"
    " the creative energy flowing through the night. Set vivid constraints and guardrails"
    " when asked, and respond with short confirmations."
)
STATE_PATH = Path("data/discord_sessions/state.json")
QUEUE_PATH = Path("data/discord_jobs")
MAX_HISTORY = 50


@dataclass
class HistoryEntry:
    kind: str
    prompt: str
    user_id: str
    username: str
    timestamp: float
    message_id: Optional[str] = None


@dataclass
class SessionState:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    history: List[HistoryEntry] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "history": [asdict(entry) for entry in self.history],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        history = [HistoryEntry(**item) for item in data.get("history", [])]
        created_at = float(data.get("created_at", time.time()))
        updated_at = float(data.get("updated_at", created_at))
        system_prompt = data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        return cls(system_prompt=system_prompt, history=history, created_at=created_at, updated_at=updated_at)


class DiscordSessionStore:
    def __init__(self, path: Path, max_history: int = MAX_HISTORY) -> None:
        self.path = path
        self.max_history = max_history
        self._lock = asyncio.Lock()
        self._sessions: Dict[str, SessionState] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            LOGGER.warning("Failed to load session state: %s", exc)
            return
        sessions = payload.get("sessions", {})
        for channel_id, info in sessions.items():
            try:
                self._sessions[channel_id] = SessionState.from_dict(info)
            except Exception as exc:  # pragma: no cover - defensive load guard
                LOGGER.warning("Skipping corrupt session entry for %s: %s", channel_id, exc)

    async def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialised = {cid: session.to_dict() for cid, session in self._sessions.items()}
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump({"sessions": serialised}, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.path)

    async def set_system_prompt(self, channel_id: str, prompt: str) -> SessionState:
        async with self._lock:
            session = self._sessions.get(channel_id, SessionState())
            session.system_prompt = prompt.strip() or DEFAULT_SYSTEM_PROMPT
            session.history.clear()
            session.updated_at = time.time()
            self._sessions[channel_id] = session
            await self._persist()
            return session

    async def record_generation(
        self,
        channel_id: str,
        entry: HistoryEntry,
    ) -> SessionState:
        async with self._lock:
            session = self._sessions.get(channel_id, SessionState())
            session.history.append(entry)
            if len(session.history) > self.max_history:
                session.history = session.history[-self.max_history :]
            session.updated_at = time.time()
            self._sessions[channel_id] = session
            await self._persist()
            return session

    async def reset(self, channel_id: str) -> None:
        async with self._lock:
            if channel_id in self._sessions:
                del self._sessions[channel_id]
                await self._persist()

    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            return {cid: session.to_dict() for cid, session in self._sessions.items()}

    async def get_session(self, channel_id: str) -> SessionState:
        async with self._lock:
            return self._sessions.get(channel_id, SessionState())


class DiscordGenerationQueue:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        *,
        channel_id: str,
        guild_id: Optional[str],
        user: Dict[str, Any],
        session: SessionState,
        prompt: str,
        interaction: Dict[str, Any],
    ) -> Path:
        job_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        job_payload = {
            "job_id": job_id,
            "template": DEFAULT_TEMPLATE,
            "channel_id": channel_id,
            "guild_id": guild_id,
            "requested_by": user,
            "created_at": time.time(),
            "session": session.to_dict(),
            "vars": {
                "HOURS_SYSTEM_PROMPT": session.system_prompt,
                "HOURS_SCENE_PROMPT": prompt,
            },
            "interaction": {
                "id": interaction.get("id"),
                "token": interaction.get("token"),
            },
        }
        path = self.base_dir / f"{job_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(job_payload, f, indent=2, ensure_ascii=False)
        LOGGER.info("Queued Discord generation %s for channel %s", job_id, channel_id)
        return path


def verify_discord_signature(request: Request, body: bytes) -> None:
    public_key = os.getenv("DISCORD_PUBLIC_KEY")
    skip = os.getenv("DISCORD_SKIP_SIGNATURE_CHECK")
    if not public_key:
        if skip:
            LOGGER.warning("DISCORD_PUBLIC_KEY missing; skipping signature verification (development mode)")
            return
        raise HTTPException(status_code=500, detail="DISCORD_PUBLIC_KEY environment variable not set")

    signature = request.headers.get("X-Signature-Ed25519")
    timestamp = request.headers.get("X-Signature-Timestamp")
    if not signature or not timestamp:
        raise HTTPException(status_code=401, detail="Missing Discord signature headers")
    try:
        verify_key = VerifyKey(bytes.fromhex(public_key))
    except Exception as exc:  # pragma: no cover - configuration guard
        raise HTTPException(status_code=500, detail=f"Invalid DISCORD_PUBLIC_KEY: {exc}") from exc

    message = timestamp.encode() + body
    try:
        verify_key.verify(message, bytes.fromhex(signature))
    except BadSignatureError as exc:
        raise HTTPException(status_code=401, detail="Invalid request signature") from exc


def _options_to_dict(options: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    if not options:
        return {}
    out = {}
    for option in options:
        name = option.get("name")
        if not name:
            continue
        if option.get("type") in {1, 2}:  # subcommand or group
            out[name] = _options_to_dict(option.get("options"))
        else:
            out[name] = option.get("value")
    return out


def _format_response(content: str, *, ephemeral: bool = True) -> Dict[str, Any]:
    data: Dict[str, Any] = {"content": content}
    if ephemeral:
        data["flags"] = 1 << 6  # EPHEMERAL flag
    return {"type": 4, "data": data}


session_store = DiscordSessionStore(STATE_PATH)
generation_queue = DiscordGenerationQueue(QUEUE_PATH)
app = FastAPI(title="Discord Hours Webhook")


@app.get("/healthz")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/discord/sessions")
async def list_sessions() -> JSONResponse:
    snapshot = await session_store.snapshot()
    return JSONResponse(content=snapshot)


@app.post("/discord/interactions")
async def handle_interaction(request: Request) -> JSONResponse:
    raw_body = await request.body()
    verify_discord_signature(request, raw_body)
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc

    interaction_type = payload.get("type")
    if interaction_type == 1:  # PING
        return JSONResponse(content={"type": 1})

    if interaction_type != 2:  # APPLICATION_COMMAND
        raise HTTPException(status_code=400, detail="Unsupported interaction type")

    data = payload.get("data") or {}
    command_name = (data.get("name") or "").lower()
    options = _options_to_dict(data.get("options"))
    channel_id = payload.get("channel_id")
    user_info = payload.get("member", {}).get("user") or payload.get("user") or {}
    guild_id = payload.get("guild_id")
    if not channel_id:
        raise HTTPException(status_code=400, detail="Missing channel_id in interaction")

    if command_name == "system":
        prompt = (options.get("prompt") or options.get("value") or "").strip()
        if not prompt:
            return JSONResponse(content=_format_response("Please provide a system prompt.", ephemeral=True))
        session = await session_store.set_system_prompt(channel_id, prompt)
        content = (
            "️✅ Updated system prompt for this channel. History cleared.\n"
            f"Current system prompt:\n``{session.system_prompt}``"
        )
        return JSONResponse(content=_format_response(content, ephemeral=True))

    if command_name == "gen":
        prompt = (options.get("prompt") or options.get("scene") or options.get("value") or "").strip()
        if not prompt:
            return JSONResponse(content=_format_response("Please supply a generation prompt.", ephemeral=True))
        user = {
            "id": user_info.get("id"),
            "username": user_info.get("username"),
            "global_name": user_info.get("global_name"),
        }
        history_entry = HistoryEntry(
            kind="gen",
            prompt=prompt,
            user_id=user.get("id") or "unknown",
            username=user.get("username") or user.get("global_name") or "unknown",
            timestamp=time.time(),
            message_id=payload.get("id"),
        )
        session = await session_store.record_generation(channel_id, history_entry)
        job_path = generation_queue.enqueue(
            channel_id=channel_id,
            guild_id=guild_id,
            user=user,
            session=session,
            prompt=prompt,
            interaction={"id": payload.get("id"), "token": payload.get("token")},
        )
        content = (
            "🗂️ Queued a new Hours generation.\n"
            f"Prompt: **{prompt}**\n"
            f"System prompt snapshot: `{session.system_prompt[:180]}{'…' if len(session.system_prompt) > 180 else ''}`\n"
            f"Queue file: `{job_path.name}`"
        )
        return JSONResponse(content=_format_response(content, ephemeral=False))

    if command_name == "reset":
        await session_store.reset(channel_id)
        return JSONResponse(content=_format_response("Session reset for this channel. System prompt restored to default.", ephemeral=True))

    return JSONResponse(content=_format_response(f"Unknown command '{command_name}'.", ephemeral=True))


def create_app() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("discord_webhook_server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
