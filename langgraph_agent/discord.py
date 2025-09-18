"""Discord-specific helpers used by the agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DiscordMessage:
    """Payload describing a Discord post."""

    channel_id: str
    content: str
    media_url: Optional[str] = None

    def to_tool_arguments(self) -> dict[str, str]:
        payload = {
            "channel": self.channel_id,
            "message": self.content,
        }
        if self.media_url:
            payload["mediaUrl"] = self.media_url
        return payload


__all__ = ["DiscordMessage"]
