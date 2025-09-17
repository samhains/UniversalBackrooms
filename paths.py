"""Centralized filesystem paths for runtime artifacts."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
VAR_DIR = ROOT / "var"
BACKROOMS_LOGS_DIR = VAR_DIR / "backrooms_logs"
TASKS_DIR = VAR_DIR / "tasks"
TASKS_DB_PATH = TASKS_DIR / "tasks.db"


def ensure_runtime_dirs() -> None:
    """Create runtime directories on import so scripts can rely on them."""
    for path in (BACKROOMS_LOGS_DIR, TASKS_DIR):
        path.mkdir(parents=True, exist_ok=True)


ensure_runtime_dirs()


def backrooms_log_path(*relative: str) -> Path:
    """Convenience helper to build paths inside the backrooms logs directory."""
    return BACKROOMS_LOGS_DIR.joinpath(*relative)
