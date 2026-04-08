from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_project_env() -> None:
    """
    Load environment variables for the project CLI entrypoints.

    Preferred layout:
      - <repo>/.env

    Backward-compatible fallback:
      - <repo>/cgbv/.env

    Existing process environment variables always win.
    """
    repo_root = Path(__file__).resolve().parents[2]
    root_env = repo_root / ".env"
    legacy_env = repo_root / "cgbv" / ".env"

    if root_env.exists():
        load_dotenv(root_env, override=False)
    if legacy_env.exists():
        load_dotenv(legacy_env, override=False)
