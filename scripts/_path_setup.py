"""Utility to ensure the repository's src directory is importable."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path() -> None:
    """Prepend the repository's ``src`` directory to ``sys.path`` if needed."""

    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


ensure_src_on_path()
