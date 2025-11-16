# goat_bench/utils/helpers.py
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict


def clear_screen():
    """
    Clear the current terminal screen.
    Uses shutil.get_terminal_size() indirectly via system calls where possible,
    but falls back to printing blank lines when the command fails.
    """
    command = "cls" if os.name == "nt" else "clear"
    # os.system return codes are ignored; fall back to newline spam on failure.
    if os.system(command) != 0:
        print("\n" * 4)


def print_header(title: str, width: int = 50, fill: str = "-"):
    """
    Print a simple header block used across CLI menus.
    """
    line = fill * width
    print(line)
    print(f" {title}")
    print(line)


def ensure_dir(path: Path) -> Path:
    """
    Ensure that `path` exists, creating directories as needed.
    Returns the path for convenience/chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file using UTF-8 encoding.
    Raises FileNotFoundError/JSONDecodeError if the file is invalid.
    """
    data = path.read_text(encoding="utf-8")
    return json.loads(data)


__all__ = ["clear_screen", "print_header", "ensure_dir", "load_json"]
