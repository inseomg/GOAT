# goat_bench/utils/helpers.py
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict


def clear_screen():
    """
    Clear the terminal buffer using ANSI escape codes so Colab/REPLs do not print stray characters.
    Falls back to printing blank lines when ANSI is not supported.
    """
    seq = "\033[2J\033[H"
    try:
        print(seq, end="")
    except Exception:
        command = "cls" if os.name == "nt" else "clear"
        if os.system(command) != 0:
            print("\n" * 4)


def print_header(title: str, width: int = 54, fill: str = "="):
    """
    Print a centered header block used across CLI menus.
    """
    line = fill * width
    centered = title.strip().upper().center(width)
    print(line)
    print(centered)
    print(line)


def ensure_dir(path: Path) -> Path:
    """
    Ensure that `path` exists, creating directories as needed.
    Returns the path for convenience/chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


_EXIT_TRIGGERED = False
_EXIT_BUFFER = ""


def _check_exit_file() -> bool:
    flag = os.environ.get("GOAT_EXIT_FILE", ".goat_exit")
    if not flag:
        return False
    path = Path(flag)
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass
        return True
    return False


def _check_exit_stdin() -> bool:
    global _EXIT_BUFFER
    try:
        if os.name == "nt":
            import msvcrt

            triggered = False
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    if _EXIT_BUFFER.strip().lower() == "exit":
                        triggered = True
                    _EXIT_BUFFER = ""
                else:
                    _EXIT_BUFFER += ch
            return triggered
        else:
            import select

            if not sys.stdin or not sys.stdin.isatty():
                return False
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            for _ in ready:
                line = sys.stdin.readline()
                if line.strip().lower() == "exit":
                    return True
    except Exception:
        return False
    return False


def exit_requested() -> bool:
    """
    Non-blocking flag that becomes True if the user types 'exit' + Enter
    or creates a GOAT_EXIT_FILE (default: .goat_exit) while training.
    """
    global _EXIT_TRIGGERED
    if _EXIT_TRIGGERED:
        return True
    if _check_exit_file() or _check_exit_stdin():
        _EXIT_TRIGGERED = True
    return _EXIT_TRIGGERED


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file using UTF-8 encoding.
    Raises FileNotFoundError/JSONDecodeError if the file is invalid.
    """
    data = path.read_text(encoding="utf-8")
    return json.loads(data)


class ConsoleSpinner:
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, text: str = "작업 중"):
        self.text = text
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._percent = 0

    def _run(self):
        idx = 0
        while not self._stop.is_set():
            frame = self.FRAMES[idx % len(self.FRAMES)]
            sys.stdout.write(f"\r{self.text} {frame} {self._percent:3d}%")
            sys.stdout.flush()
            idx += 1
            if self._percent < 95:
                self._percent += 1
            time.sleep(0.12)
        sys.stdout.write(f"\r{self.text} ✔ 100%\n")
        sys.stdout.flush()

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._percent = 99
        self._stop.set()
        self._thread.join()
        self._thread = None
        self._stop.clear()
        self._percent = 0


__all__ = ["clear_screen", "print_header", "ensure_dir", "load_json", "exit_requested", "ConsoleSpinner"]
