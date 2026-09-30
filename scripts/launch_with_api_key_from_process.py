#!/usr/bin/env python3
"""Exec a command using DEEPSEEK_API_KEY inherited from another same-user process."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        raise SystemExit("a command is required")
    prefix = b"DEEPSEEK_API_KEY="
    value = next((item[len(prefix):] for item in Path(f"/proc/{args.pid}/environ").read_bytes().split(b"\0")
                  if item.startswith(prefix)), None)
    if value is None:
        raise SystemExit("DEEPSEEK_API_KEY is unavailable")
    environment = os.environ.copy()
    environment["DEEPSEEK_API_KEY"] = value.decode()
    os.execvpe(args.command[0], args.command, environment)


if __name__ == "__main__":
    main()
