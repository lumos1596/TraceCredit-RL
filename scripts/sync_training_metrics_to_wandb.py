#!/usr/bin/env python3
"""Mirror the local console metrics of a running RL+OPD job to W&B."""

import argparse
import math
import os
import re
import time
from pathlib import Path

import wandb


ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"(?:^|\s)step:(\d+)\s+-\s+")


def parse_metrics(line):
    line = ANSI.sub("", line).strip()
    match = STEP.search(line)
    if not match:
        return None
    metrics = {}
    for field in line[match.end() :].split(" - "):
        key, sep, value = field.partition(":")
        if not sep:
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isfinite(number):
            metrics[key] = number
    if not metrics:
        return None
    return int(match.group(1)), metrics, line


def alive(pid):
    if not pid:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-id", default="tracecredit-opd-r9-20260926")
    parser.add_argument("--run-name")
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()

    run = wandb.init(
        project="Tree-GRPO",
        name=args.run_name or f"{args.experiment}-resume-step9",
        id=args.run_id,
        resume="allow",
        mode="online",
        dir=str(args.log.parent.parent),
        config={"checkpoint_step": 9, "source": "local console metrics"},
    )
    print(f"W&B run: {run.url}", flush=True)

    offset = 0
    inode = None
    pending = ""
    last_step = -1
    previous = {}
    idle_after_exit = 0
    try:
        while True:
            if args.log.is_file():
                stat = args.log.stat()
                if inode != stat.st_ino or stat.st_size < offset:
                    inode, offset, pending = stat.st_ino, 0, ""
                with args.log.open("r", encoding="utf-8", errors="replace") as source:
                    source.seek(offset)
                    block = source.read()
                    offset = source.tell()
                if block:
                    idle_after_exit = 0
                    pending += block
                    *lines, pending = pending.split("\n")
                    for line in lines:
                        parsed = parse_metrics(line)
                        if parsed is None:
                            continue
                        step, metrics, raw = parsed
                        if step <= last_step:
                            continue
                        for key in ("actor/ref_kl", "actor/policy_entropy"):
                            if key in metrics and key in previous:
                                metrics[f"monitor/{key.replace('/', '_')}_delta"] = metrics[key] - previous[key]
                            if key in metrics:
                                previous[key] = metrics[key]
                        metrics["training/console_line"] = raw
                        run.log(metrics, step=step)
                        last_step = step
                        print(f"synced step {step}: ref_kl={metrics.get('actor/ref_kl')} entropy={metrics.get('actor/policy_entropy')}", flush=True)
            if not alive(args.watch_pid):
                idle_after_exit += 1
                if idle_after_exit >= 2:
                    break
            time.sleep(args.poll_seconds)
    finally:
        if args.log.is_file():
            artifact = wandb.Artifact(f"{args.experiment}-console-log", type="training-log")
            artifact.add_file(str(args.log))
            run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
