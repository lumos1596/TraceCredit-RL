#!/usr/bin/env python3
"""Serve a frozen NodeSkill analyzer over a small local HTTP API."""

from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FIELDS = {"failure_type", "missing_relation", "next_operation", "stop_condition"}


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("model output contains no JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("invalid NodeSkill schema")
    return value


class Analyzer:
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(args.device).eval()
        self.device = args.device
        self.max_input = args.max_input
        self.max_new_tokens = args.max_new_tokens
        self.lock = threading.Lock()
        self.requests = 0

    def generate(self, prompt: str) -> dict:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_input
        ).to(self.device)
        with self.lock, torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(
            output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        self.requests += 1
        return extract_json(text)


class Handler(BaseHTTPRequestHandler):
    analyzer: Analyzer = None

    def _json(self, status: int, value: dict):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[node-skill] {fmt % args}", flush=True)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._json(200, {"ok": True, "requests": self.analyzer.requests})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/generate":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = str(payload.get("prompt", ""))
            if not prompt or len(prompt) > 100_000:
                raise ValueError("invalid prompt length")
            self._json(200, {"skill": self.analyzer.generate(prompt)})
        except Exception as error:
            self._json(400, {"error": f"{type(error).__name__}: {error}"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8127)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-input", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    analyzer = Analyzer(args)
    handler = type("BoundHandler", (Handler,), {"analyzer": analyzer})
    print(f"[node-skill] serving http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
