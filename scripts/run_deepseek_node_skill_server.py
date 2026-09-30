#!/usr/bin/env python3
"""Expose DeepSeek's chat API through the local NodeSkill analyzer protocol."""
from __future__ import annotations
import argparse, hashlib, json, os, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
LEGACY_FIELDS = {"failure_type", "missing_relation", "next_operation", "stop_condition"}
OPID_FIELDS = {"episode_summary", "episode_skill", "step_skills"}

class Analyzer:
    def __init__(self, args):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required")
        self.api_url, self.model = args.api_url, args.model
        self.timeout, self.max_tokens, self.retries = args.timeout, args.max_tokens, args.retries
        self.lock, self.cache = threading.Lock(), {}
        self.requests = self.cache_hits = 0

    def generate(self, prompt):
        key = hashlib.sha256(prompt.encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                self.cache_hits += 1
                return self.cache[key]
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": prompt}],
                           "temperature": 0, "max_tokens": self.max_tokens,
                           "response_format": {"type": "json_object"}}).encode()
        request = urllib.request.Request(self.api_url, data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST")
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode())
                skill = json.loads(payload["choices"][0]["message"]["content"])
                if not isinstance(skill, dict) or set(skill) not in (LEGACY_FIELDS, OPID_FIELDS):
                    raise ValueError("invalid analyzer schema")
                if set(skill) == LEGACY_FIELDS:
                    if not all(isinstance(skill[field], str) and skill[field].strip()
                               for field in LEGACY_FIELDS):
                        raise ValueError("NodeSkill fields must be non-empty strings")
                elif not (
                    isinstance(skill["episode_summary"], str)
                    and skill["episode_summary"].strip()
                    and isinstance(skill["episode_skill"], str)
                    and skill["episode_skill"].strip()
                    and isinstance(skill["step_skills"], dict)
                    and all(isinstance(key, str) and isinstance(value, str) and value.strip()
                            for key, value in skill["step_skills"].items())
                ):
                    raise ValueError("invalid OPID analyzer fields")
                with self.lock:
                    self.cache[key] = skill
                    self.requests += 1
                return skill
            except (KeyError, TypeError, ValueError, json.JSONDecodeError,
                    urllib.error.URLError, TimeoutError) as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 4))
        raise RuntimeError(f"DeepSeek request failed: {type(last_error).__name__}: {last_error}")

class Handler(BaseHTTPRequestHandler):
    analyzer = None
    def send_json(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, fmt, *args):
        print(f"[deepseek-node-skill] {fmt % args}", flush=True)
    def do_GET(self):
        if self.path.rstrip("/") != "/health":
            return self.send_json(404, {"error": "not found"})
        with self.analyzer.lock:
            status = {"ok": True, "model": self.analyzer.model, "requests": self.analyzer.requests,
                      "cache_hits": self.analyzer.cache_hits, "cache_entries": len(self.analyzer.cache)}
        self.send_json(200, status)
    def do_POST(self):
        if self.path.rstrip("/") != "/generate":
            return self.send_json(404, {"error": "not found"})
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode())
            prompt = str(payload.get("prompt", ""))
            if not prompt or len(prompt) > 100_000:
                raise ValueError("invalid prompt length")
            self.send_json(200, {"skill": self.analyzer.generate(prompt)})
        except Exception as error:
            self.send_json(502, {"error": f"{type(error).__name__}: {error}"})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8128)
    parser.add_argument("--api-url", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--model", default="deepseek-chat"); parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--max-tokens", type=int, default=256); parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args(); analyzer = Analyzer(args)
    handler = type("BoundHandler", (Handler,), {"analyzer": analyzer})
    print(f"[deepseek-node-skill] serving http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()
if __name__ == "__main__": main()
