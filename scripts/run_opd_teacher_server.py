#!/usr/bin/env python3
"""Privileged-teacher scoring server for Self-OPD (answer-conditioned 7B).

The student trainer composes a privileged teacher context on its own side
(original prompt + parent prefix + gold-answer cheat sheet + the policy's
query tokens) and ships the *unpadded* token ids plus the query positions to
this server.  The server only runs the forward and returns per-position,
temperature-adjusted ``log_softmax`` rows, which keeps the student's JSD and
directional-lift diagnostics bit-compatible with the legacy local-teacher
path (see dp_actor._fetch_remote_teacher_log_probs for the wire format).

Run on a dedicated GPU, e.g.:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_opd_teacher_server.py \
        --model models/Qwen2.5-7B-Instruct --port 8126
"""

from __future__ import annotations

import argparse
import json
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import transformers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/Qwen2.5-7B-Instruct',
                        help='HF-format teacher model directory')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8126)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-batch-tokens', type=int, default=16384,
                        help='cap on padded tokens (rows x max_len) per forward')
    parser.add_argument('--max-row-length', type=int, default=8192,
                        help='reject rows longer than this many tokens')
    parser.add_argument('--vocab-size', type=int, default=0,
                        help='truncate/renormalize log-probs to the first N vocab '
                             'columns (0 = keep the teacher model\'s full vocab). '
                             'Required when the student model has a smaller vocab '
                             'than the teacher, e.g. Qwen2.5 3B (151936) vs 7B (152064).')
    return parser


class TeacherServer:
    def __init__(self, args):
        print(f'[teacher] loading {args.model} on {args.device} ...', flush=True)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa'
        ).to(args.device).eval()
        self.device = args.device
        self.max_batch_tokens = int(args.max_batch_tokens)
        self.max_row_length = int(args.max_row_length)
        self.full_vocab_size = int(self.model.config.vocab_size)
        self.vocab_size = int(args.vocab_size) or self.full_vocab_size
        if not 0 < self.vocab_size <= self.full_vocab_size:
            raise ValueError(
                f'--vocab-size must be in (0, {self.full_vocab_size}]'
            )
        self.forward_lock = threading.Lock()
        self.request_count = 0
        print(
            f'[teacher] ready vocab={self.vocab_size} '
            f'(full={self.full_vocab_size}, truncated={self.vocab_size < self.full_vocab_size})',
            flush=True,
        )

    def score(self, header, ids_flat):
        temperature = float(header.get('temperature', 1.0))
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        rows = header.get('rows') or []
        expected = sum(int(row['len']) for row in rows)
        if ids_flat.size != expected:
            raise ValueError(f'id payload {ids_flat.size} != header {expected}')

        sequences = []
        cursor = 0
        for row in rows:
            length = int(row['len'])
            positions = [int(p) for p in row.get('query_positions', [])]
            if length <= 0:
                raise ValueError('row length must be positive')
            if length > self.max_row_length:
                raise ValueError(f'row length {length} exceeds {self.max_row_length}')
            for position in positions:
                if position < 0 or position >= length:
                    raise ValueError(
                        f'query position {position} out of range for row of {length}'
                    )
            sequences.append((ids_flat[cursor:cursor + length].copy(), positions))
            cursor += length

        results = []
        pad_id = int(self.tokenizer.pad_token_id)
        batch = []
        batch_max_len = 0

        def flush_batch():
            nonlocal batch, batch_max_len
            if not batch:
                return
            max_len = max(len(ids) for ids, _ in batch)
            group = len(batch)
            input_ids = torch.full((group, max_len), pad_id, dtype=torch.long)
            attention = torch.zeros((group, max_len), dtype=torch.long)
            for row_index, (ids, _) in enumerate(batch):
                input_ids[row_index, max_len - len(ids):] = torch.from_numpy(ids)
                attention[row_index, max_len - len(ids):] = 1
            position_ids = (attention.cumsum(-1) - 1).clamp_min(0) * attention
            # Think+search supervision positions may be non-contiguous because
            # control tags remain in the causal context. Keep the tail starting
            # at the earliest requested position, rather than assuming q
            # requested rows occupy the last q logits.
            keep = min(
                max(
                    (len(ids) - min(positions) for ids, positions in batch if positions),
                    default=1,
                ),
                max_len,
            )
            keep_start = max_len - keep
            with self.forward_lock, torch.inference_mode():
                logits = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention.to(self.device),
                    position_ids=position_ids.to(self.device),
                    use_cache=False,
                    logits_to_keep=keep,
                ).logits
            for row_index, (ids, positions) in enumerate(batch):
                if not positions:
                    results.append(np.zeros((0, self.vocab_size), dtype=np.float32))
                    continue
                # Unpadded position p -> padded coordinate p + (max_len - L),
                # then into the kept tail window starting at keep_start.
                window_offsets = [
                    p + (max_len - len(ids)) - keep_start for p in positions
                ]
                for offset in window_offsets:
                    if offset < 0 or offset >= keep:
                        raise ValueError(
                            f'query window offset {offset} outside [0, {keep}); '
                            'row length or positions are inconsistent'
                        )
                window_positions = torch.as_tensor(
                    window_offsets, dtype=torch.long, device=logits.device
                )
                # Index the position rows first, then log_softmax: gathering
                # after log_softmax with a (q, 1) index silently reads rows
                # 0..q-1 instead of the requested positions.
                row_logits = logits[row_index][window_positions]
                row_log_probs = torch.log_softmax(
                    row_logits.float() / temperature, dim=-1
                )
                if self.vocab_size < self.full_vocab_size:
                    # Restrict the teacher to the student's vocabulary: slice
                    # then renormalize so JSD shapes line up exactly.
                    kept = row_log_probs[:, :self.vocab_size]
                    row_log_probs = kept - torch.logsumexp(
                        kept, dim=-1, keepdim=True
                    )
                results.append(row_log_probs.cpu().numpy().astype(np.float32))
            batch = []
            batch_max_len = 0

        for ids, positions in sequences:
            projected_max_len = max(batch_max_len, len(ids))
            if batch and (len(batch) + 1) * projected_max_len > self.max_batch_tokens:
                flush_batch()
            batch.append((ids, positions))
            batch_max_len = max(batch_max_len, len(ids))
        flush_batch()

        if len(results) != len(sequences):
            raise RuntimeError('internal batching produced wrong row count')
        return results, sequences


class Handler(BaseHTTPRequestHandler):
    server_model: TeacherServer = None  # injected

    def log_message(self, fmt, *args):  # route access logs through print
        print(f'[teacher] {self.address_string()} {fmt % args}', flush=True)

    def _send_payload(self, header: dict, payload: bytes, status: int = 200):
        header_bytes = json.dumps(header).encode('utf-8')
        body = struct.pack('<I', len(header_bytes)) + header_bytes + payload
        self.send_response(status)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip('/') == '/health':
            body = json.dumps({
                'ok': True,
                'vocab': self.server_model.vocab_size,
                'requests': self.server_model.request_count,
            }).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.rstrip('/') != '/score':
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length)
            if len(raw) < 4:
                raise ValueError('truncated request')
            (header_len,) = struct.unpack('<I', raw[:4])
            header = json.loads(raw[4:4 + header_len])
            ids_flat = np.frombuffer(raw, dtype=np.int64, offset=4 + header_len)
            if ids_flat.size and (int(ids_flat.min()) < 0
                                  or int(ids_flat.max()) >= self.server_model.full_vocab_size):
                raise ValueError('token id out of teacher vocabulary')
            started = time.time()
            results, _ = self.server_model.score(header, ids_flat)
            elapsed = time.time() - started
            self.server_model.request_count += 1
            payload = b''.join(result.tobytes() for result in results)
            header_out = {
                'dtype': 'float32',
                'vocab': self.server_model.vocab_size,
                'row_lengths': [int(result.shape[0]) for result in results],
            }
            self._send_payload(header_out, payload)
            print(
                f'[teacher] score rows={len(results)} '
                f'tokens={int(ids_flat.size)} bytes={len(payload)} '
                f'seconds={elapsed:.3f}',
                flush=True,
            )
        except Exception as error:  # surface the failure to the trainer
            print(f'[teacher] ERROR {type(error).__name__}: {error}', flush=True)
            try:
                self._send_payload({'error': f'{type(error).__name__}: {error}'}, b'', status=400)
            except Exception:
                pass


def main():
    args = build_parser().parse_args()
    server_model = TeacherServer(args)
    handler = type('BoundHandler', (Handler,), {'server_model': server_model})
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f'[teacher] serving on http://{args.host}:{args.port}/score', flush=True)
    httpd.serve_forever()


if __name__ == '__main__':
    main()
