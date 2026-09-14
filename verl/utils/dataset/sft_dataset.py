"""Supervised fine-tuning dataset for chat-style parquet data.

The dataset keeps the chat-template tokenization and the supervision mask in
one place.  In particular, the mask is obtained from the token span added by
each cumulative chat prefix instead of relying on tokenizing an assistant
message in isolation (which is not generally safe for chat templates).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


_ASSISTANT_ROLES = {"assistant", "bot", "model"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [value]
    return list(value)


def _read_parquet(paths: Sequence[str | os.PathLike[str]]) -> list[dict[str, Any]]:
    """Read one or more parquet files without imposing a dataframe dependency."""
    paths = [str(path) for path in paths]
    if not paths:
        return []

    try:
        import pyarrow.parquet as pq

        rows: list[dict[str, Any]] = []
        for path in paths:
            rows.extend(pq.read_table(path).to_pylist())
        return rows
    except ImportError:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Reading parquet requires pyarrow or pandas") from exc
        rows = []
        for path in paths:
            rows.extend(pd.read_parquet(path).to_dict(orient="records"))
        return rows


def _get_nested(value: Any, keys: Sequence[str] | str | None) -> Any:
    """Get a possibly nested value, accepting either ``a.b`` or key lists."""
    if keys is None:
        return value
    if isinstance(keys, str):
        keys = keys.split(".")
    for key in keys:
        if isinstance(value, Mapping):
            value = value[key]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            value = value[int(key)]
        else:
            raise KeyError(key)
    return value


def _decode_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return value


def _coerce_messages(value: Any) -> list[dict[str, Any]]:
    value = _decode_json(value)
    if isinstance(value, Mapping):
        if "messages" in value:
            value = _decode_json(value["messages"])
        else:
            value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("messages must be a list of message dictionaries")

    messages = []
    for message in value:
        message = _decode_json(message)
        if not isinstance(message, Mapping):
            raise TypeError("each chat message must be a dictionary")
        if "role" not in message or "content" not in message:
            raise ValueError("each chat message needs role and content")
        normalized = dict(message)
        normalized["role"] = str(message["role"])
        messages.append(normalized)
    return messages


class SFTDataset(Dataset):
    """Chat SFT examples loaded from parquet files.

    ``messages`` is preferred and is expected to contain a list of Qwen chat
    messages.  The prompt/response arguments are retained for compatibility
    with existing FSDP trainer call sites; when no ``messages`` column exists,
    they are used to construct a system/user/assistant conversation.
    """

    def __init__(
        self,
        parquet_files: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        tokenizer: Any,
        prompt_key: str = "prompt",
        prompt_dict_keys: Sequence[str] | str | None = None,
        response_key: str = "response",
        response_dict_keys: Sequence[str] | str | None = None,
        max_length: int = 4096,
        truncation: str = "right",
        **_: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.prompt_dict_keys = prompt_dict_keys
        self.response_key = response_key
        self.response_dict_keys = response_dict_keys
        self.max_length = int(max_length)
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        truncation = str(truncation).lower()
        if truncation in {"right", "longest_first", "true"}:
            self.truncation = "right"
        elif truncation in {"error", "none", "false"}:
            self.truncation = "error"
        else:
            raise ValueError("truncation must be 'right' or 'error'")

        paths = _as_list(parquet_files)
        self.data = _read_parquet(paths)
        if not self.data:
            raise ValueError("no examples found in parquet_files")

    def __len__(self) -> int:
        return len(self.data)

    def _messages(self, row: Mapping[str, Any]) -> list[dict[str, Any]]:
        if "messages" in row and row["messages"] is not None:
            return _coerce_messages(row["messages"])

        if self.prompt_key in row:
            prompt_value = _decode_json(_get_nested(row[self.prompt_key], self.prompt_dict_keys))
            if isinstance(prompt_value, Sequence) and not isinstance(prompt_value, (str, bytes)):
                if all(isinstance(item, Mapping) and "role" in item for item in prompt_value):
                    return _coerce_messages(prompt_value)

        prompt = _get_nested(row[self.prompt_key], self.prompt_dict_keys)
        if self.response_key is None:
            raise ValueError(
                f"row has no messages column and response_key is None; "
                f"cannot build an assistant response from {self.prompt_key!r}"
            )
        response = _get_nested(row[self.response_key], self.response_dict_keys)
        prompt = _decode_json(prompt)
        response = _decode_json(response)
        messages: list[dict[str, Any]] = []
        if isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)):
            messages.extend(_coerce_messages(prompt))
        else:
            messages.append({"role": "user", "content": str(prompt)})
        if isinstance(response, Sequence) and not isinstance(response, (str, bytes)):
            messages.extend(_coerce_messages(response))
        else:
            messages.append({"role": "assistant", "content": str(response)})
        return messages

    def _template_ids(self, messages: list[dict[str, Any]]) -> list[list[int]]:
        """Return cumulative prefix tokenizations, with no generation prompt."""
        prefixes = []
        for end in range(1, len(messages) + 1):
            encoded = self.tokenizer.apply_chat_template(
                messages[:end],
                tokenize=True,
                add_generation_prompt=False,
            )
            if isinstance(encoded, Mapping):
                encoded = encoded["input_ids"]
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.tolist()
            if encoded and isinstance(encoded[0], list):
                encoded = encoded[0]
            prefixes.append([int(token) for token in encoded])
        return prefixes

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        messages = self._messages(self.data[index])
        prefixes = self._template_ids(messages)
        if not prefixes:
            raise ValueError("chat example has no messages")

        input_ids = prefixes[-1]
        loss_mask = [0] * len(input_ids)
        previous_length = 0
        for message, prefix in zip(messages, prefixes):
            current_length = len(prefix)
            if message["role"].lower() in _ASSISTANT_ROLES:
                start = min(previous_length, current_length)
                loss_mask[start:current_length] = [1] * max(0, current_length - start)
            previous_length = current_length

        if not any(loss_mask):
            raise ValueError("chat example contains no assistant tokens to supervise")

        if len(input_ids) > self.max_length:
            if self.truncation == "error":
                raise ValueError(
                    f"example at index {index} has {len(input_ids)} tokens, "
                    f"exceeding max_length={self.max_length}"
                )
            input_ids = input_ids[-self.max_length :]
            loss_mask = loss_mask[-self.max_length :]

        if not any(loss_mask):
            raise ValueError(
                f"example at index {index} has no assistant tokens after "
                f"truncation to max_length={self.max_length}"
            )

        pad_length = self.max_length - len(input_ids)
        if pad_length:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            if pad_token_id is None:
                raise ValueError("tokenizer needs pad_token_id or eos_token_id")
            input_ids = input_ids + [int(pad_token_id)] * pad_length
            loss_mask = loss_mask + [0] * pad_length

        ids = torch.tensor(input_ids, dtype=torch.long)
        mask = torch.tensor(loss_mask, dtype=torch.long)
        attention_mask = torch.ones_like(ids)
        if pad_length:
            attention_mask[-pad_length:] = 0
        position_ids = torch.arange(ids.numel(), dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": mask,
        }


if __name__ == "__main__":  # pragma: no cover - executable smoke test
    import tempfile

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(f"self-test dependencies unavailable: {exc}")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sft.parquet"
        table = pa.table(
            {
                "messages": [
                    json.dumps(
                        [
                            {"role": "system", "content": "You are helpful."},
                            {"role": "user", "content": "Say hello."},
                            {"role": "assistant", "content": "Hello!"},
                        ]
                    )
                ]
            }
        )
        pq.write_table(table, path)
        tokenizer = AutoTokenizer.from_pretrained("models/Qwen2.5-3B-Instruct")
        sample = SFTDataset(path, tokenizer, max_length=128)[0]
        assert all(sample[key].ndim == 1 for key in sample)
        assert sample["loss_mask"].sum().item() > 0
        print("SFTDataset self-test passed")
