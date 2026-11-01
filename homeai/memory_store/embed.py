# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Sentence embeddings on the CPU (ONNX), for memory search.

Why the CPU: the GPU is full in voice mode (MEMORY_PLAN §3: no VRAM for
memory). A small embedding model is a few milliseconds per sentence on the
9900X, inside the recall budget.

Each model needs three things done its own way, and getting any of them wrong
silently lowers recall quality rather than failing loudly:
* **pooling**: CLS token (BGE), mean of tokens (nomic, MiniLM), or a ready
  ``sentence_embedding`` output (EmbeddingGemma);
* **prefixes**: retrieval models are trained with different text before
  queries and documents;
* **dimension**: Matryoshka models may be truncated, then re-normalised.

``EMBED_SPECS`` holds those per model; ``tools/memory/embed_bench.py`` chose
the default.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from ..config import VENDOR

log = logging.getLogger(__name__)

EMBED_DIR = VENDOR / "embeddings"


@dataclass(frozen=True)
class EmbedSpec:
    name: str
    repo: str                 # Hugging Face repo with onnx/ and tokenizer.json
    onnx_file: str            # path inside the repo
    pooling: str              # cls | mean | output
    dim: int                  # stored dimension (Matryoshka models are truncated)
    query_prefix: str = ""
    doc_prefix: str = ""
    max_tokens: int = 256


EMBED_SPECS: dict[str, EmbedSpec] = {
    s.name: s for s in (
        EmbedSpec("bge-small-en-v1.5", "Xenova/bge-small-en-v1.5", "onnx/model.onnx", "cls", 384,
                  query_prefix="Represent this sentence for searching relevant passages: "),
        EmbedSpec("all-MiniLM-L6-v2", "Xenova/all-MiniLM-L6-v2", "onnx/model.onnx", "mean", 384),
        EmbedSpec("nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1.5", "onnx/model.onnx",
                  "mean", 384, query_prefix="search_query: ", doc_prefix="search_document: "),
        EmbedSpec("embeddinggemma-300m", "onnx-community/embeddinggemma-300m-ONNX",
                  "onnx/model.onnx", "output", 384,
                  query_prefix="task: search result | query: ", doc_prefix="title: none | text: "),
    )
}

# Chosen by tools/memory/embed_bench.py, 2026-11-01, 4 threads, with ~160 real
# transcript exchanges as distractors: nomic recall@5 0.98 / MRR 0.95 at
# 11 ms p95 per query; embeddinggemma 0.96 / 0.90; MiniLM 0.90 / 0.84;
# bge-small 0.80 / 0.65 (it was fine without distractors: long real
# exchanges swamp it). Fixtures were written by Claude, so treat as indicative.
DEFAULT_MODEL = "nomic-embed-text-v1.5"
# Every model above is stored at this width; the column is vector(384).
DIM = 384


class EmbedError(RuntimeError):
    pass


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str], kind: str = "doc") -> np.ndarray: ...


def model_dir(spec: EmbedSpec, root: Path = EMBED_DIR) -> Path:
    return root / spec.name


def download(spec: EmbedSpec, root: Path = EMBED_DIR) -> Path:
    """Fetch the model files once. Network; never called by the voice loop."""
    from huggingface_hub import hf_hub_download

    dest = model_dir(spec, root)
    for f in (spec.onnx_file, "tokenizer.json"):
        try:
            hf_hub_download(spec.repo, f, local_dir=dest)
        except Exception as exc:  # noqa: BLE001
            raise EmbedError(f"download {spec.repo}/{f}: {exc}") from exc
    try:
        # Large models keep their weights beside the graph; most have none.
        hf_hub_download(spec.repo, spec.onnx_file + "_data", local_dir=dest)
    except Exception:  # noqa: BLE001
        pass
    return dest


class OnnxEmbedder:
    """Thread-safe; load once per process."""

    def __init__(self, spec: EmbedSpec | str = DEFAULT_MODEL, root: Path = EMBED_DIR,
                 threads: int = 4) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.spec = EMBED_SPECS[spec] if isinstance(spec, str) else spec
        self.name = self.spec.name
        self.dim = self.spec.dim
        d = model_dir(self.spec, root)
        onnx_path, tok_path = d / self.spec.onnx_file, d / "tokenizer.json"
        if not onnx_path.exists() or not tok_path.exists():
            raise EmbedError(f"embedding model {self.name} missing in {d}; "
                             f"run tools/memory/embed_bench.py --download")
        self._tok = Tokenizer.from_file(str(tok_path))
        self._tok.enable_truncation(self.spec.max_tokens)
        self._tok.enable_padding()
        opts = ort.SessionOptions()
        # llama-server owns 12 cores; stay small so recall never starves it.
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(str(onnx_path), opts,
                                          providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self._outputs = [o.name for o in self._sess.get_outputs()]
        self._lock = threading.Lock()

    def embed(self, texts: Sequence[str], kind: str = "doc") -> np.ndarray:
        """Unit-length float32 vectors, shape (len(texts), dim)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefix = self.spec.query_prefix if kind == "query" else self.spec.doc_prefix
        enc = self._tok.encode_batch([prefix + (t or "") for t in texts])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        with self._lock:
            out = self._sess.run(None, feed)
        return pool(dict(zip(self._outputs, out)), mask, self.spec.pooling, self.dim)


def pool(outputs: dict[str, np.ndarray], mask: np.ndarray, how: str, dim: int) -> np.ndarray:
    """Pooling, Matryoshka truncation, normalisation. Pure, so it is unit-tested."""
    if how == "output":
        vec = outputs.get("sentence_embedding")
        if vec is None:
            raise EmbedError("model has no sentence_embedding output")
    else:
        hidden = outputs.get("last_hidden_state")
        if hidden is None:
            hidden = next(iter(outputs.values()))
        if how == "cls":
            vec = hidden[:, 0, :]
        elif how == "mean":
            m = mask[..., None].astype(np.float32)
            vec = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
        else:
            raise EmbedError(f"unknown pooling {how!r}")
    if vec.shape[1] < dim:
        raise EmbedError(f"model gives {vec.shape[1]} dims, spec wants {dim}")
    vec = vec[:, :dim].astype(np.float32)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.maximum(norms, 1e-12)


class HashEmbedder:
    """Deterministic bag-of-words vectors for tests: shared words, close vectors.

    Not a language model; it lets store and recall tests run without model
    files and with predictable neighbours.
    """

    name = "hash-test"

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str], kind: str = "doc") -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", (t or "").lower()):
                h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(), "big")
                out[i, h % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out
