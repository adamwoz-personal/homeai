# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Embedding pooling and the test embedder (memory_store/embed.py)."""

from __future__ import annotations

import numpy as np
import pytest

from homeai.memory_store import embed
from homeai.memory_store.embed import DIM, EMBED_SPECS, EmbedError, HashEmbedder, pool


def hidden():
    # batch 2, 3 tokens, 4 dims; second sequence has one padding token
    h = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    mask = np.array([[1, 1, 1], [1, 1, 0]])
    return {"last_hidden_state": h}, mask, h


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_cls_pooling_takes_first_token():
    out, mask, h = hidden()
    np.testing.assert_allclose(pool(out, mask, "cls", 4), unit(h[:, 0, :]), rtol=1e-6)


def test_mean_pooling_ignores_padding():
    out, mask, h = hidden()
    expect = np.stack([h[0].mean(axis=0), h[1, :2].mean(axis=0)])
    np.testing.assert_allclose(pool(out, mask, "mean", 4), unit(expect), rtol=1e-6)


def test_output_pooling_uses_sentence_embedding():
    v = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(pool({"sentence_embedding": v}, None, "output", 3), [[0.6, 0.8, 0.0]])
    with pytest.raises(EmbedError):
        pool({"last_hidden_state": v}, None, "output", 3)


def test_matryoshka_truncation_then_normalise():
    v = np.array([[3.0, 4.0, 12.0]], dtype=np.float32)
    out = pool({"sentence_embedding": v}, None, "output", 2)
    np.testing.assert_allclose(out, [[0.6, 0.8]], rtol=1e-6)
    assert out.dtype == np.float32


def test_too_few_dims_and_bad_pooling():
    out, mask, _ = hidden()
    with pytest.raises(EmbedError):
        pool(out, mask, "mean", 5)
    with pytest.raises(EmbedError):
        pool(out, mask, "max", 4)


def test_zero_vector_does_not_divide_by_zero():
    z = np.zeros((1, 3), dtype=np.float32)
    assert np.isfinite(pool({"sentence_embedding": z}, None, "output", 3)).all()


def test_every_spec_stores_the_column_width():
    assert DIM == 384
    assert all(s.dim == DIM for s in EMBED_SPECS.values())
    assert embed.DEFAULT_MODEL in EMBED_SPECS


def test_hash_embedder_neighbours_share_words():
    e = HashEmbedder()
    a, b, c = e.embed(["corn snake care", "caring for a corn snake", "garage door opener"])
    assert a @ b > a @ c
    assert np.linalg.norm(a) == pytest.approx(1.0)
    assert e.embed([]).shape == (0, DIM)
    assert not e.embed([""]).any()


def test_missing_model_files_raise(tmp_path):
    pytest.importorskip("tokenizers")
    with pytest.raises(EmbedError, match="missing"):
        embed.OnnxEmbedder(embed.DEFAULT_MODEL, root=tmp_path)


def test_real_model_if_present():
    spec = EMBED_SPECS[embed.DEFAULT_MODEL]
    if not (embed.model_dir(spec) / spec.onnx_file).exists():
        pytest.skip("embedding model not downloaded")
    e = embed.OnnxEmbedder()
    q = e.embed(["what's Ryan's favourite snake"], kind="query")
    d = e.embed(["Ryan's favourite animal is the king cobra", "The garage door is broken"])
    assert q.shape == (1, DIM)
    assert (q @ d.T)[0, 0] > (q @ d.T)[0, 1]
