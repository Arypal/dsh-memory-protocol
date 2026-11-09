"""
Memory Skill — ONNX Embedder

Provides the ``Embedder`` class for generating text embeddings using
ONNX Runtime with the ``bge-large-en-v1.5`` model (1024-dim).

Key design:
- Lazy-load: the ONNX model is loaded on the first ``embed()`` call.
- Fallback: if the ONNX model cannot be loaded (missing file, missing runtime),
  a deterministic SHA-256–based fallback is used (test-safe).
- Caching: the ONNX session and tokenizer are cached in-memory after first load.

Usage::

    from memory_skill.contracts import MemorySkillConfig
    from memory_skill.embedder import Embedder

    cfg = MemorySkillConfig()
    emb = Embedder(cfg)
    vec = emb.embed("Hello, world!")
    vecs = emb.embed_batch(["one", "two", "three"])
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory_skill.contracts import MemorySkillConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Embedder
# ═══════════════════════════════════════════════════════════════════════════════


# ── Singleton cache: same model_path → shared Embedder ──
_embedder_cache: dict[str, Embedder] = {}


def clear_cache() -> None:
    """Clear the module-level embedder cache.

    Call between tests to force a fresh Embedder instance (avoids
    cross-test state leakage from the singleton cache).
    """
    _embedder_cache.clear()


class Embedder:
    """Generate text embeddings via ONNX Runtime with a deterministic fallback.

    The embedder lazily loads the ONNX model on the first call to ``embed()``
    or ``embed_batch()``.  If the model file is not found or ONNX Runtime is
    unavailable, it falls back to a SHA-256–based deterministic embedding
    (unless ``fallback_enabled=False``, in which case ``ModelLoadError`` is
    raised).

    Parameters
    ----------
    config:
        ``MemorySkillConfig`` instance (``embedding_dim`` and ``model_path``
        are the relevant fields).
    fallback_enabled:
        If ``True`` (default), use a deterministic hash-based fallback when
        the ONNX model cannot be loaded.  Set to ``False`` to raise
        ``ModelLoadError`` on failure.
    """

    def __init__(
        self,
        config: MemorySkillConfig,
        fallback_enabled: bool = True,
    ) -> None:
        # Singleton: reuse existing embedder for same model path
        key = config.model_path
        if key in _embedder_cache:
            self._session = _embedder_cache[key]._session
            self._tokenizer = _embedder_cache[key]._tokenizer
            self._load_attempted = _embedder_cache[key]._load_attempted
            self._dim = _embedder_cache[key]._dim
            self._model_path = _embedder_cache[key]._model_path
            self._fallback_enabled = _embedder_cache[key]._fallback_enabled
            self._load_error = _embedder_cache[key]._load_error
            return

        self._dim: int = config.embedding_dim
        self._model_path: str = config.model_path
        self._fallback_enabled: bool = fallback_enabled

        # Lazy-load state
        self._session: object | None = None
        self._tokenizer: object | None = None
        self._load_attempted: bool = False
        self._load_error: str | None = None

        _embedder_cache[key] = self

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """Current embedding mode: ``"onnx"`` or ``"fallback"``.

        Triggers the lazy load (once) so the reported mode reflects a real
        load attempt rather than uninitialized state.
        """
        self._ensure_model_loaded()
        return "onnx" if self._session is not None else "fallback"

    @property
    def degraded(self) -> bool:
        """True when the ONNX model is unavailable and the hash fallback is in use.

        While degraded, semantic retrieval / dedup / learning judgments are
        reduced to SHA-256 hash equality — visibly worse than real embeddings.
        """
        return self.mode == "fallback"

    @property
    def reason(self) -> str | None:
        """Human-readable reason for the degraded mode (``None`` when healthy)."""
        return self._load_error

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a 1024-dim float vector.

        Parameters
        ----------
        text:
            The input text.  An empty string yields an all-zero vector.

        Returns
        -------
        list[float]
            The embedding vector (length == ``config.embedding_dim``).
        """
        if not text:
            return [0.0] * self._dim

        self._ensure_model_loaded()

        if self._session is not None:
            return self._onnx_embed(text)
        return self._fallback_embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts at once.

        Parameters
        ----------
        texts:
            A list of input strings.  An empty list returns an empty list.

        Returns
        -------
        list[list[float]]
            One embedding vector per input string.
        """
        if not texts:
            return []

        self._ensure_model_loaded()

        if self._session is not None:
            return self._onnx_embed_batch(texts)
        return [self._fallback_embed(t) for t in texts]

    # ── Model loading ─────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        """Attempt to load the ONNX model (lazily, retrying after interruption).

        On failure, either raise ``ModelLoadError`` or use the fallback,
        depending on ``fallback_enabled``.

        A load attempt is final only when it produced a session or recorded a
        concrete error. A caller that times out can abandon the attempt
        mid-way, leaving both empty; the next call then retries instead of
        staying in degraded mode forever.
        """
        if self._load_attempted and (
            self._session is not None or self._load_error is not None
        ):
            return
        self._load_attempted = True
        self._load_error = None
        try:
            self._load_onnx_model()
        except Exception as exc:
            if not self._fallback_enabled:
                from memory_skill.contracts import ModelLoadError

                raise ModelLoadError(
                    f"Failed to load ONNX model from '{self._model_path}': {exc}"
                ) from exc
            # Fallback enabled — use hash-based path with warning.
            self._load_error = str(exc)
            logging.warning(
                "ONNX model unavailable from '%s' — using SHA-256 fallback. "
                "Semantic search will be DEGRADED (hash equality only).",
                self._model_path,
            )

    def _load_onnx_model(self) -> None:
        """Load the ONNX model and tokenizer into memory."""
        import onnxruntime as ort

        model_path = self._model_path

        # Support both a directory (containing model.onnx) and a direct file.
        if os.path.isdir(model_path):
            onnx_file = os.path.join(model_path, "model.onnx")
            if not os.path.isfile(onnx_file):
                onnx_file = os.path.join(model_path, "onnx", "model.onnx")
        else:
            onnx_file = model_path

        if not os.path.isfile(onnx_file):
            raise FileNotFoundError(
                f"ONNX model file not found: {onnx_file}"
            )

        # Load ONNX session — prefer GPU (CUDA) when available, fall back to CPU
        self._session = ort.InferenceSession(
            onnx_file,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        # Attempt to load the tokenizer if tokenizers package is available.
        try:
            self._load_tokenizer(model_path)
        except Exception:
            logging.warning("Tokenizer load failed, using fallback")

    def _load_tokenizer(self, model_dir: str) -> None:
        """Load the HuggingFace tokenizer from ``model_dir``."""
        from tokenizers import Tokenizer

        # Common tokenizer file names
        candidates = [
            "tokenizer.json",
            os.path.join("tokenizer", "tokenizer.json"),
        ]
        for cand in candidates:
            path = os.path.join(model_dir, cand)
            if os.path.isfile(path):
                self._tokenizer = Tokenizer.from_file(path)
                return

    # ── ONNX inference ────────────────────────────────────────────────────

    def _onnx_embed(self, text: str) -> list[float]:
        """Run ONNX inference for a single text."""
        results = self._onnx_embed_batch([text])
        return results[0]

    def _onnx_embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Run ONNX inference for a batch of texts (mean pooling)."""
        import numpy as np

        session = self._session
        assert session is not None

        # Use tokenizer if available, otherwise fall back.
        if self._tokenizer is not None:
            encodings = self._tokenizer.encode_batch(texts)
            # bge-large-en-v1.5 has 512 positional encodings — truncate
            # sequences to avoid "indices element out of data bounds" errors.
            max_len = min(max(len(e.ids) for e in encodings), 512)
            input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
            attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)
            for i, enc in enumerate(encodings):
                length = min(len(enc.ids), max_len)
                input_ids[i, :length] = enc.ids[:length]
                attention_mask[i, :length] = 1
        else:
            # No tokenizer — use a simple character-level fallback.
            max_len = min(max((len(t) for t in texts), default=1), 512)
            input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
            attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)
            for i, t in enumerate(texts):
                ids = [ord(c) % 30000 for c in t[:max_len]]
                input_ids[i, :len(ids)] = ids
                attention_mask[i, :len(ids)] = 1

        # Single-segment inputs for BERT-like models.
        token_type_ids = np.zeros_like(input_ids)

        # Get input names from the ONNX model
        input_names = [inp.name for inp in session.get_inputs()]

        feed: dict[str, np.ndarray] = {}
        for name in input_names:
            lowered = name.lower()
            if "attention" in lowered:
                feed[name] = attention_mask
            elif "token_type" in lowered or "segment" in lowered:
                feed[name] = token_type_ids
            else:
                feed[name] = input_ids

        outputs = session.run(None, feed)
        # The first output is typically (batch, seq_len, hidden_dim) or
        # (batch, hidden_dim) for pooled models.
        last_hidden = outputs[0]

        # Mean pooling over the sequence dimension if the output is 3D.
        if last_hidden.ndim == 3:
            mask_expanded = np.expand_dims(
                attention_mask.astype(np.float32), axis=-1
            )
            summed = np.sum(last_hidden * mask_expanded, axis=1)
            counts = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
            embeddings = summed / counts
        else:
            embeddings = last_hidden

        # L2 normalize each row
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        embeddings = embeddings / norms

        return embeddings.astype(np.float32).tolist()

    # ── Fallback (hash-based) ─────────────────────────────────────────────

    def _fallback_embed(self, text: str) -> list[float]:
        """Deterministic SHA-256–based fallback embedding.

        Produces a ``config.embedding_dim``-dim L2-normalized vector from
        the text hash.  Empty string returns an all-zero vector.
        """
        dim = self._dim

        if not text:
            return [0.0] * dim

        digest = hashlib.sha256(text.encode("utf-8")).digest()  # 32 bytes

        vec: list[float] = []
        for i in range(dim):
            # Cycle through the 32-byte digest to fill dim dimensions.
            b1 = digest[(i * 2) % 32]
            b2 = digest[(i * 2 + 1) % 32]
            val = ((b1 << 8) | b2) / 65535.0
            vec.append(val)

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec
