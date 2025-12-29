# -*- coding: utf-8 -*-
"""Shared RAG helpers.

Goals:
- Keep ingest and retriever using the SAME embedding/device logic.
- Provide clear CUDA dependency errors when device is forced to cuda.
- Keep changes minimal and compatible with the existing LangChain version.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


DEFAULT_COLLECTION = "medical_kb"
DEFAULT_HF_EMBED_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_BCE_MODEL = "maidalun1020/bce-embedding-base_v1"


def apply_windows_openmp_workaround() -> None:
    """Avoid common OpenMP runtime conflicts on Windows."""
    try:
        if platform.system().lower() != "windows":
            return
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    except Exception:
        return


# Apply early to avoid `OMP: Error #15` on Windows when torch/numpy load OpenMP.
apply_windows_openmp_workaround()


def env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def env_flag(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = (os.getenv(name) or "").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def resolve_app_dir(from_file: Path) -> Path:
    """Return the app/ directory (…/app)."""
    return from_file.resolve().parents[1]


def resolve_persist_dir(app_dir: Path) -> Path:
    p = env_str("RAG_PERSIST_DIR", "")
    if p:
        pp = Path(p)
        if pp.is_absolute():
            return pp
        return (app_dir / pp).resolve()
    return app_dir / "rag/kb_store"


def resolve_allowed_kb_dir(app_dir: Path) -> Path:
    """The only allowed default ingestion dir (data isolation)."""
    return (app_dir / "rag/kb_docs/dataset-v2/合并数据-CSV格式").resolve()


def resolve_kb_dir_strict(app_dir: Path) -> Path:
    """Resolve kb_dir and enforce it's within the allowed directory.

    - Default: allowed dir.
    - RAG_KB_DIR may be set to the allowed dir or a subdir of it.
    """
    allowed = resolve_allowed_kb_dir(app_dir)
    raw = env_str("RAG_KB_DIR", "")

    if raw == "":
        kb_dir = allowed
    else:
        pp = Path(raw)
        # Keep compatibility with existing usage where RAG_KB_DIR is relative to app/
        kb_dir = (pp.resolve() if pp.is_absolute() else (app_dir / pp).resolve())

    try:
        kb_dir.relative_to(allowed)
    except Exception as e:
        raise RuntimeError(
            "为防止评测/说明污染KB，本项目仅允许入库目录位于：\n"
            f"  allowed={allowed}\n"
            f"  got={kb_dir}\n"
            "如确需变更，请把数据放到 allowed 目录下（可使用其子目录），并通过 RAG_KB_DIR 指向该子目录。"
        ) from e

    return kb_dir


def resolve_embedding_device() -> str:
    """Resolve embedding device.

    Env priority (new -> old):
    - RAG_DEVICE: auto|cuda|mps|cpu (or cuda:0)
    - RAG_EMBEDDING_DEVICE: auto|cuda|mps|cpu (or cuda:0)

    - auto (default): use cuda if available, else mps (Apple Silicon), else cpu.
    - cuda / cuda:0: require torch + CUDA (NVIDIA GPU).
    - mps: require torch + MPS (Apple Silicon GPU).
    - cpu: use CPU.
    """
    raw = (env_str("RAG_DEVICE", "") or env_str("RAG_EMBEDDING_DEVICE", "auto") or "auto").strip()
    raw_l = raw.lower()
    if raw_l == "gpu":
        raw_l = "cuda"

    if raw_l != "auto":
        # Explicit CUDA request
        if raw_l.startswith("cuda"):
            try:
                import torch  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "已设置 RAG_DEVICE=cuda（或 RAG_EMBEDDING_DEVICE=cuda），但当前环境缺少 torch。\n"
                    "解决：安装支持CUDA的torch（torch+cuXXX），或改用 RAG_DEVICE=cpu。\n"
                    "参考：pytorch.org/get-started/locally/"
                ) from e
            if not bool(getattr(torch.cuda, "is_available", lambda: False)()):
                raise RuntimeError(
                    "已设置 RAG_DEVICE=cuda（或 RAG_EMBEDDING_DEVICE=cuda），但 torch.cuda.is_available() 为 False。\n"
                    "可能原因：未安装CUDA版torch/驱动不匹配/无GPU。\n"
                    "解决：安装匹配CUDA的torch与驱动，或改用 RAG_DEVICE=cpu。"
                )
            return raw_l

        # Explicit MPS request (Apple Silicon)
        if raw_l == "mps":
            try:
                import torch  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "已设置 RAG_DEVICE=mps，但当前环境缺少 torch。\n"
                    "解决：安装 torch（pip install torch），或改用 RAG_DEVICE=cpu。"
                ) from e
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise RuntimeError(
                    "已设置 RAG_DEVICE=mps，但 torch.backends.mps.is_available() 为 False。\n"
                    "可能原因：非 Apple Silicon 设备/macOS 版本过低/torch 版本不支持 MPS。\n"
                    "解决：确认是 Apple Silicon Mac 且 macOS >= 12.3，或改用 RAG_DEVICE=cpu。"
                )
            return "mps"

        if raw_l == "cpu":
            return "cpu"

        return raw_l

    # auto: CUDA > MPS > CPU
    try:
        import torch  # type: ignore

        # Check CUDA first (NVIDIA GPU)
        if bool(getattr(torch.cuda, "is_available", lambda: False)()):
            return "cuda"

        # Check MPS (Apple Silicon GPU)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Additional check: MPS must be built
            if torch.backends.mps.is_built():
                return "mps"
    except Exception:
        pass

    return "cpu"


@dataclass(frozen=True)
class EmbeddingInfo:
    provider_requested: str
    provider_used: str
    model_name: str
    device: str
    fallback_reason: str = ""


def _import_embeddings_interface():
    try:
        from langchain_core.embeddings import Embeddings  # type: ignore

        return Embeddings
    except Exception:
        try:
            from langchain.embeddings.base import Embeddings  # type: ignore

            return Embeddings
        except Exception as e:
            raise RuntimeError("缺少 langchain 依赖，无法构建 Embeddings。") from e


def make_embeddings() -> Tuple[object, EmbeddingInfo]:
    """Create embedding function for LangChain/Chroma.

    Default provider is BCEmbedding; if unavailable, fallback to HuggingFaceEmbeddings with a clear warning.
    Supports CUDA, MPS (Apple Silicon), and CPU. MPS will fallback to CPU if not supported.
    """
    apply_windows_openmp_workaround()

    provider_requested = (
        env_str("RAG_PROVIDER", "")
        or env_str("RAG_EMBEDDINGS_PROVIDER", "")
        or "bce"
    ).lower()
    device = resolve_embedding_device()

    # BCE (preferred) - 使用 sentence-transformers 加载，支持 MPS
    if provider_requested in {"bce", "bcembedding"}:
        model_name = (
            env_str("RAG_EMBED_MODEL", "")
            or env_str("RAG_BCE_MODEL", "")
            or DEFAULT_BCE_MODEL
        )
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            Embeddings = _import_embeddings_interface()

            class BCEEmbeddings(Embeddings):
                """BCE Embeddings using sentence-transformers (supports MPS)."""

                def __init__(self, name: str, dev: str):
                    self._name = name
                    self._dev = dev
                    self._actual_dev = dev
                    self._model = None

                def _get_model(self) -> SentenceTransformer:
                    if self._model is not None:
                        return self._model

                    try:
                        self._model = SentenceTransformer(self._name, device=self._dev)
                        self._actual_dev = self._dev
                        print(
                            f"[RAG_EMBED] INFO model={self._name} device={self._dev}",
                            flush=True,
                        )
                    except Exception as e:
                        # MPS fallback to CPU
                        if self._dev == "mps":
                            print(
                                f"[RAG_EMBED] WARN mps_fallback reason={type(e).__name__}: {e}",
                                flush=True,
                            )
                            self._model = SentenceTransformer(self._name, device="cpu")
                            self._actual_dev = "cpu"
                            print("[RAG_EMBED] INFO fallback_to_cpu success", flush=True)
                        else:
                            raise
                    return self._model

                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    m = self._get_model()
                    vecs = m.encode(texts, show_progress_bar=False)
                    try:
                        return [v.tolist() for v in vecs]
                    except Exception:
                        return [[float(x) for x in v] for v in vecs]

                def embed_query(self, text: str) -> List[float]:
                    m = self._get_model()
                    vecs = m.encode([text], show_progress_bar=False)
                    v = vecs[0]
                    try:
                        return v.tolist()
                    except Exception:
                        return [float(x) for x in v]

            return (
                BCEEmbeddings(model_name, device),
                EmbeddingInfo(
                    provider_requested=provider_requested,
                    provider_used="bce",
                    model_name=model_name,
                    device=device,
                    fallback_reason="",
                ),
            )
        except Exception as e:
            # fall back below
            fallback_reason = f"sentence-transformers加载BCE模型失败：{type(e).__name__}: {e}"

            provider_requested = provider_requested  # keep
            # continue to HF fallback

            provider_used = "hf"
            hf_model = env_str("HF_EMBEDDING_MODEL", "") or DEFAULT_HF_EMBED_MODEL
            try:
                from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
            except Exception as e2:
                raise RuntimeError(
                    "sentence-transformers 不可用，且缺少 HuggingFaceEmbeddings（langchain 依赖不完整）。\n"
                    f"sentence-transformers error: {fallback_reason}\n"
                    f"HF error: {type(e2).__name__}: {e2}"
                ) from e2

            print(
                "[RAG_EMBED] WARN fallback_to=HuggingFaceEmbeddings "
                f"reason={fallback_reason} model={hf_model} device={device}",
                flush=True,
            )
            return (
                HuggingFaceEmbeddings(model_name=hf_model, model_kwargs={"device": device}),
                EmbeddingInfo(
                    provider_requested=provider_requested,
                    provider_used=provider_used,
                    model_name=hf_model,
                    device=device,
                    fallback_reason=fallback_reason,
                ),
            )

    # HF (explicit)
    hf_model = env_str("HF_EMBEDDING_MODEL", "") or DEFAULT_HF_EMBED_MODEL
    try:
        from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少 langchain 依赖，无法构建 HuggingFaceEmbeddings。") from e

    return (
        HuggingFaceEmbeddings(model_name=hf_model, model_kwargs={"device": device}),
        EmbeddingInfo(
            provider_requested=provider_requested,
            provider_used="hf",
            model_name=hf_model,
            device=device,
            fallback_reason="",
        ),
    )
