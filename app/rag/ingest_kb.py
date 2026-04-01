import os
import platform
import re
import shutil
import csv
import json
import unicodedata
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Allow running as a script: `python app/rag/ingest_kb.py`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.rag.utils.rag_shared import (
    DEFAULT_COLLECTION,
    apply_windows_openmp_workaround,
    env_flag,
    env_int,
    env_str,
    make_embeddings,
    resolve_allowed_kb_dir,
    resolve_app_dir,
    resolve_kb_dir_strict,
    resolve_persist_dir,
)
from app.rag.opensearch_store import OpenSearchConfig, OpenSearchHybridStore


apply_windows_openmp_workaround()


# -----------------------------
# Data contracts
# -----------------------------
@dataclass
class RawDoc:
    text: str
    metadata: Dict


@dataclass
class IngestProgress:
    version: int
    files: Dict[str, Dict[str, Any]]


def _env_backend() -> str:
    backend = (env_str("RAG_BACKEND", "") or "opensearch").strip().lower()
    if not backend or backend in {"opensearch", "open-search"}:
        return "opensearch"
    raise RuntimeError("README 对齐后入库仅支持 OpenSearch，请移除 faiss 等非 OpenSearch 后端配置。")


def _env_opensearch_url() -> str:
    return env_str("OPENSEARCH_URL", "").strip()


def _env_opensearch_index() -> str:
    return (env_str("OPENSEARCH_INDEX", "") or env_str("RAG_COLLECTION", "") or DEFAULT_COLLECTION).strip()


def _env_opensearch_username() -> str:
    return env_str("OPENSEARCH_USERNAME", "").strip()


def _env_opensearch_password() -> str:
    return env_str("OPENSEARCH_PASSWORD", "").strip()


def _env_opensearch_verify_ssl() -> bool:
    return env_flag("OPENSEARCH_VERIFY_SSL", "0")


def _env_opensearch_vector_dim() -> int:
    value = env_int("OPENSEARCH_VECTOR_DIM", default=768)
    if not value:
        return 768
    return max(1, int(value))


def _env_opensearch_knn_k() -> int:
    value = env_int("OPENSEARCH_KNN_K", default=30)
    if not value:
        return 30
    return max(1, int(value))


def _env_opensearch_num_candidates() -> int:
    value = env_int("OPENSEARCH_NUM_CANDIDATES", default=128)
    if not value:
        return 128
    return max(1, int(value))


def get_opensearch_store() -> OpenSearchHybridStore:
    url = _env_opensearch_url()
    if not url:
        raise RuntimeError("已启用 OpenSearch 入库，但未配置 OPENSEARCH_URL。")

    return OpenSearchHybridStore(
        OpenSearchConfig(
            url=url,
            index_name=_env_opensearch_index(),
            vector_dim=_env_opensearch_vector_dim(),
            username=_env_opensearch_username(),
            password=_env_opensearch_password(),
            verify_ssl=_env_opensearch_verify_ssl(),
            knn_k=_env_opensearch_knn_k(),
            num_candidates=_env_opensearch_num_candidates(),
        )
    )


def _embed_texts(embeddings: Any, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    if hasattr(embeddings, "embed_documents"):
        return [list(vec or []) for vec in embeddings.embed_documents(texts)]
    return [list((embeddings.embed_query(text) or [])) for text in texts]


def _flush_opensearch_batch(store: OpenSearchHybridStore, embeddings: Any, batch: List[Dict[str, Any]]) -> int:
    if not batch:
        return 0
    vectors = _embed_texts(embeddings, [str(item.get("text") or "") for item in batch])
    docs: List[Dict[str, Any]] = []
    for item, vector in zip(batch, vectors):
        payload = dict(item)
        payload["embedding"] = vector
        docs.append(payload)
    written = store.bulk_upsert(docs)
    batch.clear()
    return written


def _resolve_path_maybe_relative(base_dir: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (base_dir / pp).resolve()


def _progress_key(kb_dir: Path, path: Path) -> str:
    """Stable relative key for progress file."""
    try:
        return str(path.resolve().relative_to(kb_dir.resolve())).replace("\\", "/")
    except Exception:
        return str(path.name)


def _load_progress(progress_path: Path) -> IngestProgress:
    try:
        if not progress_path.exists():
            return IngestProgress(version=1, files={})
        data = json.loads(progress_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return IngestProgress(version=1, files={})
        files = data.get("files")
        if not isinstance(files, dict):
            files = {}
        cleaned: Dict[str, Dict[str, Any]] = {}
        for k, v in files.items():
            if isinstance(k, str) and isinstance(v, dict):
                cleaned[k] = v
        return IngestProgress(version=int(data.get("version") or 1), files=cleaned)
    except Exception:
        return IngestProgress(version=1, files={})


def _save_progress(progress_path: Path, progress: IngestProgress) -> None:
    try:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"version": progress.version, "files": progress.files}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, progress_path)
    except Exception:
        return


def _append_bad_row(
    log_path: Path,
    *,
    source_file: str,
    row: Optional[int],
    reason: str,
    sample: str,
) -> None:
    """Append a rejection record for hard-gate failures.

    - Must only write UTF-8 text.
    - Keep samples short to avoid huge logs.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        safe_sample = (sample or "").replace("\n", " ").replace("\r", " ")
        if len(safe_sample) > 240:
            safe_sample = safe_sample[:237] + "..."

        rec = {
            "source_file": str(source_file or ""),
            "row": int(row) if isinstance(row, int) else None,
            "reason": str(reason or ""),
            "sample": safe_sample,
        }
        log_path.open("a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        return


# -----------------------------
# Text cleaning / hard gate
# -----------------------------
_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_WHITESPACE = re.compile(r"[ \t\f\v]+")
_SEMANTIC_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])")
_STRUCTURED_PREFIX_KEYS = ("科室：", "主题：", "患者问题：", "医生回答：")


def _sanitize_text(s: str) -> str:
    s = s or ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\x00", "")
    s = _CTRL.sub("", s)
    s = unicodedata.normalize("NFKC", s)
    # keep newlines, normalize other spaces
    lines = []
    for line in s.split("\n"):
        line = line.replace("\u00a0", " ")
        line = _WHITESPACE.sub(" ", line).strip()
        lines.append(line)
    s = "\n".join([ln for ln in lines if ln != ""])
    return s.strip()


def _hard_gate(text: str) -> bool:
    """Return True if text is acceptable for ingestion."""
    if not isinstance(text, str):
        return False
    if "\x00" in text:
        return False
    # U+FFFD replacement char indicates decode damage (mojibake risk).
    if "\ufffd" in text:
        return False
    # too short is useless
    if len(text.strip()) < 10:
        return False
    return True


def _token_count_default(text: str) -> int:
    cleaned = str(text or "").strip()
    if not cleaned:
        return 0
    try:
        import tiktoken  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少 tiktoken，无法执行 token 回退分块。") from e
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(cleaned))


def _split_semantic_units(text: str) -> List[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", cleaned) if part and part.strip()]
    units: List[str] = []
    for paragraph in paragraphs or [cleaned]:
        lines = [line.strip() for line in paragraph.splitlines() if line and line.strip()]
        if len(lines) > 1 and any(line.startswith(("科室：", "主题：", "患者问题：", "医生回答：")) for line in lines):
            units.extend(lines)
            continue
        sentences = [part.strip() for part in _SEMANTIC_SENTENCE_SPLIT.split(paragraph) if part and part.strip()]
        if sentences:
            units.extend(sentences)
        else:
            units.append(paragraph)
    return units


def _join_chunk_parts(parts: List[str]) -> str:
    if not parts:
        return ""
    if all("\n" not in part for part in parts):
        joined = "".join(parts)
    else:
        joined = "\n".join(parts)
    return joined.strip()


def _split_structured_prefix(text: str) -> Tuple[str, str]:
    lines = [line.rstrip() for line in str(text or "").splitlines() if line and line.strip()]
    if len(lines) < 2:
        return "", str(text or "").strip()
    if not any(line.startswith(_STRUCTURED_PREFIX_KEYS[:-1]) for line in lines[:3]):
        return "", str(text or "").strip()
    if not lines[-1].startswith("医生回答："):
        return "", str(text or "").strip()
    prefix = "\n".join(lines[:-1]).strip()
    body = lines[-1].strip()
    return prefix, body


def _token_backoff_split(
    text: str,
    *,
    max_tokens: int,
    max_chars: int,
    overlap_tokens: int,
    token_counter,
) -> List[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []

    if " " in cleaned:
        atoms = [part for part in cleaned.split(" ") if part]
        joiner = " "
    else:
        atoms = [ch for ch in cleaned if ch.strip()]
        joiner = ""

    if not atoms:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    step_back = max(0, int(overlap_tokens))
    while start < len(atoms):
        end = start
        best = ""
        while end < len(atoms):
            candidate = joiner.join(atoms[start : end + 1]).strip()
            if not candidate:
                end += 1
                continue
            if token_counter(candidate) > max_tokens or len(candidate) > max_chars:
                break
            best = candidate
            end += 1

        if not best:
            best = joiner.join(atoms[start : start + 1]).strip()
            end = start + 1

        chunks.append(best)
        if end >= len(atoms):
            break
        start = max(start + 1, end - step_back)
    return chunks


def _semantic_chunk_text(
    text: str,
    *,
    target_tokens: int,
    max_tokens: int,
    max_chars: int,
    overlap_tokens: int = 0,
    token_counter=None,
) -> List[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []

    counter = token_counter or _token_count_default
    if counter(cleaned) <= max_tokens and len(cleaned) <= max_chars:
        return [cleaned]

    prefix, body = _split_structured_prefix(cleaned)
    body_clean = body.strip() if body.strip() else cleaned
    units = _split_semantic_units(body_clean)
    if not units:
        units = [body_clean]

    chunks: List[str] = []
    current_parts: List[str] = []

    def _emit_current() -> None:
        if not current_parts:
            return
        joined = _join_chunk_parts(current_parts)
        if prefix:
            joined = f"{prefix}\n{joined}".strip()
        chunks.append(joined)
        current_parts.clear()

    for unit in units:
        candidate_parts = current_parts + [unit]
        candidate_body = _join_chunk_parts(candidate_parts)
        candidate_text = f"{prefix}\n{candidate_body}".strip() if prefix else candidate_body
        if current_parts and counter(candidate_text) > target_tokens:
            _emit_current()
            candidate_parts = [unit]
            candidate_body = _join_chunk_parts(candidate_parts)
            candidate_text = f"{prefix}\n{candidate_body}".strip() if prefix else candidate_body

        if counter(candidate_text) <= max_tokens and len(candidate_text) <= max_chars:
            current_parts.extend(candidate_parts[len(current_parts) :])
            continue

        if current_parts:
            _emit_current()

        fallback_chunks = _token_backoff_split(
            unit,
            max_tokens=max_tokens,
            max_chars=max_chars if not prefix else max(32, max_chars - len(prefix) - 1),
            overlap_tokens=overlap_tokens,
            token_counter=counter,
        )
        for fallback in fallback_chunks:
            final_chunk = f"{prefix}\n{fallback}".strip() if prefix else fallback
            chunks.append(final_chunk)

    _emit_current()
    return [chunk for chunk in chunks if chunk]


def _read_text_file(path: Path) -> str:
    # NOTE: we avoid errors="replace" here; use ignore to reduce accidental U+FFFD injection.
    return path.read_text(encoding="utf-8", errors="ignore")


# -----------------------------
# Encoding heuristic for Chinese CSVs
# -----------------------------
def _count_cjk(s: str) -> int:
    return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")


def _count_latin1_suspects(s: str) -> int:
    return sum(1 for ch in s if "\u00c0" <= ch <= "\u00ff")


def _pick_best_encoding(path: Path, candidates: List[str]) -> str:
    """Pick a likely-correct text encoding for Chinese KB CSVs."""
    try:
        with path.open("rb") as f:
            sample_bytes = f.read(65536)
    except Exception:
        return candidates[0] if candidates else "utf-8"

    best_enc = candidates[0] if candidates else "utf-8"
    best_score = -10**18

    for enc in candidates:
        try:
            s = sample_bytes.decode(enc, errors="replace")
        except Exception:
            continue

        repl = s.count("\ufffd")
        bom_garble = s.count("ï»¿")
        latin1 = _count_latin1_suspects(s)
        mojibake_markers = s.count("Ã") + s.count("Â")
        cjk = _count_cjk(s)

        score = (cjk * 5) - (repl * 200) - (bom_garble * 200) - (mojibake_markers * 50) - (latin1 * 3)
        if score > best_score:
            best_score = score
            best_enc = enc

    return best_enc


# -----------------------------
# Markdown / PDF helpers (kept for compatibility)
# -----------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$")


def _parse_yaml_front_matter(md_text: str) -> Tuple[Dict[str, str], str]:
    if not md_text:
        return {}, ""
    lines = md_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, md_text

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, md_text

    meta: Dict[str, str] = {}
    for raw in lines[1:end_idx]:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            meta[k] = v

    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    return meta, body


def _split_markdown_into_sections(md_text: str) -> List[Tuple[str, str]]:
    lines = md_text.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_title = ""
    current_buf: List[str] = []

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            if current_buf:
                sections.append((current_title, current_buf))
            current_title = m.group(2).strip()
            current_buf = []
        else:
            current_buf.append(line)

    if current_buf:
        sections.append((current_title, current_buf))

    out: List[Tuple[str, str]] = []
    for title, buf in sections:
        text = "\n".join(buf).strip()
        if text:
            out.append((title, text))
    return out


def _read_pdf_pages(path: Path) -> List[RawDoc]:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("缺少依赖 pypdf，无法读取PDF。请先安装 pypdf。") from e

    reader = PdfReader(str(path))
    docs: List[RawDoc] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = _sanitize_text(text)
        if not _hard_gate(text):
            continue
        docs.append(
            RawDoc(
                text=text,
                metadata={"source": str(path.name), "page": i, "section": ""},
            )
        )
    return docs


# -----------------------------
# CSV readers (robust + hard gate)
# -----------------------------
def _clean_cell(v: object, max_len: int = 6000) -> str:
    s = "" if v is None else str(v)
    s = _sanitize_text(s)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _csv_get_cell(row: List[str], header: List[str], *names: str) -> str:
    for name in names:
        name_l = name.strip().lower()
        if name_l in header:
            i = header.index(name_l)
            if i < len(row):
                return _clean_cell(row[i])
    return ""


def _read_csv_docs(path: Path, *, max_rows: Optional[int] = None) -> List[RawDoc]:
    docs: List[RawDoc] = []

    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    chosen = _pick_best_encoding(path, encodings)

    def _read_with_encoding(enc: str) -> List[RawDoc]:
        with path.open("r", encoding=enc, errors="replace", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.excel
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            except Exception:
                pass
            if getattr(dialect, "delimiter", None) in {"\r", "\n"}:
                dialect = csv.excel

            reader = csv.reader(f, dialect)
            rows_read = 0
            header: Optional[List[str]] = None

            for row_idx, row in enumerate(reader):
                if not row:
                    continue
                if len(row) == 1 and isinstance(row[0], str) and "," in row[0]:
                    row = [c.strip() for c in row[0].split(",")]

                if header is None:
                    cand = [str(x).strip().lower() for x in row]
                    if any(k in cand for k in ["department", "ask", "answer", "title", "科室", "问题", "回答", "标题"]):
                        header = cand
                        continue
                    header = []

                if max_rows is not None and rows_read >= max_rows:
                    break

                department = ""
                title = ""
                ask = ""
                answer = ""

                header_cols: List[str] = header or []
                if header_cols:
                    department = _csv_get_cell(row, header_cols, "department", "科室")
                    title = _csv_get_cell(row, header_cols, "title", "标题")
                    ask = _csv_get_cell(row, header_cols, "ask", "question", "问题")
                    answer = _csv_get_cell(row, header_cols, "answer", "response", "回答")
                else:
                    if len(row) >= 1:
                        department = _clean_cell(row[0])
                    if len(row) >= 2:
                        title = _clean_cell(row[1])
                    if len(row) >= 3:
                        ask = _clean_cell(row[2])
                    if len(row) >= 4:
                        answer = _clean_cell(row[3])

                if not ask or not answer:
                    continue

                qa_text = "\n".join(
                    [
                        f"科室：{department}" if department else "",
                        f"主题：{title}" if title else "",
                        f"患者问题：{ask}",
                        f"医生回答：{answer}",
                    ]
                ).strip()
                qa_text = _sanitize_text(qa_text)
                if not _hard_gate(qa_text):
                    continue

                docs.append(
                    RawDoc(
                        text=qa_text,
                        metadata={
                            "source_file": str(path.name),
                            "source": str(path.name),
                            "page": None,
                            "section": department or "",
                            "department": department,
                            "title": title,
                            "row": row_idx,
                            "domain": "medical_qa",
                        },
                    )
                )
                rows_read += 1

            return docs

    try:
        return _read_with_encoding(chosen)
    except Exception:
        last_err: Optional[Exception] = None
        for enc in encodings:
            if enc == chosen:
                continue
            try:
                return _read_with_encoding(enc)
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"无法读取CSV文件：{path}. last_error={type(last_err).__name__}: {last_err}")


def _iter_csv_docs(
    path: Path,
    *,
    start_row: int = 0,
    max_rows: Optional[int] = None,
    bad_rows_path: Optional[Path] = None,
) -> Iterator[Tuple[int, RawDoc]]:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    chosen = _pick_best_encoding(path, encodings)

    def _iter_with_encoding(enc: str) -> Iterator[Tuple[int, RawDoc]]:
        with path.open("r", encoding=enc, errors="replace", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.excel
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            except Exception:
                pass
            if getattr(dialect, "delimiter", None) in {"\r", "\n"}:
                dialect = csv.excel

            reader = csv.reader(f, dialect)
            rows_yielded = 0
            header: Optional[List[str]] = None

            for row_idx, row in enumerate(reader):
                if row_idx < start_row:
                    continue
                if not row:
                    continue
                if len(row) == 1 and isinstance(row[0], str) and "," in row[0]:
                    row = [c.strip() for c in row[0].split(",")]

                if header is None:
                    cand = [str(x).strip().lower() for x in row]
                    if any(k in cand for k in ["department", "ask", "answer", "title", "科室", "问题", "回答", "标题"]):
                        header = cand
                        continue
                    header = []

                if max_rows is not None and rows_yielded >= max_rows:
                    break

                department = ""
                title = ""
                ask = ""
                answer = ""

                header_cols: List[str] = header or []
                if header_cols:
                    department = _csv_get_cell(row, header_cols, "department", "科室")
                    title = _csv_get_cell(row, header_cols, "title", "标题")
                    ask = _csv_get_cell(row, header_cols, "ask", "question", "问题")
                    answer = _csv_get_cell(row, header_cols, "answer", "response", "回答")
                else:
                    if len(row) >= 1:
                        department = _clean_cell(row[0])
                    if len(row) >= 2:
                        title = _clean_cell(row[1])
                    if len(row) >= 3:
                        ask = _clean_cell(row[2])
                    if len(row) >= 4:
                        answer = _clean_cell(row[3])

                if not ask or not answer:
                    continue

                # Build raw text BEFORE sanitization so we can hard-reject decode damage.
                qa_text_raw = "\n".join(
                    [
                        f"科室：{department}" if department else "",
                        f"主题：{title}" if title else "",
                        f"患者问题：{ask}",
                        f"医生回答：{answer}",
                    ]
                ).strip()

                # Hard gate: reject if contains NUL or U+FFFD replacement char.
                if ("\x00" in qa_text_raw) or ("\ufffd" in qa_text_raw):
                    if bad_rows_path is not None:
                        _append_bad_row(
                            bad_rows_path,
                            source_file=str(path.name),
                            row=int(row_idx),
                            reason="hard_gate_reject",
                            sample=qa_text_raw,
                        )
                    continue

                qa_text = _sanitize_text(qa_text_raw)
                if not _hard_gate(qa_text):
                    if bad_rows_path is not None:
                        _append_bad_row(
                            bad_rows_path,
                            source_file=str(path.name),
                            row=int(row_idx),
                            reason="hard_gate_reject_after_sanitize",
                            sample=qa_text,
                        )
                    continue

                yield (
                    row_idx,
                    RawDoc(
                        text=qa_text,
                        metadata={
                            "source_file": str(path.name),
                            "source": str(path.name),
                            "page": None,
                            "section": department or "",
                            "department": department,
                            "title": title,
                            "row": row_idx,
                            "domain": "medical_qa",
                        },
                    ),
                )
                rows_yielded += 1

            return

    try:
        yield from _iter_with_encoding(chosen)
        return
    except Exception:
        last_err: Optional[Exception] = None
        for enc in encodings:
            if enc == chosen:
                continue
            try:
                yield from _iter_with_encoding(enc)
                return
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"无法读取CSV文件：{path}. last_error={type(last_err).__name__}: {last_err}")


def load_raw_docs(kb_dir: Path) -> List[RawDoc]:
    docs: List[RawDoc] = []
    if not kb_dir.exists():
        return docs

    for path in sorted(kb_dir.rglob("*")):
        if path.is_dir():
            continue

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            docs.extend(_read_pdf_pages(path))
        elif suffix == ".csv":
            max_rows = env_int("RAG_CSV_MAX_ROWS", default=None)
            docs.extend(_read_csv_docs(path, max_rows=max_rows))
        elif suffix in {".md", ".markdown"}:
            md = _read_text_file(path)
            fm, body = _parse_yaml_front_matter(md)
            base_md = {
                "source_file": str(path.name),
                "source": (fm.get("source_url") or str(path.name)).strip(),
                "source_url": (fm.get("source_url") or "").strip(),
                "title": (fm.get("title") or "").strip(),
                "publisher": (fm.get("publisher") or "").strip(),
                "captured_at": (fm.get("captured_at") or "").strip(),
                "domain": (fm.get("domain") or "").strip(),
                "page": None,
            }
            for section_title, section_text in _split_markdown_into_sections(body):
                st = _sanitize_text(section_text)
                if not _hard_gate(st):
                    continue
                docs.append(RawDoc(text=st, metadata={**base_md, "section": section_title or ""}))
        elif suffix in {".txt"}:
            # 默认不入库 txt（避免把总结/评测说明等污染KB）。
            if not env_flag("RAG_INGEST_TXT", "0"):
                continue
            txt = _sanitize_text(_read_text_file(path))
            if not _hard_gate(txt):
                continue
            docs.append(RawDoc(text=txt, metadata={"source": str(path.name), "page": None, "section": ""}))
        else:
            continue

    return docs


def build_and_persist_store(
    kb_dir: Path,
    persist_dir: Path,
    collection_name: str = "medical_kb",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    embedding_model_name: Optional[str] = None,
) -> int:
    """Ingest kb_docs into the README-aligned OpenSearch store."""
    _env_backend()

    # 强制“数据隔离”：只允许入库合并CSV目录（或其子目录）
    # 仍保留：RAG_INGEST_TXT=1 时可入库 txt（但在本工程默认目录里不会有总结/评测文件）

    bad_rows_path = persist_dir / "bad_rows.log"

    embeddings, emb_info = make_embeddings()
    print(
        "[RAG_INGEST] "
        f"provider={emb_info.provider_used} model={emb_info.model_name} device={emb_info.device} "
        f"kb_dir={kb_dir} persist_dir={persist_dir} collection={collection_name}",
        flush=True,
    )
    do_reset = env_flag("RAG_RESET", "0")
    if do_reset:
        try:
            shutil.rmtree(persist_dir, ignore_errors=True)
        except Exception:
            pass

    persist_dir.mkdir(parents=True, exist_ok=True)

    opensearch_store = get_opensearch_store()

    # 默认只入库 CSV：只扫描 kb_dir 下的 *.csv
    csv_files = sorted([p for p in kb_dir.rglob("*.csv") if p.is_file()]) if kb_dir.exists() else []

    # 可选：仅当 RAG_INGEST_TXT=1 时才允许入库 TXT（避免评测/说明污染KB）
    txt_files: List[Path] = []
    if env_flag("RAG_INGEST_TXT", "0"):
        txt_files = sorted([p for p in kb_dir.rglob("*.txt") if p.is_file()]) if kb_dir.exists() else []

    if not csv_files and not txt_files:
        print(f"[RAG_INGEST] no_files_under_kb_dir={kb_dir} (csv={len(csv_files)} txt={len(txt_files)})", flush=True)
        return 0

    def _infer_department_group(csv_path: Path) -> str:
        try:
            parts = list(csv_path.parts)
            for i, part in enumerate(parts):
                if part in {"Data_数据", "Data", "数据"}:
                    if i + 1 < len(parts):
                        return str(parts[i + 1])
        except Exception:
            pass
        # fallback: use parent folder name
        return str(csv_path.parent.name)

    progress_path = Path((env_str("RAG_PROGRESS_PATH", "") or (persist_dir / "ingest_progress.json")))
    if not progress_path.is_absolute():
        progress_path = persist_dir / progress_path

    resume = env_flag("RAG_RESUME", "1")
    progress = IngestProgress(version=1, files={}) if (do_reset or (not resume)) else _load_progress(progress_path)
    if do_reset:
        try:
            if progress_path.exists():
                progress_path.unlink()
        except Exception:
            pass

    # 兼容旧参数，但默认不限科室（此工程以“合并CSV”目录为主）
    per_dept_max_rows = env_int("RAG_QA_PER_DEPT_MAX_ROWS", default=0) or 0

    dept_counts: Dict[str, int] = {}
    try:
        dc = progress.files.get("__dept_counts__")
        if isinstance(dc, dict):
            for k, v in dc.items():
                if not isinstance(k, str):
                    continue
                try:
                    dept_counts[k] = int(v)
                except Exception:
                    dept_counts[k] = 0
    except Exception:
        dept_counts = {}

    batch_size = int((env_str("RAG_INGEST_BATCH_SIZE", "256") or "256"))
    persist_every_batches = int((env_str("RAG_PERSIST_EVERY_N_BATCHES", "10") or "10"))
    progress_every_rows = int((env_str("RAG_PROGRESS_EVERY_ROWS", "2000") or "2000"))

    split_csv = env_flag("RAG_SPLIT_CSV", "0")
    csv_soft_limit = int((env_str("RAG_CSV_SOFT_MAX_CHARS", "6000") or "6000"))
    max_csv_rows = env_int("RAG_CSV_MAX_ROWS", default=None)

    batch: List[Any] = []
    chunk_counter = 0
    batches_written = 0

    try:
        count0 = opensearch_store.count()
    except Exception:
        count0 = None

    print(
        "[RAG_INGEST] "
        f"files_csv={len(csv_files)} files_txt={len(txt_files)} batch_size={batch_size} persist_every_batches={persist_every_batches} "
        f"resume={resume} reset={do_reset} csv_max_rows={max_csv_rows} initial_count={count0}",
        flush=True,
    )

    # Ingest TXT files (optional)
    if txt_files:
        for txt_path in txt_files:
            try:
                raw_txt = _read_text_file(txt_path)
            except Exception:
                continue
            # Hard gate markers on raw text
            if ("\x00" in raw_txt) or ("\ufffd" in raw_txt):
                _append_bad_row(
                    bad_rows_path,
                    source_file=str(txt_path.name),
                    row=None,
                    reason="hard_gate_reject_txt",
                    sample=raw_txt,
                )
                continue

            txt = _sanitize_text(raw_txt)
            if not _hard_gate(txt):
                _append_bad_row(
                    bad_rows_path,
                    source_file=str(txt_path.name),
                    row=None,
                    reason="hard_gate_reject_txt_after_sanitize",
                    sample=txt,
                )
                continue

            md = {
                "source_file": str(txt_path.name),
                "source": str(txt_path.name),
                "page": None,
                "section": "",
                "domain": "txt",
                "chunk_id": f"{txt_path.name}:0",
            }

            chunk_counter += 1
            payload = dict(md)
            payload["text"] = str(txt)
            batch.append(payload)
            if len(batch) >= batch_size:
                _flush_opensearch_batch(opensearch_store, embeddings, batch)
                batches_written += 1

    for csv_path in csv_files:
        # 对“合并CSV目录”里的文件，科室就是文件名（去后缀）
        dept = csv_path.stem

        if per_dept_max_rows > 0 and dept_counts.get(dept, 0) >= per_dept_max_rows:
            key = _progress_key(kb_dir, csv_path)
            st = progress.files.get(key, {})
            if not (isinstance(st, dict) and bool(st.get("done"))):
                progress.files[key] = {"last_row": int(st.get("last_row") or 0) if isinstance(st, dict) else 0, "done": True, "reason": "dept_limit_reached"}
                progress.files["__dept_counts__"] = dict(dept_counts)
                _save_progress(progress_path, progress)
            print(f"[SKIP] dept={dept} reached_limit={per_dept_max_rows} file={csv_path.name}", flush=True)
            continue

        key = _progress_key(kb_dir, csv_path)
        st = progress.files.get(key, {})
        if isinstance(st, dict) and bool(st.get("done")):
            reason = str(st.get("reason") or "")
            if not (reason == "dept_limit_reached" and (per_dept_max_rows == 0 or int(dept_counts.get(dept, 0) or 0) < per_dept_max_rows)):
                continue

        start_row = 0
        if isinstance(st, dict):
            try:
                start_row = int(st.get("last_row") or 0)
            except Exception:
                start_row = 0

        rows_since_save = 0
        last_seen_row = start_row

        print(f"[FILE] dept={dept} start file={key} start_row={start_row} dept_ingested={dept_counts.get(dept,0)}", flush=True)

        for row_idx, rd in _iter_csv_docs(
            csv_path,
            start_row=start_row,
            max_rows=max_csv_rows,
            bad_rows_path=bad_rows_path,
        ):
            last_seen_row = row_idx
            text = rd.text
            if not text:
                continue

            if per_dept_max_rows > 0 and dept_counts.get(dept, 0) >= per_dept_max_rows:
                break

            dept_counts[dept] = int(dept_counts.get(dept, 0)) + 1

            if (not split_csv) and len(text) <= csv_soft_limit:
                chunks = [text]
            else:
                chunks = _semantic_chunk_text(
                    text,
                    target_tokens=max(1, int(chunk_size)),
                    max_tokens=max(int(chunk_size), int(chunk_size) + max(0, int(chunk_overlap))),
                    max_chars=max(int(csv_soft_limit), int(chunk_size)),
                    overlap_tokens=max(0, int(chunk_overlap)),
                )

            for local_idx, chunk in enumerate(chunks):
                chunk_counter += 1
                md = dict(rd.metadata)
                md["department_group"] = dept
                md["source_path"] = key
                md["chunk_id"] = f"{rd.metadata.get('source','')}:{rd.metadata.get('row','')}:{local_idx}"
                payload = dict(md)
                payload["text"] = str(chunk)
                batch.append(payload)

                if len(batch) >= batch_size:
                    _flush_opensearch_batch(opensearch_store, embeddings, batch)
                    batches_written += 1

            rows_since_save += 1
            if rows_since_save >= progress_every_rows:
                progress.files[key] = {"last_row": int(last_seen_row), "done": False}
                progress.files["__dept_counts__"] = dict(dept_counts)
                _save_progress(progress_path, progress)
                rows_since_save = 0
                print(f"[PROGRESS] dept={dept} file={csv_path.name} last_row={last_seen_row} dept_ingested={dept_counts.get(dept,0)} chunks_total={chunk_counter}", flush=True)

        reason = "done"
        if per_dept_max_rows > 0 and dept_counts.get(dept, 0) >= per_dept_max_rows:
            reason = "dept_limit_reached"

        progress.files[key] = {"last_row": int(last_seen_row), "done": True, "reason": reason}
        progress.files["__dept_counts__"] = dict(dept_counts)
        _save_progress(progress_path, progress)
        print(f"[DONE_FILE] dept={dept} file={csv_path.name} last_row={last_seen_row} dept_ingested={dept_counts.get(dept,0)} reason={reason}", flush=True)

    if batch:
        _flush_opensearch_batch(opensearch_store, embeddings, batch)
    try:
        count1 = opensearch_store.count()
    except Exception:
        count1 = None

    print(
        "[RAG_INGEST] done "
        f"provider={emb_info.provider_used} model={emb_info.model_name} device={emb_info.device} "
        f"collection={collection_name} backend={_env_backend()} count={count1} chunks_ingested={chunk_counter}",
        flush=True,
    )
    return chunk_counter


def main() -> None:
    # ingest_kb.py位于 app/rag/ingest_kb.py
    app_dir = resolve_app_dir(Path(__file__))

    allowed = resolve_allowed_kb_dir(app_dir)
    kb_dir = resolve_kb_dir_strict(app_dir)
    persist_dir = resolve_persist_dir(app_dir)

    chunk_size = int(env_str("RAG_CHUNK_SIZE", "800") or "800")
    chunk_overlap = int(env_str("RAG_CHUNK_OVERLAP", "100") or "100")

    print(f"[RAG_INGEST] allowed_kb_dir={allowed}", flush=True)
    print(f"[RAG_INGEST] kb_dir={kb_dir}", flush=True)
    print(f"[RAG_INGEST] persist_dir={persist_dir}", flush=True)

    count = build_and_persist_store(
        kb_dir=kb_dir,
        persist_dir=persist_dir,
        collection_name=env_str("RAG_COLLECTION", DEFAULT_COLLECTION) or DEFAULT_COLLECTION,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model_name=None,
    )

    print(f"Ingested chunks: {count}", flush=True)


if __name__ == "__main__":
    main()
