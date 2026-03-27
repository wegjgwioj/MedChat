import argparse
import csv
import json
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx


@dataclass
class TurnResult:
    dialog_idx: int
    turn_idx: int
    user_text: str
    mode: str
    latency_ms: float
    citations: int
    rag_hits: Optional[int]
    rag_latency_ms: Optional[float]
    planner_strategy: Optional[str]
    evidence_quality_level: Optional[str]
    evidence_quality_reason: Optional[str]
    error: Optional[str]


def _safe_get(d: Any, *path: str) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _iter_patient_utterances(dialog: Any) -> Iterable[str]:
    if not isinstance(dialog, list):
        return
    for turn in dialog:
        if not isinstance(turn, dict):
            continue
        if turn.get("id") != "Patients":
            continue
        s = turn.get("Sentence")
        if isinstance(s, str) and s.strip():
            yield s.strip()


def _load_meddg_dialogs(path: Path) -> List[Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"MedDG pickle top-level must be list, got: {type(obj)}")
    return obj


def _describe_meddg(obj: Any, *, max_dialogs: int = 2, max_turns: int = 3) -> None:
    """Print a defensive, human-readable schema peek for MedDG-like pickles.

    Requirement:
    - Must print sample count and field names.
    - Must NOT assume the format is correct.
    """
    print("[MedDG] top_type=", type(obj))
    if isinstance(obj, list):
        print("[MedDG] dialogs=", len(obj))
        for i, d in enumerate(obj[:max_dialogs]):
            print(f"[MedDG] dialog[{i}] type={type(d)}")
            if isinstance(d, list):
                print(f"[MedDG] dialog[{i}] turns={len(d)}")
                for j, t in enumerate(d[:max_turns]):
                    if isinstance(t, dict):
                        keys = sorted(list(t.keys()))
                        print(f"[MedDG] dialog[{i}].turn[{j}] keys={keys}")
                        # print a tiny type preview for the most common fields
                        for k in ["id", "Sentence"]:
                            if k in t:
                                v = t.get(k)
                                print(f"  - {k}: type={type(v)}")
                    else:
                        print(f"[MedDG] dialog[{i}].turn[{j}] type={type(t)}")
            else:
                try:
                    ln = len(d)  # type: ignore[arg-type]
                    print(f"[MedDG] dialog[{i}] len={ln}")
                except Exception:
                    pass
    else:
        try:
            ln = len(obj)  # type: ignore[arg-type]
            print("[MedDG] len=", ln)
        except Exception:
            pass


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = max(0, min(len(values_sorted) - 1, int(round((p / 100.0) * (len(values_sorted) - 1)))))
    return float(values_sorted[k])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MedDG E2E evaluation via /v1/agent/chat_v2")
    parser.add_argument("--meddg_path", default=None, help="Optional explicit MedDG pickle path")
    parser.add_argument("--meddg_dir", default="app/MedDG_UTF8", help="Directory containing MedDG split pickles")
    parser.add_argument("--split", default="test", help="MedDG split name, e.g. test/train/dev")
    parser.add_argument("--n", "--limit", dest="n", type=int, default=100, help="Number of dialogs to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_turns", type=int, default=8, help="Max patient turns per dialog")
    parser.add_argument("--timeout_s", type=float, default=60.0, help="HTTP timeout per request")
    parser.add_argument("--top_k", type=int, default=5, help="Agent RAG top_k")
    parser.add_argument("--top_n", type=int, default=30, help="Agent RAG top_n")
    parser.add_argument("--use_rerank", type=int, default=1, help="Use rerank: 1/0")
    parser.add_argument("--out_dir", default="reports", help="Output directory")
    return parser


def resolve_meddg_path(args: argparse.Namespace) -> Path:
    if getattr(args, "meddg_path", None):
        meddg_path = Path(args.meddg_path)
    else:
        meddg_path = Path(args.meddg_dir) / f"{args.split}.pk"
    if not meddg_path.exists():
        raise FileNotFoundError(
            f"MedDG 数据文件不存在：{meddg_path}。"
            "可通过 --meddg_path 显式指定，或准备 app/MedDG_UTF8/<split>.pk 后再运行。"
        )
    return meddg_path


def run_dialog(
    client: httpx.Client,
    base_url: str,
    dialog_idx: int,
    user_turns: List[str],
    max_turns: int,
    timeout_s: float,
    top_k: int,
    top_n: int,
    use_rerank: bool,
) -> Tuple[List[TurnResult], Optional[str]]:
    session_id = f"meddg_eval_{int(time.time())}_{random.randint(1000, 9999)}_{dialog_idx}"
    results: List[TurnResult] = []

    for turn_idx, user_text in enumerate(user_turns[:max_turns]):
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "user_message": user_text,
            "top_k": int(top_k),
            "top_n": int(top_n),
            "use_rerank": bool(use_rerank),
        }
        t0 = time.perf_counter()
        try:
            r = client.post(f"{base_url}/v1/agent/chat_v2", json=payload, timeout=timeout_s)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            r.raise_for_status()
            data = r.json()

            mode = str(data.get("mode") or "")
            citations = data.get("citations")
            if not isinstance(citations, list):
                citations_n = 0
            else:
                citations_n = len(citations)

            rag_hits = _safe_get(data, "trace", "rag_stats", "hits")
            rag_latency_ms = _safe_get(data, "trace", "rag_stats", "latency_ms")
            if isinstance(rag_hits, (int, float)):
                rag_hits_v = int(rag_hits)
            else:
                rag_hits_v = None
            if isinstance(rag_latency_ms, (int, float)):
                rag_latency_v = float(rag_latency_ms)
            else:
                rag_latency_v = None

            planner_strategy = _safe_get(data, "trace", "planner_strategy")
            if planner_strategy is not None and not isinstance(planner_strategy, str):
                planner_strategy = str(planner_strategy)

            evidence_quality_level = _safe_get(data, "trace", "rag_stats", "evidence_quality", "level")
            if evidence_quality_level is not None and not isinstance(evidence_quality_level, str):
                evidence_quality_level = str(evidence_quality_level)

            evidence_quality_reason = _safe_get(data, "trace", "rag_stats", "evidence_quality", "reason")
            if evidence_quality_reason is not None and not isinstance(evidence_quality_reason, str):
                evidence_quality_reason = str(evidence_quality_reason)

            results.append(
                TurnResult(
                    dialog_idx=dialog_idx,
                    turn_idx=turn_idx,
                    user_text=user_text,
                    mode=mode,
                    latency_ms=latency_ms,
                    citations=citations_n,
                    rag_hits=rag_hits_v,
                    rag_latency_ms=rag_latency_v,
                    planner_strategy=planner_strategy,
                    evidence_quality_level=evidence_quality_level,
                    evidence_quality_reason=evidence_quality_reason,
                    error=None,
                )
            )
        except Exception as e:  # noqa: BLE001
            latency_ms = (time.perf_counter() - t0) * 1000.0
            results.append(
                TurnResult(
                    dialog_idx=dialog_idx,
                    turn_idx=turn_idx,
                    user_text=user_text,
                    mode="error",
                    latency_ms=latency_ms,
                    citations=0,
                    rag_hits=None,
                    rag_latency_ms=None,
                    planner_strategy=None,
                    evidence_quality_level=None,
                    evidence_quality_reason=None,
                    error=str(e),
                )
            )
            return results, str(e)

    return results, None


def summarize(turns: List[TurnResult]) -> Dict[str, Any]:
    if not turns:
        return {
            "turns": 0,
            "answer_rate": 0.0,
            "ask_rate": 0.0,
            "escalate_rate": 0.0,
            "error_rate": 0.0,
            "citation_rate": 0.0,
            "hit_rate": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }

    total = len(turns)
    answer = sum(1 for t in turns if t.mode == "answer")
    ask = sum(1 for t in turns if t.mode == "ask")
    escalate = sum(1 for t in turns if t.mode == "escalate")
    error = sum(1 for t in turns if t.mode == "error")

    answer_turns = [t for t in turns if t.mode == "answer"]
    citation_rate = 0.0
    if answer_turns:
        citation_rate = sum(1 for t in answer_turns if (t.citations or 0) > 0) / len(answer_turns)

    hit_rate = sum(1 for t in turns if (t.rag_hits or 0) > 0) / total
    quality_counts: Dict[str, int] = {}
    low_quality = 0
    for t in turns:
        level = str(t.evidence_quality_level or "").strip()
        if not level:
            continue
        quality_counts[level] = quality_counts.get(level, 0) + 1
        if level in {"low", "none"}:
            low_quality += 1

    latencies = [t.latency_ms for t in turns]

    return {
        "turns": total,
        "answer_rate": answer / total,
        "ask_rate": ask / total,
        "escalate_rate": escalate / total,
        "error_rate": error / total,
        "citation_rate": citation_rate,
        "hit_rate": hit_rate,
        "evidence_quality_counts": quality_counts,
        "low_evidence_rate": (low_quality / total) if total else 0.0,
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p95_latency_ms": _percentile(latencies, 95),
    }


def compute_avg_ask_before_first_answer(per_dialog: Dict[int, List[TurnResult]]) -> float:
    values: List[int] = []
    for _, turns in per_dialog.items():
        ask_before = 0
        seen_answer = False
        for t in turns:
            if t.mode == "answer":
                seen_answer = True
                break
            if t.mode == "ask":
                ask_before += 1
        if seen_answer:
            values.append(ask_before)
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> int:
    parser = build_arg_parser()
    parser.add_argument("--base_url", default="http://127.0.0.1:8000", help="API base URL")
    args = parser.parse_args()

    meddg_path = resolve_meddg_path(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dialogs = _load_meddg_dialogs(meddg_path)
    print(f"[MedDG] loaded_from={meddg_path}")
    _describe_meddg(dialogs)
    rng = random.Random(args.seed)

    indices = list(range(len(dialogs)))
    rng.shuffle(indices)
    indices = indices[: min(args.n, len(indices))]

    all_turns: List[TurnResult] = []
    per_dialog: Dict[int, List[TurnResult]] = {}

    with httpx.Client() as client:
        for i, dialog_idx in enumerate(indices):
            dialog = dialogs[dialog_idx]
            user_turns = list(_iter_patient_utterances(dialog))
            if not user_turns:
                continue

            turns, err = run_dialog(
                client=client,
                base_url=args.base_url,
                dialog_idx=dialog_idx,
                user_turns=user_turns,
                max_turns=args.max_turns,
                timeout_s=args.timeout_s,
                top_k=args.top_k,
                top_n=args.top_n,
                use_rerank=bool(int(args.use_rerank)),
            )
            all_turns.extend(turns)
            per_dialog[dialog_idx] = turns

            if (i + 1) % 10 == 0:
                print(f"progress: {i + 1}/{len(indices)} dialogs")
            if err is not None:
                print(f"dialog {dialog_idx} aborted due to error: {err}")

    summary = summarize(all_turns)
    summary["dialogs_evaluated"] = len(per_dialog)
    summary["avg_ask_turns_before_answer"] = compute_avg_ask_before_first_answer(per_dialog)

    # strategy distribution
    strategy_counts: Dict[str, int] = {}
    for t in all_turns:
        if t.planner_strategy:
            strategy_counts[t.planner_strategy] = strategy_counts.get(t.planner_strategy, 0) + 1
    summary["planner_strategy_counts"] = strategy_counts

    summary_path = out_dir / "meddg_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Keep existing filename for backwards compatibility.
    cases_path = out_dir / "meddg_eval_cases.csv"
    with cases_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dialog_idx",
                "turn_idx",
                "user_text",
                "mode",
                "latency_ms",
                "citations",
                "rag_hits",
                "rag_latency_ms",
                "planner_strategy",
                "evidence_quality_level",
                "evidence_quality_reason",
                "error",
            ]
        )
        for t in all_turns:
            w.writerow(
                [
                    t.dialog_idx,
                    t.turn_idx,
                    t.user_text,
                    t.mode,
                    f"{t.latency_ms:.2f}",
                    t.citations,
                    "" if t.rag_hits is None else t.rag_hits,
                    "" if t.rag_latency_ms is None else f"{t.rag_latency_ms:.2f}",
                    t.planner_strategy or "",
                    t.evidence_quality_level or "",
                    t.evidence_quality_reason or "",
                    t.error or "",
                ]
            )

    # Also emit a generic filename as requested by some runbooks.
    cases_alias = out_dir / "cases.csv"
    try:
        cases_alias.write_text(cases_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    print(f"wrote: {summary_path}")
    print(f"wrote: {cases_path}")
    if cases_alias.exists():
        print(f"wrote: {cases_alias}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
