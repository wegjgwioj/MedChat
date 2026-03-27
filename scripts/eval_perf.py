import argparse
import asyncio
import json
import pickle
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx


@dataclass
class Result:
    ok: bool
    latency_ms: float
    status_code: Optional[int]
    error: Optional[str]


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = int(round((p / 100.0) * (len(values_sorted) - 1)))
    idx = max(0, min(len(values_sorted) - 1, idx))
    return float(values_sorted[idx])


def _load_meddg_patient_sentences(meddg_path: Path, limit: int = 5000) -> List[str]:
    with meddg_path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"MedDG pickle top-level must be list, got: {type(obj)}")

    # Print a tiny schema peek for reproducibility.
    print(f"[MedDG] loaded_from={meddg_path}")
    print("[MedDG] dialogs=", len(obj))
    if obj:
        d0 = obj[0]
        print("[MedDG] dialog[0] type=", type(d0))
        if isinstance(d0, list) and d0:
            t0 = d0[0]
            print("[MedDG] dialog[0].turn[0] type=", type(t0))
            if isinstance(t0, dict):
                print("[MedDG] dialog[0].turn[0] keys=", sorted(list(t0.keys())))

    out: List[str] = []
    for dialog in obj:
        if not isinstance(dialog, list):
            continue
        for turn in dialog:
            if not isinstance(turn, dict):
                continue
            if turn.get("id") != "Patients":
                continue
            s = turn.get("Sentence")
            if isinstance(s, str) and s.strip():
                out.append(s.strip())
                if len(out) >= limit:
                    return out
    return out


async def _one_request(client: httpx.AsyncClient, url: str, json_payload: Dict[str, Any], timeout_s: float) -> Result:
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=json_payload, timeout=timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code >= 400:
            return Result(ok=False, latency_ms=latency_ms, status_code=resp.status_code, error=resp.text[:2000])
        return Result(ok=True, latency_ms=latency_ms, status_code=resp.status_code, error=None)
    except Exception as e:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return Result(ok=False, latency_ms=latency_ms, status_code=None, error=str(e))


async def run_load_test(
    *,
    url: str,
    payloads: Sequence[Dict[str, Any]],
    concurrency: int,
    total_requests: int,
    timeout_s: float,
) -> Dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    results: List[Result] = []

    async with httpx.AsyncClient() as client:

        async def run_one(i: int) -> None:
            payload = payloads[i % len(payloads)]
            async with sem:
                res = await _one_request(client, url, payload, timeout_s)
                results.append(res)

        t0 = time.perf_counter()
        await asyncio.gather(*(run_one(i) for i in range(total_requests)))
        wall_ms = (time.perf_counter() - t0) * 1000.0

    ok_lat = [r.latency_ms for r in results if r.ok]
    all_lat = [r.latency_ms for r in results]
    err = [r for r in results if not r.ok]

    return {
        "url": url,
        "concurrency": concurrency,
        "requests": total_requests,
        "ok": len(ok_lat),
        "errors": len(err),
        "error_rate": (len(err) / len(results)) if results else 0.0,
        "avg_ms": statistics.mean(all_lat) if all_lat else 0.0,
        "p95_ms": _percentile(all_lat, 95),
        "avg_ok_ms": statistics.mean(ok_lat) if ok_lat else 0.0,
        "p95_ok_ms": _percentile(ok_lat, 95),
        "wall_ms": wall_ms,
        "rps": (len(results) / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0,
        "sample_error": (err[0].error if err else None),
        "sample_status": (err[0].status_code if err else None),
    }


def _parse_concurrency_list(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        v = int(p)
        if v <= 0:
            continue
        out.append(v)
    return out or [1]


def _parse_concurrency_values(values: Any) -> List[int]:
    if values is None:
        return [1, 5, 10]
    if isinstance(values, str):
        return _parse_concurrency_list(values)
    if isinstance(values, (list, tuple)):
        parts: List[str] = []
        for value in values:
            if value is None:
                continue
            parts.append(str(value))
        if not parts:
            return [1, 5, 10]
        merged = ",".join(parts)
        return _parse_concurrency_list(merged)
    return _parse_concurrency_list(str(values))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight perf eval (async httpx) for rag + agent endpoints")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--concurrency", nargs="+", default=["1", "5", "10"], help="Concurrency list, e.g. 1 5 10 or 1,5,10")
    parser.add_argument("--requests", "--limit", dest="requests", type=int, default=20, help="Requests per concurrency per endpoint")
    parser.add_argument("--timeout_s", type=float, default=30.0, help="HTTP timeout per request")
    parser.add_argument("--top_k", type=int, default=5, help="RAG top_k")
    parser.add_argument("--meddg_path", default=None, help="Optional MedDG pickle for sampling queries")
    parser.add_argument("--sample_queries", type=int, default=50, help="How many patient sentences to sample as queries")
    parser.add_argument("--out_dir", default="reports", help="Output directory")
    return parser


def _make_rag_payloads(queries: List[str], top_k: int) -> List[Dict[str, Any]]:
    if not queries:
        queries = ["头疼怎么办", "发烧38.5怎么处理", "风疹病毒怎么感染", "皮疹伴发热", "咳嗽两周"]
    return [{"query": q, "top_k": top_k} for q in queries]


def _make_agent_payloads(queries: List[str]) -> List[Dict[str, Any]]:
    if not queries:
        queries = ["我头疼三天了", "发烧39度", "风疹病毒怎么感染", "咳嗽两周", "皮疹很痒"]
    payloads: List[Dict[str, Any]] = []
    base = f"perf_{int(time.time())}_{random.randint(1000, 9999)}"
    for i, q in enumerate(queries):
        payloads.append({"session_id": f"{base}_{i}", "user_message": q, "top_k": 5, "top_n": 30, "use_rerank": True})
    return payloads


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries: List[str] = []
    if args.meddg_path:
        try:
            all_q = _load_meddg_patient_sentences(Path(args.meddg_path), limit=max(200, args.sample_queries))
            random.Random(42).shuffle(all_q)
            queries = all_q[: args.sample_queries]
            print(f"[MedDG] sampled_patient_sentences={len(queries)}")
            if queries:
                print(f"[MedDG] sample_query_0={queries[0][:60]}")
        except Exception as e:  # noqa: BLE001
            print(f"warn: failed to load MedDG queries: {e}")
            queries = []

    rag_url = f"{args.base_url}/v1/rag/retrieve"
    agent_url = f"{args.base_url}/v1/agent/chat_v2"

    concurrencies = _parse_concurrency_values(args.concurrency)

    rag_payloads = _make_rag_payloads(queries, args.top_k)
    agent_payloads = _make_agent_payloads(queries)

    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": args.base_url,
        "concurrency": concurrencies,
        "requests_per_concurrency": args.requests,
        "timeout_s": args.timeout_s,
        "endpoints": {},
    }

    for name, url, payloads in [
        ("rag_retrieve", rag_url, rag_payloads),
        ("agent_chat_v2", agent_url, agent_payloads),
    ]:
        endpoint_results: List[Dict[str, Any]] = []
        for c in concurrencies:
            res = asyncio.run(
                run_load_test(
                    url=url,
                    payloads=payloads,
                    concurrency=c,
                    total_requests=args.requests,
                    timeout_s=args.timeout_s,
                )
            )
            endpoint_results.append(res)
            print(f"{name} concurrency={c} rps={res['rps']:.2f} err_rate={res['error_rate']:.2%} p95_ms={res['p95_ms']:.2f}")
        report["endpoints"][name] = endpoint_results

    out_path = out_dir / "perf_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
