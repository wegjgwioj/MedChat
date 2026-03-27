# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_STEP_TO_ARTIFACT = {
    "meddg_e2e": "meddg_eval_summary.json",
    "rag_quality": "rag_eval_summary.json",
    "perf": "perf_eval.json",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MedChat evaluation suite and aggregate summaries")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--meddg_path", default=None, help="Optional explicit MedDG pickle path")
    parser.add_argument("--out_dir", default="reports", help="Output directory")
    parser.add_argument("--skip_meddg", action="store_true", help="Skip MedDG end-to-end evaluation")
    parser.add_argument("--skip_rag", action="store_true", help="Skip RAG quality evaluation")
    parser.add_argument("--skip_perf", action="store_true", help="Skip perf evaluation")
    return parser


def build_run_plan(
    *,
    base_url: str,
    meddg_path: Optional[Path],
    out_dir: Path,
    skip_meddg: bool,
    skip_rag: bool,
    skip_perf: bool,
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []

    def _base_cmd(script_name: str) -> List[str]:
        cmd = [sys.executable, f"scripts/{script_name}", "--base_url", str(base_url), "--out_dir", str(out_dir)]
        if meddg_path is not None:
            cmd.extend(["--meddg_path", str(meddg_path)])
        return cmd

    if not skip_meddg:
        plan.append({"name": "meddg_e2e", "command": _base_cmd("eval_meddg_e2e.py")})
    if not skip_rag:
        plan.append({"name": "rag_quality", "command": _base_cmd("eval_rag_quality.py")})
    if not skip_perf:
        plan.append({"name": "perf", "command": _base_cmd("eval_perf.py")})

    return plan


def collect_suite_summary(*, out_dir: Path, executed_steps: List[str], skipped_steps: List[str]) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {}
    for step in executed_steps:
        artifact_name = _STEP_TO_ARTIFACT.get(step)
        if not artifact_name:
            continue
        path = out_dir / artifact_name
        if path.exists():
            artifacts[step] = json.loads(path.read_text(encoding="utf-8"))
        else:
            artifacts[step] = {"missing": artifact_name}

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": str(out_dir),
        "executed_steps": list(executed_steps),
        "skipped_steps": list(skipped_steps),
        "artifacts": artifacts,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meddg_path = Path(args.meddg_path) if args.meddg_path else None

    plan = build_run_plan(
        base_url=args.base_url,
        meddg_path=meddg_path,
        out_dir=out_dir,
        skip_meddg=bool(args.skip_meddg),
        skip_rag=bool(args.skip_rag),
        skip_perf=bool(args.skip_perf),
    )

    executed_steps: List[str] = []
    for step in plan:
        subprocess.run(step["command"], check=True)
        executed_steps.append(str(step["name"]))

    skipped_steps = [name for name in _STEP_TO_ARTIFACT if name not in executed_steps]
    suite_summary = collect_suite_summary(out_dir=out_dir, executed_steps=executed_steps, skipped_steps=skipped_steps)
    suite_path = out_dir / "eval_suite_summary.json"
    suite_path.write_text(json.dumps(suite_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[eval_suite] summary_written={suite_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
