# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path


def test_build_plan_respects_skip_flags(tmp_path: Path):
    from scripts import eval_run_all

    plan = eval_run_all.build_run_plan(
        base_url="http://127.0.0.1:8000",
        meddg_path=tmp_path / "test.pk",
        out_dir=tmp_path / "reports",
        skip_meddg=False,
        skip_rag=True,
        skip_perf=False,
    )

    names = [item["name"] for item in plan]
    assert names == ["meddg_e2e", "perf"]
    assert any("eval_meddg_e2e.py" in part for part in plan[0]["command"])
    assert any("eval_perf.py" in part for part in plan[1]["command"])


def test_collect_suite_summary_reads_existing_outputs(tmp_path: Path):
    from scripts import eval_run_all

    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    (out_dir / "meddg_eval_summary.json").write_text(json.dumps({"answer_rate": 0.8}), encoding="utf-8")
    (out_dir / "perf_eval.json").write_text(json.dumps({"endpoints": {"agent": {"p95_ms": 123}}}), encoding="utf-8")

    summary = eval_run_all.collect_suite_summary(
        out_dir=out_dir,
        executed_steps=["meddg_e2e", "perf"],
        skipped_steps=["rag_quality"],
    )

    assert summary["executed_steps"] == ["meddg_e2e", "perf"]
    assert summary["skipped_steps"] == ["rag_quality"]
    assert summary["artifacts"]["meddg_e2e"]["answer_rate"] == 0.8
    assert summary["artifacts"]["perf"]["endpoints"]["agent"]["p95_ms"] == 123


def test_main_runs_commands_and_writes_suite_summary(monkeypatch, tmp_path: Path):
    from scripts import eval_run_all

    meddg_path = tmp_path / "test.pk"
    meddg_path.write_bytes(b"stub")
    out_dir = tmp_path / "reports"
    out_dir.mkdir()

    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        if any("eval_meddg_e2e.py" in part for part in cmd):
            (out_dir / "meddg_eval_summary.json").write_text(json.dumps({"answer_rate": 0.9}), encoding="utf-8")
        if any("eval_perf.py" in part for part in cmd):
            (out_dir / "perf_eval.json").write_text(json.dumps({"endpoints": {}}), encoding="utf-8")
        return 0

    monkeypatch.setattr(eval_run_all.subprocess, "run", fake_run)

    rc = eval_run_all.main(
        [
            "--base_url",
            "http://127.0.0.1:8000",
            "--meddg_path",
            str(meddg_path),
            "--out_dir",
            str(out_dir),
            "--skip_rag",
        ]
    )

    assert rc == 0
    assert len(calls) == 2
    suite_path = out_dir / "eval_suite_summary.json"
    assert suite_path.exists()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    assert suite["executed_steps"] == ["meddg_e2e", "perf"]
    assert suite["skipped_steps"] == ["rag_quality"]
