# -*- coding: utf-8 -*-

from __future__ import annotations

import pickle
from pathlib import Path

import pytest


def test_eval_meddg_cli_supports_split_and_limit(tmp_path: Path):
    from scripts import eval_meddg_e2e

    meddg_dir = tmp_path / "MedDG_UTF8"
    meddg_dir.mkdir()
    sample_path = meddg_dir / "test.pk"
    with sample_path.open("wb") as f:
        pickle.dump([], f)

    parser = eval_meddg_e2e.build_arg_parser()
    args = parser.parse_args(["--meddg_dir", str(meddg_dir), "--split", "test", "--limit", "12"])

    assert args.n == 12
    assert eval_meddg_e2e.resolve_meddg_path(args) == sample_path


def test_eval_rag_quality_cli_supports_split_and_limit(tmp_path: Path):
    from scripts import eval_rag_quality

    meddg_dir = tmp_path / "MedDG_UTF8"
    meddg_dir.mkdir()
    sample_path = meddg_dir / "test.pk"
    with sample_path.open("wb") as f:
        pickle.dump([], f)

    parser = eval_rag_quality.build_arg_parser()
    args = parser.parse_args(["--meddg_dir", str(meddg_dir), "--split", "test", "--limit", "25"])

    assert args.n == 25
    assert eval_rag_quality.resolve_meddg_path(args) == sample_path


def test_eval_cli_raises_clear_error_when_meddg_file_missing(tmp_path: Path):
    from scripts import eval_meddg_e2e

    parser = eval_meddg_e2e.build_arg_parser()
    args = parser.parse_args(["--meddg_dir", str(tmp_path / "missing_dir"), "--split", "test"])

    with pytest.raises(FileNotFoundError) as exc:
        eval_meddg_e2e.resolve_meddg_path(args)

    message = str(exc.value)
    assert "MedDG" in message
    assert "test.pk" in message


def test_eval_perf_concurrency_parser_accepts_space_or_csv():
    from scripts import eval_perf

    assert eval_perf._parse_concurrency_values(["1", "5", "10"]) == [1, 5, 10]
    assert eval_perf._parse_concurrency_values(["1,5,10"]) == [1, 5, 10]
    assert eval_perf._parse_concurrency_values(["1,5", "10"]) == [1, 5, 10]


def test_eval_perf_limit_alias_maps_to_requests():
    from scripts import eval_perf

    parser = eval_perf.build_arg_parser()
    args = parser.parse_args(["--limit", "40", "--concurrency", "1", "5", "10"])

    assert args.requests == 40
    assert eval_perf._parse_concurrency_values(args.concurrency) == [1, 5, 10]
