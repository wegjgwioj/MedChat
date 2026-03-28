# -*- coding: utf-8 -*-

from __future__ import annotations


def _word_tokens(text: str) -> int:
    return len([part for part in str(text).split(" ") if part])


def test_semantic_chunking_keeps_sentence_boundaries_before_fallback() -> None:
    from app.rag.ingest_kb import _semantic_chunk_text

    text = (
        "第一句讲胃痛和反酸。"
        "第二句讲饮食调整和休息。"
        "第三句讲出现黑便要及时就医。"
    )

    chunks = _semantic_chunk_text(
        text,
        target_tokens=24,
        max_tokens=40,
        max_chars=30,
        token_counter=len,
    )

    assert len(chunks) == 2
    assert chunks[0].endswith("休息。")
    assert chunks[1] == "第三句讲出现黑便要及时就医。"


def test_semantic_chunking_falls_back_to_token_windows_for_oversized_unit() -> None:
    from app.rag.ingest_kb import _semantic_chunk_text

    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"

    chunks = _semantic_chunk_text(
        text,
        target_tokens=4,
        max_tokens=5,
        max_chars=200,
        token_counter=_word_tokens,
    )

    assert len(chunks) >= 3
    assert "alpha beta gamma delta" in chunks[0]
    assert all(_word_tokens(chunk) <= 5 for chunk in chunks)
    rebuilt = " ".join(chunk.replace("\n", " ").strip() for chunk in chunks)
    assert "alpha beta gamma" in rebuilt
    assert "lambda mu" in rebuilt


def test_semantic_chunking_preserves_structured_headers_with_body() -> None:
    from app.rag.ingest_kb import _semantic_chunk_text

    text = (
        "科室：内科\n"
        "主题：胃痛反酸处理\n"
        "患者问题：胃痛反酸怎么办\n"
        "医生回答：建议少量多餐，避免辛辣油腻；如果出现黑便、呕血或持续剧痛，应尽快线下评估。"
    )

    chunks = _semantic_chunk_text(
        text,
        target_tokens=80,
        max_tokens=120,
        max_chars=300,
        token_counter=len,
    )

    assert chunks == [text]
