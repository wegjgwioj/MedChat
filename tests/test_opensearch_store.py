# -*- coding: utf-8 -*-

from __future__ import annotations

from app.rag.opensearch_store import OpenSearchConfig, OpenSearchHybridStore


def _store() -> OpenSearchHybridStore:
    return OpenSearchHybridStore(
        OpenSearchConfig(
            url="http://localhost:9200",
            index_name="medical_kb",
            vector_dim=4,
            username="",
            password="",
            verify_ssl=False,
            knn_k=8,
            num_candidates=32,
            hybrid_mode="app_rrf",
        ),
        client=object(),
    )


def test_index_mapping_contains_text_keyword_and_vector_fields():
    mapping = _store().build_index_body()
    properties = mapping["mappings"]["properties"]

    assert properties["text"]["type"] == "text"
    assert properties["chunk_id"]["type"] == "keyword"
    assert properties["department"]["type"] == "keyword"
    assert properties["embedding"]["type"] == "knn_vector"
    assert properties["embedding"]["dimension"] == 4


def test_bulk_operations_include_document_id_metadata_and_embedding():
    operations = _store().build_bulk_operations(
        [
            {
                "chunk_id": "chunk-1",
                "text": "发热伴咽痛两天",
                "embedding": [0.1, 0.2, 0.3, 0.4],
                "source": "qa.csv",
                "source_file": "qa.csv",
                "department": "内科",
                "title": "感冒问答",
                "row": 12,
            }
        ]
    )

    assert operations[0]["index"]["_index"] == "medical_kb"
    assert operations[0]["index"]["_id"] == "chunk-1"
    assert operations[1]["text"] == "发热伴咽痛两天"
    assert operations[1]["embedding"] == [0.1, 0.2, 0.3, 0.4]
    assert operations[1]["department"] == "内科"
    assert operations[1]["row"] == 12


def test_hybrid_query_body_contains_bm25_knn_and_optional_filter():
    body = _store().build_search_body(
        query="发热咽痛",
        query_vector=[0.2, 0.3, 0.4, 0.5],
        top_k=5,
        department="内科",
    )

    assert body["size"] == 5
    assert body["query"]["bool"]["should"][0]["match"]["text"]["query"] == "发热咽痛"
    assert body["query"]["bool"]["should"][1]["knn"]["embedding"]["vector"] == [0.2, 0.3, 0.4, 0.5]
    assert body["query"]["bool"]["filter"][0]["term"]["department"] == "内科"
