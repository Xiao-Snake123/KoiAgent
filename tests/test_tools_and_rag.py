"""工具与 RAG 检索层测试（全程离线）。"""
import hashlib
import os
import re
import tempfile

import numpy as np
import pytest

from koiagent.agent.tools import get_bargain_policy, get_current_time
from koiagent.rag import knowledge
from koiagent.rag.knowledge import KnowledgeBase, _tokenize


# --------------------------------------------------------------------------- #
# Function Calling 工具
# --------------------------------------------------------------------------- #
def test_bargain_policy_returns_tiered_guidance():
    assert "首轮" in get_bargain_policy.invoke({"bargain_count": 1})
    assert "次轮" in get_bargain_policy.invoke({"bargain_count": 2})
    assert "三轮" in get_bargain_policy.invoke({"bargain_count": 3})
    assert "上限" in get_bargain_policy.invoke({"bargain_count": 99})


def test_current_time_format():
    value = get_current_time.invoke({})
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", value)


def test_tokenize_mixed_language():
    tokens = _tokenize("Type-C 蓝牙音响")
    assert "type-c" in tokens
    assert "蓝牙" in tokens


# --------------------------------------------------------------------------- #
# RAG：关键词模式
# --------------------------------------------------------------------------- #
def test_knowledge_base_keyword_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    kb = KnowledgeBase(store_path=str(tmp_path / "kw.json"))

    assert kb.mode == "keyword"
    assert kb.size > 0

    hits = kb.search("蓝牙 续航")
    assert hits
    assert any("蓝牙" in text for _, text in hits)


def test_empty_knowledge_dir_degrades_gracefully(monkeypatch, tmp_path):
    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path / "not-exist"))
    kb = KnowledgeBase(store_path=str(tmp_path / "vs.json"))

    assert kb.size == 0
    assert kb.search("任意查询") == []


# --------------------------------------------------------------------------- #
# RAG：向量模式（注入假 Embedding，不联网）
# --------------------------------------------------------------------------- #
class _FakeEmbeddings:
    """确定性假 Embedding：稳定哈希 + 大维度。

    切勿使用内置 ``hash()``：它受 ``PYTHONHASHSEED`` 随机化影响，
    会让向量排序在不同进程间不一致，把测试变成“随机失败”。
    """

    DIM = 2048

    @staticmethod
    def _bucket(token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % _FakeEmbeddings.DIM

    def _vec(self, text):
        vector = np.zeros(self.DIM, dtype="float32")
        for token in _tokenize(text):
            vector[self._bucket(token)] += 1.0
        return vector

    def embed_documents(self, texts):
        return [self._vec(t).tolist() for t in texts]

    def embed_query(self, text):
        return self._vec(text).tolist()


def test_knowledge_base_vector_mode_and_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_MODEL", "fake-embedding")
    monkeypatch.setattr(knowledge, "build_embeddings", lambda: _FakeEmbeddings())

    store = tmp_path / "vs.json"
    kb = KnowledgeBase(store_path=str(store))

    assert kb.mode == "vector"
    assert kb.matrix is not None
    assert kb.matrix.shape[0] == kb.size
    assert store.exists(), "向量缓存应落盘"

    hits = kb.search("Type-C", k=1)
    assert hits and "Type-C" in hits[0][1]

    # 二次加载应复用缓存，仍为向量模式
    kb2 = KnowledgeBase(store_path=str(store))
    assert kb2.mode == "vector"
    assert kb2.size == kb.size


def test_embedding_build_returns_none_without_model(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    assert knowledge.build_embeddings() is None
