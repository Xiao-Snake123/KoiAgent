"""共享文本工具：分词 / 向量归一化 / 内容哈希。

抽到 ``infra`` 而不是留在 ``rag`` 里，是因为**记忆检索也需要同一套分词**。
放在下层可以让 ``rag`` 与 ``memory`` 各自引用而不互相依赖。
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, List

# 中文按 2-gram；英文/型号按空格切分并小写。**保留连字符**，否则 `Type-C` 会被拆成
# `Type` 和 `C`，导致检索 "Type-C" 命中率下降（这是修过的 bug）。
_TOKEN_SPLIT = re.compile(r"[^\w\u4e00-\u9fa5-]+")
_HAS_CJK = re.compile(r"[\u4e00-\u9fa5]")


def tokenize(text: str) -> List[str]:
    """中英文混合轻量分词：中文按 2-gram，英文/型号整体保留并小写。"""
    if not text:
        return []
    cleaned = _TOKEN_SPLIT.sub(" ", str(text))
    tokens: List[str] = []
    for word in cleaned.split():
        word = word.strip("-")
        if not word:
            continue
        if _HAS_CJK.search(word):
            if len(word) <= 2:
                tokens.append(word)
            else:
                tokens.extend(word[i : i + 2] for i in range(len(word) - 1))
        else:
            tokens.append(word.lower())
    return tokens


def l2_normalize(matrix: Any) -> Any:
    """L2 归一化，使点积等价于余弦相似度。需要 numpy（惰性导入）。"""
    import numpy as np

    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def content_hash(text: str, algorithm: str = "sha1") -> str:
    """内容哈希（**稳定**，不依赖 ``PYTHONHASHSEED``）。

    注意：绝不能用内置 ``hash()`` —— 它受进程级哈希随机化影响，
    会让依赖排序的检索结果在不同进程间不一致。
    """
    return hashlib.new(algorithm, (text or "").encode("utf-8")).hexdigest()


def keyword_overlap(query: str, document: str) -> int:
    """查询与文档的词元交集大小（轻量相关性打分）。"""
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0
    return len(query_tokens & set(tokenize(document)))


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + suffix
