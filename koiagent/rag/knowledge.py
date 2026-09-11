"""KoiAgent 的 RAG（检索增强生成）检索层。

能力
----
1. **文档加载与切分**：递归切分 ``knowledge/`` 下的 ``.md`` / ``.txt`` 文档。
2. **向量化**：基于 LangChain ``Embeddings`` 抽象（OpenAI 兼容协议），
   通过 ``EMBEDDING_MODEL`` 配置，可指向通义千问 / OpenAI / 本地任意兼容端点。
3. **向量索引**：numpy 余弦相似度检索，向量**落盘缓存**，避免每次启动重复调用
   Embedding 接口产生费用。
4. **优雅降级**：未配置 ``EMBEDDING_MODEL`` 或初始化失败时，自动回退到
   **纯本地关键词检索**（中文 bigram 分词 + 命中打分），**无需任何 API** 即可运行。
5. **可插拔**：``KnowledgeBase.search`` 是对外唯一入口，后续替换为
   Chroma / FAISS / Milvus 只需改这一个类。

环境变量
--------
- ``EMBEDDING_MODEL``      向量模型名；**留空则使用关键词检索**（默认留空）
- ``EMBEDDING_BASE_URL``   向量接口地址；留空回退 ``MODEL_BASE_URL``
- ``EMBEDDING_API_KEY``    向量接口密钥；留空回退 ``API_KEY``
- ``RAG_TOP_K``            返回片段数，默认 3
- ``RAG_CHUNK_SIZE``       切片长度，默认 300
- ``RAG_CHUNK_OVERLAP``    切片重叠，默认 60
- ``RAG_STORE_PATH``       向量缓存路径，默认 ``data/vector_store.json``
- ``KNOWLEDGE_DIR``        知识库目录，默认 ``knowledge``
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from loguru import logger

try:  # 优先使用 LangChain 官方切分器
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    _HAS_SPLITTER = True
except ImportError:  # pragma: no cover - 无该依赖时走内置兜底切分
    _HAS_SPLITTER = False


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _tokenize(text: str) -> List[str]:
    """中英文混合轻量分词：中文按 2-gram，英文/型号按空格切分并小写（保留连字符）。"""
    text = re.sub(r"[^\w\u4e00-\u9fa5-]+", " ", text)
    tokens: List[str] = []
    for word in text.split():
        word = word.strip("-")
        if not word:
            continue
        if re.search(r"[\u4e00-\u9fa5]", word):
            if len(word) <= 2:
                tokens.append(word)
            else:
                tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
        else:
            tokens.append(word.lower())
    return tokens


def _l2_normalize(matrix: Any) -> Any:
    """L2 归一化，使点积等价于余弦相似度。"""
    import numpy as np

    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# --------------------------------------------------------------------------- #
# 文档加载 / 切分
# --------------------------------------------------------------------------- #
def load_documents(kb_dir: str) -> List[Document]:
    """递归加载知识库目录下的文本文件。"""
    docs: List[Document] = []
    if not os.path.isdir(kb_dir):
        return docs
    for root, _dirs, files in os.walk(kb_dir):
        for name in sorted(files):
            if not name.lower().endswith((".md", ".txt")):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:  # pragma: no cover - 文件系统异常兜底
                logger.warning(f"读取知识库文件失败 {path}: {e}")
                continue
            docs.append(
                Document(page_content=text, metadata={"source": os.path.relpath(path, kb_dir)})
            )
    return docs


def split_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """把文档切分为检索片段。"""
    if _HAS_SPLITTER:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )
        return splitter.split_documents(docs)

    # 兜底切分：段落优先 + 定长滑窗
    chunks: List[Document] = []
    step = max(1, chunk_size - chunk_overlap)
    for doc in docs:
        for para in re.split(r"\n\s*\n", doc.page_content):
            para = para.strip()
            if not para:
                continue
            for i in range(0, len(para), step):
                piece = para[i:i + chunk_size].strip()
                if piece:
                    chunks.append(Document(page_content=piece, metadata=dict(doc.metadata)))
    return chunks


# --------------------------------------------------------------------------- #
# Embedding 构建
# --------------------------------------------------------------------------- #
def build_embeddings() -> Optional[Any]:
    """按环境变量构建 Embeddings；**未配置时返回 None**（不报错、不联网）。"""
    model = (os.getenv("EMBEDDING_MODEL") or "").strip()
    if not model:
        return None  # 未配置 → 走关键词检索，无需任何 API

    api_key = (os.getenv("EMBEDDING_API_KEY") or "").strip() or (os.getenv("API_KEY") or "").strip()
    base_url = (os.getenv("EMBEDDING_BASE_URL") or "").strip() or (os.getenv("MODEL_BASE_URL") or "").strip()

    if not api_key:
        logger.warning("已设置 EMBEDDING_MODEL 但缺少 API Key，RAG 将回退为关键词检索")
        return None

    try:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=base_url or None,
            # 非 OpenAI 官方端点（如 DashScope）需关闭 tiktoken 上下文长度校验
            check_embedding_ctx_length=False,
            chunk_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
        )
    except Exception as e:
        logger.warning(f"初始化 Embedding 失败，RAG 回退为关键词检索: {e}")
        return None


# --------------------------------------------------------------------------- #
# 知识库
# --------------------------------------------------------------------------- #
class KnowledgeBase:
    """知识库检索入口，支持向量检索与关键词检索两种模式。"""

    def __init__(
        self,
        kb_dir: Optional[str] = None,
        store_path: Optional[str] = None,
        top_k: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.kb_dir = kb_dir or os.getenv("KNOWLEDGE_DIR", "knowledge")
        self.store_path = store_path or os.getenv("RAG_STORE_PATH", "data/vector_store.json")
        self.top_k = int(top_k or os.getenv("RAG_TOP_K", "3"))
        self.chunk_size = int(chunk_size or os.getenv("RAG_CHUNK_SIZE", "300"))
        self.chunk_overlap = int(chunk_overlap or os.getenv("RAG_CHUNK_OVERLAP", "60"))

        self.chunks: List[Document] = []
        self.embeddings: Optional[Any] = None
        self.matrix: Optional[Any] = None  # numpy.ndarray, shape=(N, D)
        self.mode: str = "keyword"

        self.reload()

    # ---------------- 构建 ----------------
    def reload(self) -> None:
        """重新加载并索引知识库。"""
        self.chunks = split_documents(
            load_documents(self.kb_dir), self.chunk_size, self.chunk_overlap
        )
        if not self.chunks:
            logger.warning(f"知识库为空，检索不可用: {self.kb_dir}（请放入 .md / .txt 文档）")
            self.mode = "keyword"
            return

        self.embeddings = build_embeddings()
        if self.embeddings is None:
            self.mode = "keyword"
            logger.info(f"RAG 运行于【关键词检索】模式，已加载 {len(self.chunks)} 个知识片段")
            return

        try:
            self.matrix = self._build_matrix()
            self.mode = "vector"
            logger.info(
                f"RAG 运行于【向量检索】模式，已索引 {len(self.chunks)} 个片段，维度 {self.matrix.shape[1]}"
            )
        except Exception as e:
            self.mode = "keyword"
            logger.warning(f"构建向量索引失败，回退关键词检索: {e}")

    def _build_matrix(self) -> Any:
        import numpy as np

        model = (os.getenv("EMBEDDING_MODEL") or "").strip()
        hashes = [_hash(c.page_content) for c in self.chunks]

        cache = self._read_cache()
        cached: Dict[str, List[float]] = {}
        if cache and cache.get("model") == model:
            cached = {item["hash"]: item["vector"] for item in cache.get("items", [])}

        missing = [i for i, h in enumerate(hashes) if h not in cached]
        if missing:
            logger.info(f"需向量化的新片段: {len(missing)}/{len(self.chunks)}（其余命中本地缓存）")
            vectors = self.embeddings.embed_documents([self.chunks[i].page_content for i in missing])
            for i, vec in zip(missing, vectors):
                cached[hashes[i]] = vec
            self._write_cache(model, hashes, cached)

        matrix = np.asarray([cached[h] for h in hashes], dtype="float32")
        return _l2_normalize(matrix)

    # ---------------- 检索 ----------------
    def search(self, query: str, k: Optional[int] = None) -> List[Tuple[str, str]]:
        """检索相关片段，返回 ``[(来源, 文本), ...]``，按相关度降序。"""
        k = k or self.top_k
        if not self.chunks:
            return []
        if self.mode == "vector" and self.matrix is not None:
            try:
                return self._vector_search(query, k)
            except Exception as e:
                logger.warning(f"向量检索失败，降级为关键词检索: {e}")
        return self._keyword_search(query, k)

    def _vector_search(self, query: str, k: int) -> List[Tuple[str, str]]:
        import numpy as np

        q = np.asarray(self.embeddings.embed_query(query), dtype="float32").reshape(1, -1)
        scores = self.matrix @ _l2_normalize(q)[0]
        order = np.argsort(-scores)[:k]
        return [
            (self.chunks[i].metadata.get("source", ""), self.chunks[i].page_content) for i in order
        ]

    def _keyword_search(self, query: str, k: int) -> List[Tuple[str, str]]:
        tokens = set(_tokenize(query))
        if not tokens:
            return []
        scored = []
        for doc in self.chunks:
            score = len(tokens & set(_tokenize(doc.page_content)))
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            (doc.metadata.get("source", ""), doc.page_content) for _, doc in scored[:k]
        ]

    # ---------------- 向量缓存 ----------------
    def _read_cache(self) -> Optional[Dict[str, Any]]:
        try:
            if os.path.exists(self.store_path):
                with open(self.store_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"读取向量缓存失败: {e}")
        return None

    def _write_cache(self, model: str, hashes: List[str], vectors: Dict[str, List[float]]) -> None:
        try:
            store_dir = os.path.dirname(self.store_path)
            if store_dir:
                os.makedirs(store_dir, exist_ok=True)
            payload = {
                "model": model,
                "items": [
                    {
                        "hash": h,
                        "source": self.chunks[i].metadata.get("source", ""),
                        "text": self.chunks[i].page_content,
                        "vector": vectors[h],
                    }
                    for i, h in enumerate(hashes)
                ],
            }
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            logger.debug(f"向量缓存已写入: {self.store_path}")
        except Exception as e:
            logger.warning(f"写入向量缓存失败: {e}")

    # ---------------- 便捷属性 ----------------
    @property
    def size(self) -> int:
        """知识片段数量。"""
        return len(self.chunks)

    def stats(self) -> Dict[str, Any]:
        return {"mode": self.mode, "chunks": len(self.chunks), "top_k": self.top_k}


# 全局单例：避免每次工具调用都重建索引
_KB_SINGLETON: Optional[KnowledgeBase] = None


def get_knowledge_base(refresh: bool = False) -> KnowledgeBase:
    """获取（惰性构建的）知识库单例。"""
    global _KB_SINGLETON
    if _KB_SINGLETON is None or refresh:
        _KB_SINGLETON = KnowledgeBase()
    return _KB_SINGLETON
