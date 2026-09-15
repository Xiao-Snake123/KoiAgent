"""工具记忆：**会话级结果缓存** + 持久化调用记录。

为什么需要它
------------
Agent 在一次对话里可能反复调用同一个工具。最典型的是 RAG 检索：

    买家：续航多久？        → search_knowledge_base("续航")
    买家：那充电要多久？    → search_knowledge_base("充电")
    买家：续航再确认下？    → search_knowledge_base("续航")   ← 完全重复，白花钱

带上会话级缓存后，第三次直接命中，**省一次向量检索/Embedding 调用**。

两个组成部分
------------
1. **内存缓存**（``ToolCache``）：LRU + TTL，只在会话内有效，进程重启即失效。
   这是真正省成本的部分。
2. **调用记录**（``tool_calls`` 表）：持久化每一次调用的工具名、参数、结果摘要、
   是否命中缓存。用于**回放排障**和**统计缓存命中率**。

哪些工具可以缓存 ——「纯函数」判定
---------------------------------
只有**同一会话内、相同参数、结果稳定**的工具才可缓存：

===========================  ==========  ==================================
工具                          可缓存      原因
===========================  ==========  ==================================
``search_knowledge_base``    ✅          知识库在会话内不变
``get_bargain_policy``       ✅          纯函数，只依赖议价轮次
``get_current_time``         ❌          时间每次都在变，缓存会给出错误答案
===========================  ==========  ==================================

> ⚠️ 把 ``get_current_time`` 也缓存是很容易犯的错——它会开始回答「现在 10:00」，
> 而实际上是 11:30。所以缓存是**白名单制**而非黑名单制：新工具默认不缓存，
> 必须显式加入 :data:`CACHEABLE_TOOLS`。

会话隔离
--------
工具函数签名里没有 ``chat_id``（那是 LLM 决定的参数），所以用
:data:`current_chat_id` 这个 ``ContextVar`` 把会话标识传递到工具内部。
LangGraph 的 ``ToolNode`` 与图执行在同一异步上下文，因此 ContextVar 天然可见。

环境变量
--------
- ``TOOL_MEMORY_ENABLED``     是否启用工具记忆，默认 ``true``
- ``TOOL_MEMORY_TTL``         缓存有效期（秒），默认 ``900``
- ``TOOL_MEMORY_MAX_ENTRIES`` 单会话缓存条目上限，默认 ``500``
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from loguru import logger

from koiagent.memory.store import MemoryStore

# 允许缓存的工具白名单（新工具默认不缓存，必须显式加入）
CACHEABLE_TOOLS = {"search_knowledge_base", "get_bargain_policy"}

# 当前会话标识。由 KoiReplyBot.agenerate_reply 设置，工具函数内部读取。
current_chat_id: ContextVar[Optional[str]] = ContextVar("koi_current_chat_id", default=None)


@contextmanager
def bind_chat_id(chat_id: Optional[str]) -> Iterator[None]:
    """在上下文内绑定当前会话标识（工具记忆据此做会话隔离）。"""
    token = current_chat_id.set(chat_id)
    try:
        yield
    finally:
        current_chat_id.reset(token)


def get_current_chat_id() -> Optional[str]:
    return current_chat_id.get()


def cache_key(chat_id: Optional[str], tool_name: str, args: Dict[str, Any]) -> str:
    """缓存键：会话 + 工具名 + **规范化参数**。

    参数按键排序序列化，保证 ``{"a":1,"b":2}`` 与 ``{"b":2,"a":1}`` 命中同一条。
    """
    try:
        payload = json.dumps(args or {}, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        payload = str(args)
    return f"{chat_id or '-'}|{tool_name}|{payload}"


class ToolCache:
    """LRU + TTL 的会话级缓存（线程安全）。"""

    def __init__(self, ttl: float = 900.0, max_entries: int = 500):
        self.ttl = float(ttl)
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[str]:
        now = time.time()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            ts, value = entry
            if now - ts > self.ttl:
                # 过期：删除并视为未命中（不能返回陈旧结果）
                self._data.pop(key, None)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def invalidate(self, prefix: Optional[str] = None) -> int:
        """清除缓存：``prefix`` 为空时清空全部，否则清除匹配前缀的条目。"""
        with self._lock:
            if not prefix:
                count = len(self._data)
                self._data.clear()
                return count
            keys = [key for key in self._data if key.startswith(prefix)]
            for key in keys:
                self._data.pop(key, None)
            return len(keys)

    def clear(self) -> None:
        self.invalidate()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            size = len(self._data)
        total = self.hits + self.misses
        return {
            "entries": size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


class ToolMemory:
    """工具记忆门面：缓存 + 调用记录。"""

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        ttl: Optional[float] = None,
        max_entries: Optional[int] = None,
        enabled: Optional[bool] = None,
        persist: bool = True,
    ):
        self.store = store
        self.persist = persist
        if enabled is None:
            enabled = os.getenv("TOOL_MEMORY_ENABLED", "true").strip().lower() != "false"
        self.enabled = bool(enabled)
        self.cache = ToolCache(
            ttl=float(ttl if ttl is not None else os.getenv("TOOL_MEMORY_TTL", "900")),
            max_entries=int(
                max_entries
                if max_entries is not None
                else os.getenv("TOOL_MEMORY_MAX_ENTRIES", "500")
            ),
        )
        self._persist_failures = 0

    # ------------------------------------------------------------------ #
    # 缓存
    # ------------------------------------------------------------------ #
    def is_cacheable(self, tool_name: str) -> bool:
        return self.enabled and tool_name in CACHEABLE_TOOLS

    def lookup(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """查询缓存；未命中或不可缓存时返回 ``None``。"""
        if not self.is_cacheable(tool_name):
            return None
        return self.cache.get(cache_key(get_current_chat_id(), tool_name, args))

    def store_result(self, tool_name: str, args: Dict[str, Any], result: str) -> None:
        if self.is_cacheable(tool_name) and isinstance(result, str):
            self.cache.put(cache_key(get_current_chat_id(), tool_name, args), result)

    def invoke(
        self, tool_name: str, args: Dict[str, Any], compute: Callable[[], str]
    ) -> str:
        """``get-or-compute`` 组合子：命中缓存直接返回，否则计算并写入。

        **无论是否命中都会落一条调用记录**，这样才能统计真实的缓存命中率。
        """
        cached = self.lookup(tool_name, args)
        if cached is not None:
            logger.debug(f"[tool-memory] 命中缓存 {tool_name}({args})")
            self.record(tool_name, args, cached, cache_hit=True)
            return cached

        result = compute()
        self.store_result(tool_name, args, result)
        self.record(tool_name, args, result, cache_hit=False)
        return result

    # ------------------------------------------------------------------ #
    # 记录
    # ------------------------------------------------------------------ #
    def record(
        self, tool_name: str, args: Dict[str, Any], result: str, cache_hit: bool = False
    ) -> None:
        """持久化一条调用记录（失败不影响主流程）。"""
        if not self.persist or self.store is None:
            return
        try:
            self.store.record_tool_call(
                chat_id=get_current_chat_id(),
                tool_name=tool_name,
                args=args,
                result=result,
                cache_hit=cache_hit,
            )
        except Exception as e:
            # 记录失败是纯粹的次要问题，但连续失败说明存储层有问题，需要提示
            self._persist_failures += 1
            if self._persist_failures <= 3:
                logger.warning(f"[tool-memory] 调用记录写入失败: {e}")

    # ------------------------------------------------------------------ #
    # 维护
    # ------------------------------------------------------------------ #
    def invalidate_session(self, chat_id: Optional[str] = None) -> int:
        """清除某个会话的缓存（例如议价轮次变化后需要重查策略）。"""
        target = chat_id or get_current_chat_id()
        return self.cache.invalidate(prefix=f"{target or '-'}|")

    def invalidate_tool(self, tool_name: str) -> int:
        """清除某个工具在**所有会话**下的缓存（例如知识库热更新后）。"""
        marker = f"|{tool_name}|"
        with self.cache._lock:  # noqa: SLF001 - 同模块内访问，避免再暴露一层 API
            keys: List[str] = [key for key in self.cache._data if marker in key]
            for key in keys:
                self.cache._data.pop(key, None)
        return len(keys)

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cacheable": sorted(CACHEABLE_TOOLS),
            **self.cache.stats(),
        }


_TOOL_MEMORY_SINGLETON: Optional[ToolMemory] = None
_SINGLETON_LOCK = threading.Lock()


def get_tool_memory(store: Optional[MemoryStore] = None) -> ToolMemory:
    """获取（惰性构建的）工具记忆单例。"""
    global _TOOL_MEMORY_SINGLETON
    with _SINGLETON_LOCK:
        if _TOOL_MEMORY_SINGLETON is None:
            _TOOL_MEMORY_SINGLETON = ToolMemory(store=store)
        return _TOOL_MEMORY_SINGLETON
