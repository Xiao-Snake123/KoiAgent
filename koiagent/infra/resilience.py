"""KoiAgent 可靠性工具箱：幂等去重 / 熔断 / 并发限流 / 会话记忆治理。

这些是「能跑」到「能长期跑」之间的差距：

1. **DedupCache**      基于 LRU + TTL 的幂等去重，避免重连或重复推送导致重复回复。
2. **CircuitBreaker**  下游连续失败达阈值即熔断，快速失败而非持续打爆下游。
3. **ConcurrencyLimiter**  asyncio 并发上限，平抑瞬时并发。
4. **ThreadReaper**    Checkpointer 会话记忆治理，按 TTL + LRU 淘汰空闲会话，
   防止长跑进程内存无限增长。

环境变量（由调用方读取）
------------------------
- ``DEDUP_CACHE_SIZE`` / ``DEDUP_TTL``
- ``CIRCUIT_FAILURE_THRESHOLD`` / ``CIRCUIT_RESET_TIMEOUT``
- ``LLM_MAX_CONCURRENCY``
- ``MEMORY_MAX_THREADS`` / ``MEMORY_TTL`` / ``MEMORY_REAP_INTERVAL``
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from loguru import logger


class DedupCache:
    """基于 LRU + TTL 的幂等去重缓存（线程安全）。

    ``seen(key)`` 首次调用返回 ``False`` 并记录；重复调用返回 ``True``。
    """

    def __init__(self, maxsize: int = 2000, ttl: float = 300.0):
        self.maxsize = max(1, int(maxsize))
        self.ttl = float(ttl)
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, float]" = OrderedDict()
        self._hits = 0

    def seen(self, key: str) -> bool:
        """返回 ``True`` 表示该 key 近期已出现过（重复消息）。"""
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            if key in self._data:
                self._data.move_to_end(key)
                self._hits += 1
                return True
            self._data[key] = now
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
            return False

    def _evict_expired(self, now: float) -> None:
        while self._data:
            _key, ts = next(iter(self._data.items()))
            if now - ts > self.ttl:
                self._data.popitem(last=False)
            else:
                break

    def __len__(self) -> int:
        return len(self._data)

    @property
    def hits(self) -> int:
        """累计拦截的重复消息数。"""
        return self._hits


class CircuitBreaker:
    """熔断器。

    - ``closed``    正常放行
    - ``open``      快速失败（不再请求下游）
    - ``half_open`` 冷却结束，放行请求做探测；成功则闭合，失败则重新计时
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = max(1, int(failure_threshold))
        self.reset_timeout = float(reset_timeout)
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at = 0.0
        self._state = "closed"
        self._trips = 0

    def _refresh_locked(self) -> None:
        if self._state == "open" and (time.time() - self._opened_at) >= self.reset_timeout:
            self._state = "half_open"

    @property
    def state(self) -> str:
        with self._lock:
            self._refresh_locked()
            return self._state

    def allow(self) -> bool:
        """是否允许发起下游调用。"""
        with self._lock:
            self._refresh_locked()
            return self._state != "open"

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            if self._state != "closed":
                logger.info("[circuit] 下游恢复，熔断器闭合")
            self._state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                if self._state != "open":
                    self._trips += 1
                    logger.warning(
                        f"[circuit] 连续失败 {self._failures} 次，熔断 {self.reset_timeout:.0f}s"
                    )
                self._state = "open"
                self._opened_at = time.time()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            return {"state": self._state, "failures": self._failures, "trips": self._trips}


class ConcurrencyLimiter:
    """asyncio 并发上限。"""

    def __init__(self, limit: int = 4):
        self.limit = max(1, int(limit))
        self._sem = asyncio.Semaphore(self.limit)
        self._in_flight = 0

    @asynccontextmanager
    async def acquire(self):
        async with self._sem:
            self._in_flight += 1
            try:
                yield
            finally:
                self._in_flight -= 1

    @property
    def in_flight(self) -> int:
        return self._in_flight


class ThreadReaper:
    """Checkpointer 会话记忆治理：TTL + LRU 淘汰空闲 thread。"""

    def __init__(self, max_threads: int = 500, ttl: float = 7200.0, interval: float = 60.0):
        self.max_threads = max(1, int(max_threads))
        self.ttl = float(ttl)
        self.interval = float(interval)
        self._lock = threading.Lock()
        self._seen: "OrderedDict[str, float]" = OrderedDict()
        self._last_reap = 0.0

    def touch(self, thread_id: str) -> None:
        """记录会话活跃时间。"""
        with self._lock:
            self._seen[thread_id] = time.time()
            self._seen.move_to_end(thread_id)

    def due(self) -> bool:
        """是否到达清理检查周期。"""
        with self._lock:
            return (time.time() - self._last_reap) >= self.interval

    def select_victims(self) -> List[str]:
        """选出待清理的 thread：超 TTL 的，以及超出容量后最久未使用的。"""
        now = time.time()
        with self._lock:
            self._last_reap = now
            victims = [tid for tid, ts in self._seen.items() if now - ts > self.ttl]
            for tid in victims:
                self._seen.pop(tid, None)

            overflow = len(self._seen) - self.max_threads
            for _ in range(max(0, overflow)):
                tid, _ts = self._seen.popitem(last=False)
                victims.append(tid)

            return victims

    def forget(self, thread_id: str) -> None:
        with self._lock:
            self._seen.pop(thread_id, None)

    @property
    def size(self) -> int:
        return len(self._seen)


class AsyncRateLimiter:
    """令牌桶限速器（异步）：控制**出站动作**的速率。

    为什么需要它
    ------------
    项目本来只有 ``ConcurrencyLimiter``，但它管的是 **LLM 推理并发**，不是**发消息**。
    当模型响应很快时，Agent 可能在几秒内连续发出多条消息 —— 平台侧有风控，
    高频出站有被限制的风险。

    为什么用令牌桶而不是固定间隔 sleep
    ----------------------------------
    固定间隔（每条都 sleep N 秒）会让**正常场景也被拖慢**。令牌桶允许**突发**：
    桶里攒着 ``burst`` 个令牌时，前几条可以立即发出；用完后再按 ``min_interval``
    匀速补充。这既保住了「平滑出站」，又不牺牲正常响应速度。

    ``min_interval <= 0`` 时退化为**不限速**（直接放行），保持向后兼容。
    """

    def __init__(self, min_interval: float = 0.0, burst: int = 3):
        self.min_interval = float(min_interval)
        self.capacity = max(1, int(burst))
        self._tokens = float(self.capacity)
        self._updated = time.monotonic()
        self._lock = threading.Lock()
        self._waits = 0
        self._total_wait = 0.0

    def _refill(self, now: float) -> None:
        if self.min_interval <= 0:
            self._tokens = float(self.capacity)
            return
        elapsed = max(0.0, now - self._updated)
        self._tokens = min(self.capacity, self._tokens + elapsed / self.min_interval)
        self._updated = now

    @property
    def enabled(self) -> bool:
        return self.min_interval > 0

    def try_acquire(self) -> float:
        """尝试取一个令牌。

        返回还需等待的秒数：``0`` 表示可以立即执行；
        否则调用方应 ``sleep`` 该时长后**重新调用**（因为期间可能有其他人抢走令牌）。
        """
        if not self.enabled:
            return 0.0
        now = time.monotonic()
        with self._lock:
            self._refill(now)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            deficit = 1.0 - self._tokens
            return deficit * self.min_interval

    async def acquire(self) -> None:
        """异步获取一个令牌（必要时等待）。"""
        if not self.enabled:
            return
        waited = 0.0
        while True:
            delay = self.try_acquire()
            if delay <= 0:
                if waited > 0:
                    self._waits += 1
                    self._total_wait += waited
                    logger.debug(f"[rate-limit] 出站限速等待 {waited:.2f}s 后放行")
                return
            # 向上取整到毫秒，避免忙等
            await asyncio.sleep(max(0.001, delay))
            waited += delay

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            self._refill(time.monotonic())
            tokens = self._tokens
        return {
            "enabled": self.enabled,
            "min_interval": self.min_interval,
            "capacity": self.capacity,
            "tokens": round(tokens, 3),
            "throttled": self._waits,
            "total_wait_s": round(self._total_wait, 2),
        }
