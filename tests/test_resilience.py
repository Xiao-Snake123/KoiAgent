"""可靠性组件测试：幂等去重 / 熔断 / 并发限流 / 会话记忆治理（全程离线）。"""
import asyncio
import time

from koiagent.infra.resilience import CircuitBreaker, ConcurrencyLimiter, DedupCache, ThreadReaper


# --------------------------------------------------------------------------- #
# DedupCache
# --------------------------------------------------------------------------- #
def test_dedup_detects_repeats():
    cache = DedupCache(maxsize=10, ttl=60)
    assert cache.seen("k1") is False
    assert cache.seen("k1") is True
    assert cache.seen("k2") is False
    assert cache.hits == 1


def test_dedup_expires_by_ttl():
    cache = DedupCache(maxsize=10, ttl=0.01)
    assert cache.seen("k") is False
    time.sleep(0.03)
    assert cache.seen("k") is False  # TTL 过期后视为新消息


def test_dedup_evicts_oldest_on_overflow():
    cache = DedupCache(maxsize=2, ttl=60)
    cache.seen("a")
    cache.seen("b")
    cache.seen("c")
    assert len(cache) == 2
    assert cache.seen("a") is False  # 最旧的已被淘汰


# --------------------------------------------------------------------------- #
# CircuitBreaker
# --------------------------------------------------------------------------- #
def test_circuit_stays_closed_on_success():
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout=60)
    breaker.record_success()
    assert breaker.allow()
    assert breaker.state == "closed"


def test_circuit_opens_after_threshold():
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout=60)
    breaker.record_failure()
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state == "open"
    assert not breaker.allow()


def test_circuit_half_open_then_close():
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
    breaker.record_failure()
    assert breaker.state == "open"

    time.sleep(0.03)
    assert breaker.state == "half_open"
    assert breaker.allow()          # 半开放行探测
    breaker.record_success()
    assert breaker.state == "closed"


def test_circuit_reopens_when_half_open_fails():
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
    breaker.record_failure()
    time.sleep(0.03)
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state == "open"


def test_circuit_snapshot_reports_trips():
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=60)
    breaker.record_failure()
    snapshot = breaker.snapshot()
    assert snapshot["state"] == "open"
    assert snapshot["trips"] == 1


# --------------------------------------------------------------------------- #
# ConcurrencyLimiter
# --------------------------------------------------------------------------- #
def test_concurrency_limiter_caps_parallelism():
    async def run():
        limiter = ConcurrencyLimiter(limit=2)
        peak = 0
        current = 0

        async def worker():
            nonlocal peak, current
            async with limiter.acquire():
                current += 1
                peak = max(peak, current)
                await asyncio.sleep(0.01)
                current -= 1

        await asyncio.gather(*(worker() for _ in range(8)))
        return peak

    assert asyncio.run(run()) <= 2


# --------------------------------------------------------------------------- #
# ThreadReaper
# --------------------------------------------------------------------------- #
def test_reaper_evicts_by_ttl():
    reaper = ThreadReaper(max_threads=10, ttl=0.01, interval=0)
    reaper.touch("t1")
    time.sleep(0.03)

    victims = reaper.select_victims()
    assert "t1" in victims
    assert reaper.size == 0


def test_reaper_evicts_lru_on_overflow():
    reaper = ThreadReaper(max_threads=2, ttl=3600, interval=0)
    for thread_id in ("a", "b", "c"):
        reaper.touch(thread_id)

    victims = reaper.select_victims()
    assert victims == ["a"]          # 最久未使用
    assert reaper.size == 2


def test_reaper_due_respects_interval():
    reaper = ThreadReaper(interval=3600)
    assert reaper.due() is True      # 首次调用触发一次清理检查
    reaper.select_victims()          # 更新 last_reap
    assert reaper.due() is False     # 未到下一个周期


def test_reaper_forget_removes_entry():
    reaper = ThreadReaper(max_threads=10, ttl=3600, interval=0)
    reaper.touch("x")
    reaper.forget("x")
    assert reaper.size == 0
