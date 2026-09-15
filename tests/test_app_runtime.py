"""运行时组件测试：订单事件钩子、出站限速器，以及 KoiLive 的纯逻辑方法。

这一层原先完全没有测试覆盖 —— 而 ``KoiLive`` 恰恰是分支最多、最容易出 bug 的地方
（消息分流、人工接管、时效过滤、去重、商品描述构建）。此文件先把**不需要网络**的
部分锁定住。
"""
import asyncio
import json
import os
import time

import pytest

from koiagent.infra.resilience import AsyncRateLimiter
from koiagent.memory.tool_memory import ToolMemory
from koiagent.ops.order_events import (
    OrderEvent,
    OrderEventHandler,
    OrderEventKind,
    resolve_order_event,
)
from koiagent.storage.context import ChatContextManager


# =========================================================================== #
# 订单事件
# =========================================================================== #
def test_resolve_order_event_maps_known_reminders():
    event = resolve_order_event("等待买家付款", user_id="u1", chat_id="c1")
    assert event is not None
    assert event.kind is OrderEventKind.AWAITING_PAYMENT
    assert event.user_url.endswith("userId=u1")


def test_resolve_order_event_ignores_non_order_text():
    assert resolve_order_event("你好，在吗") is None
    assert resolve_order_event("") is None


def test_resolve_order_event_is_whitespace_tolerant():
    assert resolve_order_event("  交易关闭  ").kind is OrderEventKind.CLOSED


def test_order_event_labels_cover_all_kinds():
    for kind in OrderEventKind:
        assert kind.label


class RecordingBot:
    """记录写进 Agent 记忆的内容。"""

    def __init__(self):
        self.messages = []

    def record_message(self, chat_id, role, content):
        self.messages.append((chat_id, role, content))


class FakeMemory:
    def __init__(self):
        self.invalidated = []
        self.deals = []

    def invalidate_session(self, chat_id):
        self.invalidated.append(chat_id)
        return 2

    class _Profiles:
        def __init__(self, outer):
            self.outer = outer

        def bump_deal(self, user_id):
            self.outer.deals.append(user_id)

    @property
    def profiles(self):
        return FakeMemory._Profiles(self)


@pytest.fixture
def handler():
    bot, memory = RecordingBot(), FakeMemory()
    return OrderEventHandler(
        context_manager=ChatContextManager(db_path=":memory:"), bot=bot, memory=memory
    )


def test_awaiting_payment_records_memory(handler):
    event = OrderEvent(kind=OrderEventKind.AWAITING_PAYMENT, user_id="u1", chat_id="c1")
    assert handler.handle(event) is True
    assert event.handled is True
    assert handler.bot.messages and "等待付款" in handler.bot.messages[0][2]


def test_closed_invalidates_stale_tool_cache(handler):
    """交易关闭后价格/库存可能已变，必须让工具缓存失效。"""
    event = OrderEvent(kind=OrderEventKind.CLOSED, user_id="u1", chat_id="c1")
    assert handler.handle(event) is True
    assert handler.memory.invalidated == ["c1"]
    assert event.notes["cache_invalidated"] == 2


def test_awaiting_shipment_counts_a_deal(handler):
    event = OrderEvent(kind=OrderEventKind.AWAITING_SHIPMENT, user_id="u1", chat_id="c1")
    assert handler.handle(event) is True
    assert handler.memory.deals == ["u1"]
    assert event.notes["deal_recorded"] is True


def test_handler_swallows_exceptions(handler, monkeypatch):
    """单个事件处理失败不能打断消息主循环。"""

    def boom(_event):
        raise RuntimeError("业务动作炸了")

    monkeypatch.setattr(handler, "on_closed", boom)
    event = OrderEvent(kind=OrderEventKind.CLOSED, user_id="u1", chat_id="c1")
    assert handler.handle(event) is False


def test_handler_survives_missing_dependencies():
    """没注入 bot / memory 时不应抛错（默认行为要能独立工作）。"""
    bare = OrderEventHandler()
    assert bare.handle(OrderEvent(kind=OrderEventKind.AWAITING_PAYMENT)) is True


# =========================================================================== #
# 出站限速器
# =========================================================================== #
def test_rate_limiter_disabled_passes_through():
    limiter = AsyncRateLimiter(min_interval=0, burst=1)
    assert limiter.enabled is False
    assert limiter.try_acquire() == 0.0
    asyncio.run(limiter.acquire())  # 不应阻塞


def test_rate_limiter_allows_burst():
    limiter = AsyncRateLimiter(min_interval=10, burst=3)
    assert limiter.try_acquire() == 0.0
    assert limiter.try_acquire() == 0.0
    assert limiter.try_acquire() == 0.0
    # 桶空后应返回建议等待时长
    assert limiter.try_acquire() > 0


def test_rate_limiter_reports_wait_and_refills():
    limiter = AsyncRateLimiter(min_interval=0.05, burst=1)
    assert limiter.try_acquire() == 0.0
    assert limiter.try_acquire() > 0
    time.sleep(0.08)
    assert limiter.try_acquire() == 0.0  # 令牌已补充


def test_rate_limiter_acquire_actually_waits():
    limiter = AsyncRateLimiter(min_interval=0.05, burst=1)
    asyncio.run(limiter.acquire())

    async def measure():
        started = time.perf_counter()
        await limiter.acquire()
        return time.perf_counter() - started

    elapsed = asyncio.run(measure())
    assert elapsed >= 0.03


def test_rate_limiter_snapshot_shape():
    limiter = AsyncRateLimiter(min_interval=1, burst=2)
    limiter.try_acquire()
    snap = limiter.snapshot()
    assert snap["enabled"] is True
    assert snap["capacity"] == 2
    assert "throttled" in snap and "tokens" in snap


# =========================================================================== #
# KoiLive 纯逻辑
# =========================================================================== #
class FakeBot:
    """KoiLive 测试替身：只提供被测试方法用到的接口。"""

    def __init__(self):
        self.last_intent = None
        self.tracer = type("T", (), {"format_report": staticmethod(lambda: "report")})()
        self.recorded = []

    def record_message(self, chat_id, role, content):
        self.recorded.append((chat_id, role, content))

    async def agenerate_reply(self, *args, **kwargs):
        raise AssertionError("本测试不应真的调用模型")


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALTH_FILE", str(tmp_path / "health.json"))
    from koiagent.app import KoiLive

    return KoiLive(
        "unb=123456; _m_h5_tk=abc",
        FakeBot(),
        context_manager=ChatContextManager(db_path=str(tmp_path / "chat.db")),
    )


def test_live_parses_cookies_and_device_id(live):
    assert live.myid == "123456"
    assert live.device_id
    assert live.memory is not None


def test_live_format_price_converts_cents(live):
    assert live.format_price(1990) == 19.9
    assert live.format_price(10000) == 100.0
    assert live.format_price(None) == 0.0          # 脏数据兜底
    assert live.format_price("bad") == 0.0


def test_live_build_item_description_summarizes_skus(live):
    item = {
        "title": "降噪耳机",
        "desc": "主动降噪",
        "quantity": 5,
        "skuList": [
            {"propertyList": [{"valueText": "黑色"}], "price": 19900, "quantity": 3},
            {"propertyList": [{"valueText": "白色"}], "price": 21900, "quantity": 2},
        ],
    }
    payload = json.loads(live.build_item_description(item))
    assert payload["title"] == "降噪耳机"
    assert payload["price_range"] == "¥199.0 - ¥219.0"
    assert len(payload["sku_details"]) == 2


def test_live_build_item_description_falls_back_to_main_price(live):
    payload = json.loads(live.build_item_description({"title": "T恤", "soldPrice": 59.0}))
    assert payload["price_range"] == "¥59.0"


def test_live_toggle_keyword_and_manual_mode(live):
    live.toggle_keywords = "。"
    assert live.check_toggle_keywords(" 。 ") is True
    assert live.check_toggle_keywords("你好。") is False

    assert live.is_manual_mode("c1") is False
    assert live.toggle_manual_mode("c1") == "manual"
    assert live.is_manual_mode("c1") is True
    assert live.toggle_manual_mode("c1") == "auto"
    assert live.is_manual_mode("c1") is False


def test_live_manual_mode_auto_expires(live, monkeypatch):
    live.enter_manual_mode("c1")
    live.manual_mode_timeout = 0  # 立刻超时
    assert live.is_manual_mode("c1") is False
    assert "c1" not in live.manual_mode_conversations


def test_live_detects_bracket_system_message(live):
    assert live.is_bracket_system_message("[系统提示]") is True
    assert live.is_bracket_system_message("【中文括号】") is False
    assert live.is_bracket_system_message("普通消息") is False
    assert live.is_bracket_system_message(None) is False


def test_live_message_type_guards_are_defensive(live):
    """这些判断面对畸形消息必须返回 False 而不是抛异常。"""
    assert live.is_chat_message(None) is False
    assert live.is_chat_message({}) is False
    assert live.is_sync_package({}) is False
    assert live.is_typing_status({}) is False
    assert live.is_system_message({}) is False

    good = {"1": {"10": {"reminderContent": "hi"}}}
    assert live.is_chat_message(good) is True


def test_live_dedup_blocks_repeated_message(live):
    key = "c1|123|u1|你好"
    assert live.dedup.seen(key) is False
    assert live.dedup.seen(key) is True
    assert live.dedup.hits == 1


def test_live_touch_health_writes_fresh_file(live, tmp_path):
    live._touch_health("connected")
    payload = json.loads(open(live.health_path, encoding="utf-8").read())
    assert payload["status"] == "connected"
    assert time.time() - payload["ts"] < 5


def test_live_send_limiter_defaults_disabled(live):
    assert live.send_limiter.enabled is False


def test_live_order_handler_wired_to_context_manager(live):
    assert live.order_handler.context_manager is live.context_manager
    assert live.order_handler.memory is live.memory
