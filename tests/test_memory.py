"""记忆系统测试：存储 / 短期 / 长期 / 画像 / 工具记忆 / 门面。

测试全部**离线**运行（用假 LLM），且每个用例使用临时数据库，互不干扰。
"""
import asyncio
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from koiagent.memory import (
    CACHEABLE_TOOLS,
    ExtractedMemory,
    LongTermMemory,
    MemoryManager,
    MemoryStore,
    ProfilePatch,
    ProfileStore,
    ShortTermMemory,
    ToolMemory,
    UserProfile,
    bind_chat_id,
)
from koiagent.memory.long_term import fingerprint_of
from koiagent.memory.manager import TurnInsight
from koiagent.memory.tool_memory import cache_key, get_current_chat_id


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #
class FakeLLM:
    """返回固定结果的假模型（不联网）。"""

    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.response


@pytest.fixture
def store(tmp_path):
    return MemoryStore(str(tmp_path / "memory.db"))


def make_manager(store, summarizer=None, extractor=None):
    manager = MemoryManager(
        store=store,
        short_term=ShortTermMemory(max_messages=4, trigger=2),
        tool_memory=ToolMemory(store=store, persist=True),
    )
    manager.configure_llm(summarizer=summarizer, extractor=extractor)
    return manager


# =========================================================================== #
# 存储层
# =========================================================================== #
def test_store_upsert_and_dedup(store):
    assert store.upsert_memory("user", "u1", "fp1", "买家是做摄影的", "fact", 1.0) is True
    # 同一指纹再次写入不算新增，但权重会提升（重复提及说明更重要）
    assert store.upsert_memory("user", "u1", "fp1", "买家是做摄影的", "fact", 1.0) is False

    records = store.list_memories("user", "u1")
    assert len(records) == 1
    assert records[0].weight == pytest.approx(1.5)


def test_store_isolation_between_scopes(store):
    store.upsert_memory("user", "u1", "fp1", "A", "fact", 1.0)
    store.upsert_memory("user", "u2", "fp1", "B", "fact", 1.0)
    store.upsert_memory("chat", "u1", "fp1", "C", "fact", 1.0)
    assert store.count_memories() == 3
    assert [r.content for r in store.list_memories("user", "u1")] == ["A"]
    assert [r.content for r in store.list_memories("chat", "u1")] == ["C"]


def test_store_touch_and_prune(store):
    store.upsert_memory("user", "u1", "fp1", "事实一", "fact", 1.0)
    store.touch_memories(["fp1"], "user", "u1")
    assert store.list_memories("user", "u1")[0].hits == 1

    # 容量淘汰：只保留权重最高的若干条
    for i in range(6):
        store.upsert_memory("user", "u1", f"fp{i}", f"记忆{i}", "fact", 0.5 + i * 0.1)
    removed = store.prune_memories("user", "u1", max_facts=3, decay_days=0)
    assert removed >= 3
    assert store.count_memories("scope") == 0  # 不存在的 scope 计数为 0
    assert len(store.list_memories("user", "u1")) <= 3


def test_store_decay_prune_only_removes_unused_low_value(store):
    store.upsert_memory("user", "u1", "old", "旧的低价值记忆", "fact", 1.0)
    store.upsert_memory("user", "u1", "hot", "重要的记忆", "commitment", 2.0)
    store.touch_memories(["hot"], "user", "u1")
    # 让 created_at 变成很久以前
    store._execute(
        "UPDATE memories SET created_at = ? WHERE scope_id = 'u1'", (time.time() - 100 * 86400,),
        write=True,
    )
    store.prune_memories("user", "u1", max_facts=100, decay_days=30)
    remaining = {r.fingerprint for r in store.list_memories("user", "u1")}
    assert remaining == {"hot"}  # 被召回过的保留，从未命中的低价值项被清掉


def test_store_profile_roundtrip(store):
    store.save_profile("u1", {"user_id": "u1", "budget_max": 500})
    assert store.load_profile("u1")["budget_max"] == 500
    assert store.load_profile("nobody") is None
    assert store.count_profiles() == 1


def test_store_tool_call_stats(store):
    store.record_tool_call("c1", "search_knowledge_base", {"query": "续航"}, "结果", cache_hit=False)
    store.record_tool_call("c1", "search_knowledge_base", {"query": "续航"}, "结果", cache_hit=True)
    stats = store.tool_call_stats()
    assert stats["total"] == 2
    assert stats["cache_hits"] == 1
    assert stats["by_tool"]["search_knowledge_base"] == 2


def test_store_survives_broken_sql(store):
    """存储层任何异常都必须被吞掉并返回空结果，不能把异常抛给主流程。"""
    assert store._execute("SELECT * FROM not_a_table", (), write=False) == []


# =========================================================================== #
# 长期记忆
# =========================================================================== #
def test_fingerprint_normalizes_punctuation_and_case():
    assert fingerprint_of("已答应包邮。") == fingerprint_of("已答应包邮")
    assert fingerprint_of("Budget 500") == fingerprint_of("budget500")
    assert fingerprint_of("A") != fingerprint_of("B")


def test_long_term_remember_dedups_within_batch(store):
    memory = LongTermMemory(store=store)
    items = [
        ExtractedMemory(kind="fact", content="买家是做摄影的", weight=1.0),
        ExtractedMemory(kind="fact", content="买家是做摄影的", weight=1.0),  # 同批重复
    ]
    assert memory.remember("user", "u1", items) == 1


def test_long_term_recall_ranks_by_relevance(store):
    memory = LongTermMemory(store=store, top_k=5, min_score=2)
    memory.remember(
        "user",
        "u1",
        [
            ExtractedMemory(kind="fact", content="买家关注续航和降噪", weight=1.0),
            ExtractedMemory(kind="preference", content="买家喜欢红色", weight=1.0),
        ],
    )
    hits = memory.recall("user", "u1", query="续航怎么样")
    assert hits, "相关记忆应被召回"
    assert "续航" in hits[0].content


def test_long_term_recall_updates_hit_counters(store):
    memory = LongTermMemory(store=store, min_score=1)
    memory.remember("user", "u1", [ExtractedMemory(kind="fact", content="买家在北京", weight=1.0)])
    memory.recall("user", "u1", query="北京")
    assert store.list_memories("user", "u1")[0].hits == 1


def test_long_term_recall_filters_irrelevant(store):
    memory = LongTermMemory(store=store, min_score=5)
    memory.remember("user", "u1", [ExtractedMemory(kind="fact", content="买家在北京", weight=1.0)])
    # 完全不相关且阈值高 → 不应该把无关记忆塞进上下文
    assert memory.recall("user", "u1", query="电池容量多少毫安时") == []


def test_long_term_extract_handles_failure(store):
    memory = LongTermMemory(store=store)
    llm = FakeLLM(exc=RuntimeError("boom"))
    assert asyncio.run(memory.extract(llm, "买家问了续航", "答：20 小时")) == []


def test_long_term_extract_filters_short_content(store):
    memory = LongTermMemory(store=store)
    llm = FakeLLM(response=type("R", (), {"memories": [ExtractedMemory(kind="fact", content="好")]})())
    # 内容过短（<3 字）应被丢弃
    assert asyncio.run(memory.extract(llm, "在吗", "在的")) == []


def test_long_term_disabled_is_noop(store):
    memory = LongTermMemory(store=store, enabled=False)
    assert memory.remember("user", "u1", [ExtractedMemory(kind="fact", content="买家在北京")]) == 0
    assert memory.recall("user", "u1", "北京") == []


# =========================================================================== #
# 用户画像
# =========================================================================== #
def test_profile_merge_scalar_overrides_and_list_unions():
    profile = UserProfile(user_id="u1")
    profile.merge(ProfilePatch(intent_level="browse", interests=["续航"], budget_max=800))
    profile.merge(ProfilePatch(intent_level="ready", interests=["降噪", "续航"], budget_max=500))

    assert profile.intent_level == "ready"          # 标量：新值覆盖
    assert profile.budget_max == 500                # 标量：新值覆盖
    assert profile.interests == ["续航", "降噪"]     # 列表：取并集且去重


def test_profile_ignores_unknown_and_empty_values():
    profile = UserProfile(user_id="u1")
    changed = profile.merge(ProfilePatch(intent_level="unknown", bargain_style=None))
    assert changed == []
    assert profile.intent_level == "unknown"


def test_profile_drops_inconsistent_budget_range():
    """出现「下限 > 上限」时丢弃下限（上限更具约束力）。"""
    profile = UserProfile(user_id="u1")
    profile.merge(ProfilePatch(budget_min=900, budget_max=500))
    assert profile.budget_min is None
    assert profile.budget_max == 500


def test_profile_render_and_is_empty():
    profile = UserProfile(user_id="u1")
    assert profile.is_empty()
    assert profile.render() == ""

    profile.merge(
        ProfilePatch(
            intent_level="ready",
            bargain_style="direct",
            interests=["续航"],
            tags=["价格敏感"],
        )
    )
    profile.touch()
    profile.record_bargain(2)
    text = profile.render()
    assert "强意向" in text and "直接砍价" in text and "续航" in text and "议价 2 次" in text


def test_profile_store_roundtrip_and_counters(store):
    store_obj = ProfileStore(store=store)
    store_obj.merge("u1", ProfilePatch(intent_level="compare"))
    store_obj.touch("u1")
    store_obj.bump_bargain("u1", 3)
    store_obj.bump_deal("u1")

    profile = store_obj.get("u1")
    assert profile.intent_level == "compare"
    assert profile.interaction_count == 1
    assert profile.total_bargain_count == 3
    assert profile.deals_closed == 1


def test_profile_from_dict_ignores_unknown_fields():
    profile = UserProfile.from_dict({"user_id": "u1", "legacy_field": 1, "interests": "not-a-list"})
    assert profile.user_id == "u1"
    assert profile.interests == []


# =========================================================================== #
# 短期记忆
# =========================================================================== #
def test_short_term_boundary_and_pending_range():
    memory = ShortTermMemory(max_messages=4, trigger=2)
    assert memory.boundary(3) == 0            # 未超窗口
    assert memory.boundary(10) == 6           # 前 6 条应被摘要覆盖
    assert memory.pending_range(10, 0) == (0, 6)
    assert memory.pending_range(10, 6) == (0, 0)   # 已摘要到边界
    assert memory.should_summarize(10, 0) is True
    assert memory.should_summarize(10, 5) is False  # 只差 1 条，不划算


def test_short_term_window_keeps_latest():
    memory = ShortTermMemory(max_messages=2)
    messages = [HumanMessage(content=f"m{i}") for i in range(5)]
    assert [m.content for m in memory.window(messages)] == ["m3", "m4"]


def test_short_term_summarize_uses_llm_and_keeps_previous_on_failure():
    memory = ShortTermMemory(max_messages=2, trigger=1)
    messages = [HumanMessage(content="买家问续航"), AIMessage(content="答：20 小时")]

    ok = FakeLLM(response=AIMessage(content="买家关心续航，已答 20 小时。"))
    assert asyncio.run(memory.summarize(ok, "", messages)) == "买家关心续航，已答 20 小时。"

    bad = FakeLLM(exc=RuntimeError("boom"))
    assert asyncio.run(memory.summarize(bad, "旧摘要", messages)) == "旧摘要"


def test_short_term_build_context_includes_summary_and_window():
    memory = ShortTermMemory(max_messages=1)
    text = memory.build_context("早前聊过价格", [HumanMessage(content="最新一句")])
    assert "较早对话的摘要" in text and "最新一句" in text


# =========================================================================== #
# 工具记忆
# =========================================================================== #
def test_tool_memory_cache_hit_and_miss(store):
    memory = ToolMemory(store=store, ttl=60, persist=True)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return "检索结果"

    assert memory.invoke("search_knowledge_base", {"query": "续航"}, compute) == "检索结果"
    assert memory.invoke("search_knowledge_base", {"query": "续航"}, compute) == "检索结果"
    assert calls["n"] == 1, "第二次应命中缓存，不应再次计算"
    assert memory.cache.hits == 1


def test_tool_memory_normalizes_argument_order():
    """参数按键排序序列化，避免 {"a":1,"b":2} 与 {"b":2,"a":1} 被当成两个键。"""
    assert cache_key("c", "t", {"a": 1, "b": 2}) == cache_key("c", "t", {"b": 2, "a": 1})


def test_tool_memory_session_isolation(store):
    memory = ToolMemory(store=store, ttl=60, persist=False)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return f"结果{calls['n']}"

    with bind_chat_id("chat-A"):
        assert memory.invoke("search_knowledge_base", {"query": "x"}, compute) == "结果1"
    with bind_chat_id("chat-B"):
        # 不同会话不应共享缓存
        assert memory.invoke("search_knowledge_base", {"query": "x"}, compute) == "结果2"
    assert calls["n"] == 2


def test_tool_memory_never_caches_time_tool(store):
    memory = ToolMemory(store=store, ttl=60, persist=False)
    assert "get_current_time" not in CACHEABLE_TOOLS
    assert memory.is_cacheable("get_current_time") is False
    # 即使显式写入也不应被读出来
    memory.store_result("get_current_time", {}, "2026-01-01 00:00:00")
    assert memory.lookup("get_current_time", {}) is None


def test_tool_memory_ttl_expiry(store):
    memory = ToolMemory(store=store, ttl=0.01, persist=False)
    memory.store_result("search_knowledge_base", {"query": "x"}, "旧结果")
    time.sleep(0.05)
    assert memory.lookup("search_knowledge_base", {"query": "x"}) is None


def test_tool_memory_records_every_call(store):
    memory = ToolMemory(store=store, ttl=60, persist=True)
    with bind_chat_id("c1"):
        memory.invoke("search_knowledge_base", {"query": "x"}, lambda: "结果")
        memory.invoke("search_knowledge_base", {"query": "x"}, lambda: "结果")
    stats = store.tool_call_stats()
    assert stats["total"] == 2 and stats["cache_hits"] == 1


def test_tool_memory_invalidate(store):
    memory = ToolMemory(store=store, ttl=60, persist=False)
    memory.store_result("search_knowledge_base", {"query": "x"}, "结果")
    assert memory.invalidate_tool("search_knowledge_base") == 1
    assert memory.lookup("search_knowledge_base", {"query": "x"}) is None


def test_tool_memory_context_default_is_none():
    assert get_current_chat_id() is None


# =========================================================================== #
# 门面：MemoryManager
# =========================================================================== #
def test_manager_recall_combines_profile_and_facts(store):
    manager = make_manager(store)
    manager.profiles.merge("u1", ProfilePatch(intent_level="ready", interests=["续航"]))
    manager.long_term.remember(
        "user", "u1", [ExtractedMemory(kind="fact", content="买家在北京关注续航", weight=1.0)]
    )

    context = manager.recall(chat_id="c1", user_id="u1", query="续航")
    assert context.has_profile
    assert len(context.facts) == 1
    rendered = context.render()
    assert "买家画像" in rendered and "买家在北京关注续航" in rendered
    assert context.to_trace() == {"memory_facts": 1, "profile_hit": True}


def test_manager_recall_empty_when_disabled(store):
    manager = MemoryManager(store=store, enabled=False, tool_memory=ToolMemory(store=store))
    assert manager.recall(chat_id="c1", user_id="u1", query="x").is_empty()


def test_manager_maybe_summarize_only_when_triggered(store):
    summarizer = FakeLLM(response=AIMessage(content="压缩后的摘要"))
    manager = make_manager(store, summarizer=summarizer)

    messages = [HumanMessage(content=f"m{i}") for i in range(3)]
    assert asyncio.run(manager.maybe_summarize(messages, "", 0)) == {}
    assert summarizer.calls == 0

    messages = [HumanMessage(content=f"m{i}") for i in range(10)]
    updates = asyncio.run(manager.maybe_summarize(messages, "", 0))
    assert updates["summary"] == "压缩后的摘要"
    assert updates["summarized_count"] == 6  # 10 - max_messages(4)
    assert summarizer.calls == 1


def test_manager_remember_writes_memory_and_profile(store):
    insight = TurnInsight(
        memories=[ExtractedMemory(kind="preference", content="买家偏好安静的环境", weight=1.5)],
        profile=ProfilePatch(intent_level="compare", interests=["降噪"]),
    )
    manager = make_manager(store, extractor=FakeLLM(response=insight))

    result = asyncio.run(
        manager.remember("c1", "u1", "有没有安静一点的型号推荐", "有的，这款降噪很好", "--")
    )
    assert result["added"] == 1
    assert "interests" in result["profile_fields"]
    assert store.count_memories() == 1
    assert manager.profiles.get("u1").intent_level == "compare"
    assert manager.profiles.get("u1").interaction_count == 1


def test_manager_remember_skips_short_input(store):
    extractor = FakeLLM(response=TurnInsight())
    manager = make_manager(store, extractor=extractor)

    result = asyncio.run(manager.remember("c1", "u1", "好的", "不客气"))
    assert result["skipped"] == "input_too_short"
    assert extractor.calls == 0, "寒暄不应触发一次抽取调用（这是成本优化点）"
    # 但接待计数仍要更新
    assert manager.profiles.get("u1").interaction_count == 1


def test_manager_remember_survives_extractor_failure(store):
    manager = make_manager(store, extractor=FakeLLM(exc=RuntimeError("boom")))
    result = asyncio.run(manager.remember("c1", "u1", "这是一句足够长的买家消息", "回复"))
    assert result["skipped"] == "extraction_failed"
    assert result["added"] == 0


def test_manager_remember_falls_back_to_chat_scope_without_user_id(store):
    insight = TurnInsight(
        memories=[ExtractedMemory(kind="fact", content="买家提到明天要出差", weight=1.0)]
    )
    manager = make_manager(store, extractor=FakeLLM(response=insight))
    asyncio.run(manager.remember("c1", None, "我明天要出差，能今天发货吗", "我尽量安排"))
    assert store.count_memories("chat") == 1
    assert store.count_memories("user") == 0


def test_manager_stats_shape(store):
    manager = make_manager(store)
    stats = manager.stats()
    assert stats["enabled"] is True
    assert "store" in stats and "tool_memory" in stats and "long_term" in stats
