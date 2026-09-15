"""LangGraph 编排层结构测试：验证图节点、工具注册、异步接口与状态定义。"""
import asyncio
import inspect

import pytest

from koiagent.agent.graph import ALL_TOOLS, AgentState, KoiReplyBot


@pytest.fixture(scope="module")
def bot():
    return KoiReplyBot()


def test_graph_has_expected_nodes(bot):
    nodes = set(bot.graph.get_graph().nodes)
    assert {"guard", "recall", "classify", "agent", "tools", "critic", "finalize"} <= nodes


def test_guard_routes_to_recall_when_allowed(bot):
    """放行的输入要先经过记忆召回，才能进入意图识别与生成。"""
    assert KoiReplyBot._route_after_guard({"guard_action": "allow"}) == "recall"
    assert KoiReplyBot._route_after_guard({"guard_action": "block"}) == "finalize"


def test_tools_are_registered():
    names = {tool.name for tool in ALL_TOOLS}
    assert {"get_bargain_policy", "search_knowledge_base", "get_current_time"} <= names


def test_bargain_policy_comes_from_config_file(bot):
    """议价策略已从硬编码改为配置文件驱动，且支持热更新。"""
    from koiagent.agent.bargain import get_bargain_policy_store

    store = get_bargain_policy_store()
    text = store.advise(2)
    assert "议价轮次=2" in text
    assert "累计让价上限" in text
    assert store.describe()["tiers"] >= 1


def test_bargain_policy_survives_bad_config(tmp_path, monkeypatch):
    """配置文件损坏时必须回退内置默认策略，不影响功能。"""
    from koiagent.agent.bargain import BargainPolicy

    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    policy = BargainPolicy(path=str(bad))
    assert "让步策略" in policy.advise(1)


def test_bargain_policy_rejects_illegal_tiers(tmp_path):
    from koiagent.agent.bargain import BargainPolicy

    bad = tmp_path / "bad.json"
    bad.write_text('{"tiers": [{"max_round": 1}]}', encoding="utf-8")
    policy = BargainPolicy(path=str(bad))
    assert policy.source == "builtin"


def test_bot_exposes_required_interface(bot):
    """main.py 依赖的对外接口必须保持稳定。"""
    assert callable(bot.agenerate_reply)
    assert inspect.iscoroutinefunction(bot.agenerate_reply)
    assert callable(bot.generate_reply)  # 同步封装，供脚本 / 测试使用
    assert callable(bot.record_message)
    assert hasattr(bot, "last_intent")
    assert hasattr(bot, "tracer")


def test_nodes_are_async(bot):
    """节点必须是协程，才能配合 ainvoke 非阻塞执行。"""
    assert inspect.iscoroutinefunction(bot._classify)
    assert inspect.iscoroutinefunction(bot._agent)


def test_sync_wrapper_refuses_running_loop(bot):
    """在事件循环内调用同步封装必须显式报错，避免静默阻塞事件循环。"""

    async def _call():
        with pytest.raises(RuntimeError, match="agenerate_reply"):
            bot.generate_reply("在吗", "商品描述")

    asyncio.run(_call())


def test_agent_state_has_routing_field():
    assert "routing" in AgentState.__annotations__
    assert "intent" in AgentState.__annotations__


def test_agent_state_has_memory_fields():
    """记忆系统依赖这些状态字段在图内传递，缺失会导致静默丢失记忆。"""
    for field in (
        "summary",
        "summarized_count",
        "memory_text",
        "memory_facts",
        "profile_hit",
        "chat_id",
        "user_id",
    ):
        assert field in AgentState.__annotations__, f"AgentState 缺少字段 {field}"


def test_recall_node_returns_memory_snapshot(bot):
    """recall 节点在没有历史记忆时也必须返回可用结构，不能抛错。"""
    updates = asyncio.run(
        bot._recall(
            {"user_msg": "电池容量多少", "chat_id": "t1", "user_id": "u1", "messages": []}
        )
    )
    assert "memory_text" in updates
    assert isinstance(updates["memory_facts"], int)
    assert isinstance(updates["profile_hit"], bool)


def test_system_message_injects_memory(bot):
    message = bot._build_system_message(
        {
            "intent": "tech",
            "memory_text": "【买家画像】意向：强意向",
            "summary": "买家之前问过续航",
        }
    )
    assert "买家画像" in message.content
    assert "较早对话的摘要" in message.content


def test_system_message_tells_model_to_prefer_current_words(bot):
    """陈旧记忆与买家当前说法冲突时，必须以当前说法为准。"""
    message = bot._build_system_message({"intent": "default", "memory_text": "【买家画像】预算 500"})
    assert "以买家当前的说法为准" in message.content


def test_agent_uses_short_term_window(bot):
    """agent 节点只取窗口内的原文；超出部分由 recall 节点的摘要覆盖。"""
    from langchain_core.messages import HumanMessage

    memory = bot.memory.short_term
    messages = [HumanMessage(content=f"m{i}") for i in range(memory.max_messages + 5)]
    windowed = memory.window(messages)

    assert len(windowed) == memory.max_messages
    assert windowed[-1].content == messages[-1].content   # 保留最新的
    assert windowed[0].content == "m5"                     # 最旧的 5 条已滑出窗口


def test_aclose_is_safe_without_pending_tasks(bot):
    asyncio.run(bot.aclose())


def test_background_error_callback_consumes_exception(bot):
    """后台任务异常必须被消费，否则 asyncio 只会留下一条 never-retrieved 警告。"""

    async def boom():
        raise RuntimeError("背景任务失败")

    async def run():
        task = asyncio.create_task(boom())
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError):
            task.result()
        KoiReplyBot._log_background_error(task)  # 不应再抛

    asyncio.run(run())


def test_max_steps_is_positive(bot):
    assert bot.max_steps >= 1


def test_collect_tool_names_from_messages():
    from langchain_core.messages import AIMessage

    messages = [
        AIMessage(content="", tool_calls=[{"name": "get_bargain_policy", "args": {}, "id": "1"}]),
        AIMessage(content="你好"),
    ]
    assert KoiReplyBot._collect_tool_names(messages) == ["get_bargain_policy"]
    assert KoiReplyBot._collect_tool_names([]) == []
