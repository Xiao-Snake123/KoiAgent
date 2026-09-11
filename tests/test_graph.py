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
    assert {"guard", "classify", "agent", "tools", "critic", "finalize"} <= nodes


def test_tools_are_registered():
    names = {tool.name for tool in ALL_TOOLS}
    assert {"get_bargain_policy", "search_knowledge_base", "get_current_time"} <= names


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
