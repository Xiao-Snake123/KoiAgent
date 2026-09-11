"""MCP Server 测试：工具注册、协议调用、stdout 纯净性、.env 加载与缺省降级（全程离线）。"""
import asyncio
import contextlib
import io
import os

from koiagent.mcp.server import SERVER_NAME, server

EXPECTED_TOOLS = {
    "get_bargain_policy",
    "search_knowledge_base",
    "get_current_time",
    "ask_koi_agent",
}


def _call(name: str, arguments: dict | None = None):
    """通过 MCP 协议层调用工具（与真实客户端路径一致）。"""
    return asyncio.run(server.call_tool(name, arguments or {}))


def _text(result) -> str:
    return "".join(getattr(item, "text", "") for item in result.content)


def test_server_metadata():
    assert SERVER_NAME == "koi-agent"
    assert server.name == "koi-agent"


def test_all_tools_registered_with_schema():
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}

    assert EXPECTED_TOOLS <= names
    for tool in tools:
        assert tool.description, f"{tool.name} 缺少描述"
        assert tool.input_schema.get("type") == "object", f"{tool.name} schema 非法"


def test_bargain_policy_via_protocol():
    result = _call("get_bargain_policy", {"bargain_count": 1})
    assert not getattr(result, "isError", False)
    assert "首轮" in _text(result)


def test_knowledge_search_via_protocol():
    result = _call("search_knowledge_base", {"query": "蓝牙 续航"})
    assert not getattr(result, "isError", False)
    assert "蓝牙" in _text(result)


def test_current_time_via_protocol():
    text = _text(_call("get_current_time"))
    assert len(text) == 19 and text[4] == "-" and text[13] == ":"


def test_stdout_stays_clean_for_protocol():
    """stdio 传输下 stdout 只能承载协议帧，工具调用不得污染 stdout。"""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _call("get_bargain_policy", {"bargain_count": 2})
        _call("search_knowledge_base", {"query": "发货"})
        _call("get_current_time")

    assert buffer.getvalue() == "", f"stdout 被污染: {buffer.getvalue()[:200]!r}"


def test_load_env_reads_dotenv_file(monkeypatch, tmp_path):
    """启动时应能加载 .env，供工具与 ask_koi_agent 读取配置。"""
    import koiagent.mcp.server as mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KOI_TEST_FLAG", raising=False)
    (tmp_path / ".env").write_text("KOI_TEST_FLAG=42\n", encoding="utf-8")

    assert mod._load_env() is True
    assert os.getenv("KOI_TEST_FLAG") == "42"


def test_load_env_absent_returns_false(monkeypatch, tmp_path):
    """没有 .env 时应返回 False，不会因 .env.example 而误判为已配置。"""
    import koiagent.mcp.server as mod

    monkeypatch.chdir(tmp_path)
    assert mod._load_env() is False


def test_ask_koi_agent_degrades_without_api_key(monkeypatch):
    """未配置 API_KEY 时应返回明确提示，而不是抛错。"""
    import koiagent.mcp.server as mod

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(mod, "_bot", None)
    monkeypatch.setattr(mod, "_bot_failed", False)

    result = _call("ask_koi_agent", {"question": "这个能便宜点吗", "item_description": "商品"})

    assert not getattr(result, "isError", False)
    assert "未就绪" in _text(result)
