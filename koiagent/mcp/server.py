"""KoiAgent MCP Server —— 把项目能力通过 **Model Context Protocol** 暴露给任意 MCP 客户端。

暴露的能力
----------
| 工具 | 作用 | 是否需要 LLM |
| ---- | ---- | ---- |
| `get_bargain_policy` | 按议价轮次返回阶梯让步策略 | 否 |
| `search_knowledge_base` | RAG 检索本地知识库 | 否 |
| `get_current_time` | 当前服务器时间 | 否 |
| `ask_koi_agent` | 委派完整 Agent 图处理（防护→路由→工具→审核反思） | **是** |

启动
----
::

    python -m koiagent.mcp.server                  # stdio（Claude Desktop / Cursor 默认）
    python -m koiagent.mcp.server --transport streamable-http

客户端配置示例（``claude_desktop_config.json``）::

    {
      "mcpServers": {
        "koi-agent": {
          "command": "python",
          "args": ["-m", "koiagent.mcp.server"],
          "cwd": "C:/path/to/KoiAgent"
        }
      }
    }

设计说明
--------
1. **单一事实来源**：工具实现复用 ``koiagent.agent.tools``，避免 MCP 与 Agent 行为漂移。
2. **可离线使用**：仅暴露纯工具时**无需任何 API Key**；只有 ``ask_koi_agent``
   需要 LLM 配置，未配置时返回明确提示而非抛错。
3. **协议安全**：stdio 传输下 **stdout 必须独占给协议帧**，因此日志被强制重定向到
   stderr（loguru 默认即 stderr，此处显式强化，防止未来误加 print 破坏协议）。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from loguru import logger
from mcp.server.mcpserver import MCPServer

from koiagent.agent.tools import (
    get_bargain_policy as _bargain_tool,
    get_current_time as _time_tool,
    search_knowledge_base as _kb_tool,
)

SERVER_NAME = "koi-agent"
SERVER_VERSION = "1.0.0"

_INSTRUCTIONS = (
    "KoiAgent 是面向电商客服场景的 AI 值守 Agent。"
    "可直接调用其工具查询议价策略、检索本地知识库、获取当前时间；"
    "若已配置 API_KEY，可用 ask_koi_agent 委派完整的客服推理链路。"
)

server = MCPServer(
    name=SERVER_NAME,
    version=SERVER_VERSION,
    instructions=_INSTRUCTIONS,
)


def _invoke(tool_obj: Any, **kwargs: Any) -> str:
    """调用 LangChain 工具对象并统一为字符串返回。"""
    result = tool_obj.invoke(kwargs)
    return result if isinstance(result, str) else str(result)


# --------------------------------------------------------------------------- #
# 纯工具（无需 LLM，离线可用）
# --------------------------------------------------------------------------- #
@server.tool(description="查询指定议价轮次下应采用的阶梯让步策略与让价上限。")
def get_bargain_policy(bargain_count: int) -> str:
    """查询阶梯议价策略。

    Args:
        bargain_count: 当前会话已发生的议价轮次，从 1 开始计数。
    """
    return _invoke(_bargain_tool, bargain_count=bargain_count)


@server.tool(description="在本地商品/技术知识库中做 RAG 检索，用于回答参数、规格、售后等问题。")
def search_knowledge_base(query: str) -> str:
    """检索本地知识库。

    Args:
        query: 检索关键词或用户原问题。
    """
    return _invoke(_kb_tool, query=query)


@server.tool(description="获取当前服务器时间（Asia/Shanghai）。")
def get_current_time() -> str:
    """获取当前服务器时间，用于回答发货时效等问题。"""
    return _invoke(_time_tool)


# --------------------------------------------------------------------------- #
# 完整 Agent 工具（需要 LLM 配置）
# --------------------------------------------------------------------------- #
_bot: Optional[Any] = None
_bot_failed = False


def _get_bot() -> Optional[Any]:
    """惰性构建 ``KoiReplyBot``；缺少 LLM 配置或初始化失败时返回 None。"""
    global _bot, _bot_failed
    if _bot is not None or _bot_failed:
        return _bot
    if not (os.getenv("API_KEY") or "").strip():
        logger.warning("未配置 API_KEY，ask_koi_agent 不可用（其余工具不受影响）")
        _bot_failed = True
        return None
    try:
        from koiagent.agent.graph import KoiReplyBot

        _bot = KoiReplyBot()
        logger.info("KoiAgent 已就绪，ask_koi_agent 可用")
    except Exception as e:
        logger.warning(f"KoiAgent 初始化失败，ask_koi_agent 不可用: {e}")
        _bot_failed = True
    return _bot


@server.tool(
    description=(
        "把买家问题委派给完整的 KoiAgent 图处理"
        "（注入防护 → 意图路由 → 工具调用 → 审核反思），返回可直接发送的回复文本。"
        "需要预先配置 API_KEY。"
    ),
)
async def ask_koi_agent(question: str, item_description: str = "", chat_id: str = "") -> str:
    """委派完整 Agent 生成客服回复。

    Args:
        question: 买家的消息内容。
        item_description: 商品信息（标题/价格/规格），便于 Agent 结合上下文作答。
        chat_id: 会话标识；传入同一值可复用多轮记忆，留空则为单轮无状态调用。
    """
    bot = _get_bot()
    if bot is None:
        return "KoiAgent 未就绪：请先配置 API_KEY（模型 API）后再调用本工具。"
    return await bot.agenerate_reply(question, item_description, chat_id=chat_id or None)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def _load_env() -> bool:
    """加载当前工作目录下的 ``.env``（供工具与 ask_koi_agent 读取配置）。

    - 只加载 ``.env`` 而**不**加载 ``.env.example``：后者是占位符模板，
      加载后会让 API_KEY 看起来已配置，反而掩盖真实的未配置状态。
    - 显式传入路径而非依赖 ``load_dotenv()`` 的默认查找：后者的 ``find_dotenv``
      从**调用方模块所在目录**向上搜索，与 cwd 不一定一致，行为不可预期。
    """
    env_file = Path(os.getcwd(), ".env")
    if not env_file.is_file():
        return False
    load_dotenv(env_file)
    logger.info("已加载 .env 配置")
    return True


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="KoiAgent MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        help="传输方式，默认 stdio",
    )
    args = parser.parse_args(argv)

    # 协议安全：stdio 传输下 stdout 必须独占给协议帧，日志显式重定向到 stderr
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO").upper())

    _load_env()
    logger.info(f"KoiAgent MCP Server 启动 (transport={args.transport})")
    server.run(args.transport)
    return 0


if __name__ == "__main__":
    sys.exit(main())
