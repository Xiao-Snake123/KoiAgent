"""KoiAgent 命令行入口：``python -m koiagent``（根目录 ``python main.py`` 等价）。

职责：装配配置 → 构建 Agent 与运行时 → 常驻运行 → 优雅退出并打印运行指标。
"""
from __future__ import annotations

import asyncio
import os
import sys

from loguru import logger

from koiagent.agent.graph import KoiReplyBot
from koiagent.app import KoiLive
from koiagent.config import check_and_complete_env, load_env, setup_logging


def main() -> int:
    load_env()
    setup_logging()
    check_and_complete_env()

    cookies_str = os.getenv("COOKIES_STR") or ""
    bot = KoiReplyBot()
    live = KoiLive(cookies_str, bot)

    # 常驻进程（Ctrl+C 优雅退出，并打印本次运行指标）
    try:
        asyncio.run(live.run())
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止 KoiAgent ...")
    finally:
        logger.info("\n" + bot.tracer.format_report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
