"""KoiAgent 命令行入口：``python -m koiagent``（根目录 ``python main.py`` 等价）。

职责：装配配置 → 构建 Agent 与运行时 → 常驻运行 → 优雅退出并打印运行指标。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from loguru import logger

from koiagent.agent.graph import KoiReplyBot
from koiagent.app import KoiLive
from koiagent.config import (
    check_and_complete_env,
    describe_config,
    load_env,
    setup_logging,
    validate_config,
)


async def _shutdown(bot: KoiReplyBot) -> None:
    """等待后台记忆写回任务收尾（异常不影响退出流程）。"""
    try:
        await bot.aclose()
    except Exception as e:  # pragma: no cover - 退出路径，仅记录
        logger.debug(f"后台任务收尾异常（忽略）: {e}")


async def _run(live: KoiLive, bot: KoiReplyBot) -> None:
    """常驻主循环 + 退出前收尾。

    记忆写回是**后台任务**：如果进程直接退出，尚未落库的那一轮记忆会丢失。
    因此在 ``finally`` 里等待它们完成。用 ``asyncio.shield`` 避免
    Ctrl+C 触发的取消把收尾本身也取消掉。
    """
    try:
        await live.run()
    finally:
        try:
            await asyncio.shield(_shutdown(bot))
        except asyncio.CancelledError:
            logger.debug("主任务被取消，跳过后台记忆任务收尾")


def main() -> int:
    load_env()
    setup_logging()
    check_and_complete_env()
    # 启动期校验全部数值/布尔配置：把「填错配置」从运行期崩溃提前到启动期失败
    validate_config(strict=True)
    logger.info(f"配置快照: {json.dumps(describe_config(), ensure_ascii=False)}")

    cookies_str = os.getenv("COOKIES_STR") or ""
    bot = KoiReplyBot()
    live = KoiLive(cookies_str, bot)

    # 常驻进程（Ctrl+C 优雅退出，并打印本次运行指标）
    try:
        asyncio.run(_run(live, bot))
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止 KoiAgent ...")
    finally:
        logger.info("\n" + bot.tracer.format_report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
