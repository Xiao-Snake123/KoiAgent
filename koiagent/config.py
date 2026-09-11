"""配置加载与校验：``.env`` 装载、日志初始化、缺失配置的交互式补全。"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv, set_key
from loguru import logger

# 占位符：与 .env.example 保持一致；命中即视为「未配置」
PLACEHOLDERS = {
    "API_KEY": "默认使用通义千问,apikey通过百炼模型平台获取",
    "COOKIES_STR": "your_cookies_here",
}

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


def load_env() -> None:
    """按 ``.env`` → ``.env.example`` 顺序加载（后者不覆盖已存在的变量）。"""
    if os.path.exists(".env"):
        load_dotenv()
        logger.info("已加载 .env 配置")
    if os.path.exists(".env.example"):
        load_dotenv(".env.example")  # 不会覆盖已存在的变量
        logger.info("已加载 .env.example 默认配置")


def setup_logging() -> str:
    """初始化 loguru（输出到 stderr），返回生效的日志级别。"""
    level = os.getenv("LOG_LEVEL", "DEBUG").upper()
    logger.remove()  # 移除默认 handler
    logger.add(sys.stderr, level=level, format=LOG_FORMAT)
    logger.info(f"日志级别设置为: {level}")
    return level


def check_and_complete_env() -> bool:
    """交互式补全缺失的必填配置并写回 ``.env``，返回是否有更新。"""
    env_path = ".env"
    updated = False

    for key, placeholder in PLACEHOLDERS.items():
        current = os.getenv(key)
        if current and current != placeholder:
            continue  # 已正确配置

        logger.warning(f"配置项 [{key}] 未设置或为默认值，请输入")
        while True:
            value = input(f"请输入 {key}: ").strip()
            if not value:
                print(f"{key} 不能为空，请重新输入")
                continue

            os.environ[key] = value
            try:
                if not os.path.exists(env_path):
                    open(env_path, "w", encoding="utf-8").close()
                set_key(env_path, key, value)
                updated = True
            except Exception as e:
                logger.warning(f"无法自动写入.env文件，请手动保存: {e}")
            break

    if updated:
        logger.info("新的配置已保存/更新至 .env 文件中")
    return updated
