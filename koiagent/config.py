"""配置加载与校验：``.env`` 装载、日志初始化、缺失配置的交互式补全。"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

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


# --------------------------------------------------------------------------- #
# 配置校验（启动期快速失败）
# --------------------------------------------------------------------------- #
# 背景：代码里大量使用 `int(os.getenv("AGENT_MAX_STEPS", "4"))` 这类读取。
# 若用户把值填成 `abc`，会在**收到第一条消息时**才抛 ValueError —— 此时进程已经
# 连上平台、可能已回复过部分消息，排障成本远高于启动即失败。
#
# 因此这里用声明式 schema 在启动期把所有数值/布尔配置一次性解析并校验。
#
# 结构：key -> (类型, 最小值, 最大值, 默认值)。min/max 为 None 表示不做区间校验。
CONFIG_SCHEMA: Dict[str, Tuple] = {
    # ---- 平台行为 ----
    "HEARTBEAT_INTERVAL": (int, 1, 3600, "15"),
    "HEARTBEAT_TIMEOUT": (int, 1, 3600, "5"),
    "TOKEN_REFRESH_INTERVAL": (int, 60, 86400, "3600"),
    "TOKEN_RETRY_INTERVAL": (int, 10, 86400, "300"),
    "MANUAL_MODE_TIMEOUT": (int, 60, 604800, "3600"),
    "MESSAGE_EXPIRE_TIME": (int, 1000, 86400000, "300000"),
    "SIMULATE_HUMAN_TYPING": (bool, None, None, "False"),
    # ---- Agent 编排 ----
    "AGENT_MAX_STEPS": (int, 1, 20, "4"),
    "AGENT_MAX_MESSAGES": (int, 2, 200, "20"),
    "AGENT_MAX_REFLECTIONS": (int, 0, 5, "1"),
    "CRITIC_ENABLED": (bool, None, None, "true"),
    # ---- RAG ----
    "EMBEDDING_BATCH_SIZE": (int, 1, 256, "16"),
    "RAG_TOP_K": (int, 1, 20, "3"),
    "RAG_CHUNK_SIZE": (int, 50, 4000, "300"),
    "RAG_CHUNK_OVERLAP": (int, 0, 2000, "60"),
    # ---- 注入防护 ----
    "GUARD_ENABLED": (bool, None, None, "true"),
    "GUARD_BLOCK_SCORE": (int, 1, 100, "5"),
    "GUARD_MAX_INPUT": (int, 10, 10000, "1000"),
    # ---- 可靠性 ----
    "LLM_MAX_CONCURRENCY": (int, 1, 64, "4"),
    "CIRCUIT_FAILURE_THRESHOLD": (int, 1, 1000, "5"),
    "CIRCUIT_RESET_TIMEOUT": (float, 1, 86400, "60"),
    "DEDUP_CACHE_SIZE": (int, 1, 1000000, "2000"),
    "DEDUP_TTL": (float, 1, 86400, "300"),
    "MEMORY_MAX_THREADS": (int, 1, 1000000, "500"),
    "MEMORY_TTL": (float, 60, 2592000, "7200"),
    "MEMORY_REAP_INTERVAL": (float, 1, 86400, "60"),
    # ---- 可观测性 ----
    "METRICS_SAMPLE_LIMIT": (int, 10, 1000000, "1000"),
    "LLM_TIMEOUT": (float, 1, 600, "30"),
    "LLM_MAX_RETRIES": (int, 0, 10, "2"),
    "TRACE_ENABLED": (bool, None, None, "true"),
    "TRACE_MAX_BYTES": (int, 0, 1073741824, str(10 * 1024 * 1024)),
    "TRACE_BACKUP_COUNT": (int, 0, 100, "5"),
    "COST_INPUT_PER_1K": (float, 0, 100000, None),
    "COST_OUTPUT_PER_1K": (float, 0, 100000, None),
    # ---- 运维 ----
    "HEALTH_MAX_AGE": (float, 1, 86400, "180"),
    # ---- 记忆系统 ----
    "MEMORY_ENABLED": (bool, None, None, "true"),
    "MEMORY_TOP_K": (int, 1, 20, "5"),
    "MEMORY_MAX_FACTS": (int, 1, 500, "100"),
    "MEMORY_MIN_SCORE": (int, 1, 100, "2"),
    "MEMORY_DECAY_DAYS": (float, 0, 3650, "30"),
    "MEMORY_SUMMARY_ENABLED": (bool, None, None, "true"),
    "MEMORY_SUMMARY_TRIGGER": (int, 4, 200, "16"),
    "MEMORY_MIN_INPUT_LEN": (int, 0, 1000, "10"),
    "MEMORY_ASYNC_WRITE": (bool, None, None, "true"),
    "PROFILE_ENABLED": (bool, None, None, "true"),
    "PROFILE_MAX_LIST_ITEMS": (int, 1, 50, "10"),
    "TOOL_MEMORY_ENABLED": (bool, None, None, "true"),
    "TOOL_MEMORY_TTL": (float, 1, 86400, "900"),
    "TOOL_MEMORY_MAX_ENTRIES": (int, 1, 100000, "500"),
    # ---- 热更新 ----
    "KB_RELOAD_INTERVAL": (float, 0, 3600, "30"),
    "BARGAIN_POLICY_RELOAD_INTERVAL": (float, 0, 3600, "30"),
    # ---- 出站限频 ----
    "SEND_MIN_INTERVAL": (float, 0, 600, "0"),
    "SEND_BURST": (int, 1, 100, "3"),
}

_TRUE_VALUES = {"1", "true", "yes", "on", "y"}
_FALSE_VALUES = {"0", "false", "no", "off", "n", ""}


def _parse_typed(key: str, raw: str, typ: type, lo, hi) -> Tuple[Optional[Any], Optional[str]]:
    """解析并校验单个配置项，返回 ``(值, 错误信息)``；解析失败时值为 None。"""
    text = (raw or "").strip()
    if typ is bool:
        lowered = text.lower()
        if lowered in _TRUE_VALUES:
            return True, None
        if lowered in _FALSE_VALUES:
            return False, None
        return None, f"{key}={text!r} 不是合法布尔值（可用 true/false、1/0、yes/no）"
    try:
        value = typ(text) if text else typ(0)
    except ValueError:
        return None, f"{key}={text!r} 不是合法{'整数' if typ is int else '数字'}"
    if lo is not None and value < lo:
        return None, f"{key}={value} 小于允许的最小值 {lo}"
    if hi is not None and value > hi:
        return None, f"{key}={value} 大于允许的最大值 {hi}"
    return value, None


def validate_config(strict: bool = False) -> List[str]:
    """校验全部声明式配置项，返回**错误**信息列表（警告只记日志）。

    参数 ``strict=True`` 时，只要存在错误就抛出 ``ValueError``，
    用于在进程启动阶段快速失败；``False`` 时仅返回错误列表供调用方决策。
    """
    errors: List[str] = []
    warnings: List[str] = []

    for key, (typ, lo, hi, default) in CONFIG_SCHEMA.items():
        raw = os.getenv(key)
        if raw is None or (raw or "").strip() == "":
            if default is None:
                continue  # 可选项，未设置即不校验
            # 未设置时用默认值兜底，保证下游 int()/float() 不会炸
            os.environ[key] = default
            raw = default

        value, err = _parse_typed(key, raw, typ, lo, hi)
        if err:
            errors.append(err)
        else:
            # 回写规范化后的值（例如 "TRUE" -> "true"），统一下游读取行为
            os.environ[key] = str(value)

    # ---- 跨字段校验 ----
    try:
        chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "300"))
        chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "60"))
        if chunk_overlap >= chunk_size:
            errors.append(
                f"RAG_CHUNK_OVERLAP={chunk_overlap} 必须小于 RAG_CHUNK_SIZE={chunk_size}，"
                "否则切片无法前进（会死循环或产生重复片段）"
            )
    except ValueError:
        pass

    try:
        reap_interval = float(os.getenv("MEMORY_REAP_INTERVAL", "60"))
        memory_ttl = float(os.getenv("MEMORY_TTL", "7200"))
        if reap_interval >= memory_ttl:
            warnings.append(
                f"MEMORY_REAP_INTERVAL={reap_interval:.0f}s 不小于 MEMORY_TTL={memory_ttl:.0f}s，"
                "清理几乎不会触发，空闲会话记忆会长期滞留"
            )
    except ValueError:
        pass

    if (os.getenv("EMBEDDING_MODEL") or "").strip():
        if not ((os.getenv("EMBEDDING_API_KEY") or "").strip() or (os.getenv("API_KEY") or "").strip()):
            warnings.append("已设置 EMBEDDING_MODEL 但缺少 API Key，RAG 将回退关键词检索")

    if os.getenv("CRITIC_ENABLED", "true").strip().lower() == "false":
        warnings.append("CRITIC_ENABLED=false：审核 Agent 已关闭，输出质量将只依赖生成侧提示词")

    for message in warnings:
        logger.warning(f"[配置] {message}")
    for message in errors:
        logger.error(f"[配置] {message}")

    if errors:
        logger.error(f"[配置] 共 {len(errors)} 项配置错误，请检查 .env（参考 .env.example）")
        if strict:
            raise ValueError("配置校验失败：" + "；".join(errors))

    return errors


def describe_config() -> Dict[str, Any]:
    """返回当前生效的关键配置快照（用于启动日志，**不含任何密钥**）。"""
    keys = [
        "MODEL_NAME", "MODEL_BASE_URL", "AGENT_MAX_STEPS", "AGENT_MAX_MESSAGES",
        "AGENT_MAX_REFLECTIONS", "CRITIC_ENABLED", "GUARD_ENABLED", "GUARD_BLOCK_SCORE",
        "LLM_MAX_CONCURRENCY", "LLM_TIMEOUT", "RAG_TOP_K", "EMBEDDING_MODEL",
        "MEMORY_ENABLED", "PROFILE_ENABLED", "TOOL_MEMORY_ENABLED", "TRACE_ENABLED",
    ]
    snapshot: Dict[str, Any] = {}
    for key in keys:
        value = (os.getenv(key) or "").strip()
        if key == "EMBEDDING_MODEL":
            value = value or "<未配置，使用关键词检索>"
        elif "KEY" in key:
            value = "***" if value else "<未配置>"
        snapshot[key] = value or "<未配置>"
    snapshot["API_KEY"] = "已配置" if (os.getenv("API_KEY") or "").strip() else "未配置"
    return snapshot
