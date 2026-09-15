"""短期记忆：对话窗口构建 + **摘要压缩**。

解决的问题
----------
原先 ``agent`` 节点只做硬截断：

    history = list(state["messages"])[-AGENT_MAX_MESSAGES:]

``AGENT_MAX_MESSAGES``（默认 20）之外的对话**永久丢失**。聊到第 30 轮时，机器人
已经看不见前 10 轮说过什么，表现为「你刚才不是说过吗」「不好意思我确认一下」。

现在的做法是**压缩而不是丢弃**：

    历史消息 ─┬─ 最近 max_messages 条 ──→ 原样送入 LLM
              └─ 更早的消息 ──────────→ LLM 摘要 ──→ 作为一段「对话摘要」注入

摘要**增量累积**：只对「已经滑出窗口、且尚未被摘要过」的那一段做压缩，
并用 ``summarized_count`` 记录边界，避免重复摘要导致成本翻倍。

为什么要独立设计而不是「每轮都重新摘要」
------------------------------------------
每轮重新摘要整段历史是 O(n²) 的成本增长，且会不断丢细节（摘要的摘要）。
增量方案下，每条消息**只被摘要一次**，成本是线性的。

环境变量
--------
- ``MEMORY_SUMMARY_ENABLED``  是否启用摘要压缩，默认 ``true``
- ``MEMORY_SUMMARY_TRIGGER``  累积多少条「滑出窗口」的消息后触发一次摘要，默认 ``16``
- ``AGENT_MAX_MESSAGES``      保留原文的窗口大小，默认 ``20``
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from loguru import logger

from koiagent.infra.text import truncate

SUMMARIZE_PROMPT = """你是对话摘要助手。请把「已有摘要」与「新增对话片段」合并压缩成一段**简洁的中文摘要**。

摘要必须保留（如果出现过）：
1. 买家已明确表达的需求、关注点、疑问
2. 已经给出过的关键答复与承诺（尤其是价格、优惠、发货、售后）
3. 卖家的底线与已做出的让步（避免重复让价）
4. 买家的偏好与约束（预算上限、规格要求、时间要求、禁忌）
5. 尚未解决的悬置问题

摘要必须丢弃：寒暄、重复表达、工具调用的原始返回内容。

要求：
- 不超过 300 字
- 用第三人称陈述，不要「他说/我说」这种流水账
- 只输出摘要正文，不要任何前缀、标题或解释

【已有摘要】
{summary}

【新增对话片段】
{transcript}
"""

_ROLE_LABEL = {"human": "买家", "ai": "客服", "tool": "工具结果", "system": "系统"}


def message_text(message: Any) -> str:
    """把一条消息压平成纯文本（``content`` 可能是 list，需要拼接）。"""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        content = " ".join(parts)
    return str(content or "").strip()


def render_transcript(messages: Sequence[Any], per_message_limit: int = 300) -> str:
    """把消息序列渲染成给摘要模型看的文本。"""
    lines: List[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "ai"
        elif isinstance(message, ToolMessage):
            role = "tool"
        else:
            role = "system"
        text = message_text(message)
        if not text:
            continue
        lines.append(f"{_ROLE_LABEL[role]}: {truncate(text, per_message_limit)}")
    return "\n".join(lines)


class ShortTermMemory:
    """短期记忆：决定哪些消息保留原文、哪些需要压缩成摘要。"""

    def __init__(
        self,
        max_messages: Optional[int] = None,
        trigger: Optional[int] = None,
        enabled: Optional[bool] = None,
    ):
        self.max_messages = int(max_messages or os.getenv("AGENT_MAX_MESSAGES", "20"))
        self.trigger = int(trigger or os.getenv("MEMORY_SUMMARY_TRIGGER", "16"))
        if enabled is None:
            enabled = os.getenv("MEMORY_SUMMARY_ENABLED", "true").strip().lower() != "false"
        self.enabled = bool(enabled)

    # ------------------------------------------------------------------ #
    # 边界计算
    # ------------------------------------------------------------------ #
    def boundary(self, total: int) -> int:
        """当前窗口的起始下标：此下标之前的消息都应被摘要覆盖。"""
        return max(0, total - self.max_messages)

    def pending_range(self, total: int, summarized_count: int) -> Tuple[int, int]:
        """返回「尚未摘要、且已滑出窗口」的消息区间 ``[start, end)``。"""
        start = max(0, min(summarized_count, total))
        end = self.boundary(total)
        return (start, end) if end > start else (0, 0)

    def should_summarize(self, total: int, summarized_count: int) -> bool:
        """是否值得为累积的消息付一次摘要调用。"""
        if not self.enabled:
            return False
        start, end = self.pending_range(total, summarized_count)
        return (end - start) >= self.trigger

    # ------------------------------------------------------------------ #
    # 摘要
    # ------------------------------------------------------------------ #
    async def summarize(
        self,
        llm: Any,
        previous_summary: str,
        messages: Sequence[Any],
    ) -> str:
        """把 ``messages`` 增量合并进 ``previous_summary``，返回新摘要。

        失败时**返回原摘要**而不是抛错 —— 摘要是增强能力，不应中断对话。
        """
        transcript = render_transcript(messages)
        if not transcript.strip():
            return previous_summary

        prompt = SUMMARIZE_PROMPT.format(
            summary=previous_summary.strip() or "（暂无）",
            transcript=truncate(transcript, 6000),
        )
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            text = message_text(response)
            if text:
                logger.info(f"[memory] 短期记忆已压缩 {len(messages)} 条消息 -> {len(text)} 字摘要")
                return text
        except Exception as e:
            logger.warning(f"[memory] 对话摘要失败，保留原摘要: {e}")
        return previous_summary

    # ------------------------------------------------------------------ #
    # 上下文构建
    # ------------------------------------------------------------------ #
    def window(self, messages: Sequence[Any]) -> List[Any]:
        """取最近 ``max_messages`` 条消息作为原文窗口。"""
        items = list(messages or [])
        return items[-self.max_messages :] if self.max_messages > 0 else items

    def build_context(self, summary: str, messages: Sequence[Any]) -> str:
        """渲染短期记忆：摘要在前，原文窗口在后。"""
        parts: List[str] = []
        if summary and summary.strip():
            parts.append(f"【较早对话的摘要】{summary.strip()}")
        recent = render_transcript(self.window(messages), per_message_limit=200)
        if recent:
            parts.append(f"【最近对话原文】\n{recent}")
        return "\n".join(parts)
