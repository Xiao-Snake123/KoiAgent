"""长期记忆：从对话中抽取事实、持久化、按相关度召回、按衰减淘汰。

长期记忆与短期记忆的分工
------------------------
- **短期记忆**回答「这段对话刚才说了什么」——有界窗口 + 摘要，随对话推进而滑动。
- **长期记忆**回答「这个买家一直以来是什么情况」——跨会话保留，按需检索。

后者的价值在于：买家三天后再来，机器人仍然记得「他说过预算 500 以内」「他是做
摄影的」「上次答应给他包邮」。这些信息需要**从对话中主动抽取并结构化保存**，
而不是靠把全部历史都塞进上下文（会被 `AGENT_MAX_MESSAGES` 截断，成本也爆炸）。

抽取策略
--------
用一次**低温 + 结构化输出**的 LLM 调用，从「买家消息 + 客服回复」中抽取 4 类信息：

==============  ==========================================================
kind            含义与示例
==============  ==========================================================
``fact``        客观事实：「买家是做摄影的」「已有一台同型号」
``preference``  偏好：「偏爱国产」「希望安静一点」
``objection``   异议：「觉得贵」「担心续航」
``commitment``  承诺/约定：「已答应包邮」「答应明天发货」
==============  ==========================================================

**只抽取稳定信息**，不抽一次性内容（如「你好」）。抽取失败时返回空列表，
不影响主流程 —— 记忆是增强能力。

召回打分
--------
``score = 关键词交集 × 2 + 权重 + 时效加成``，低于 ``MEMORY_MIN_SCORE`` 的丢弃。
时效加成让「最近提到的事」优先于「很久以前的事」。

环境变量
--------
- ``MEMORY_ENABLED``      总开关，默认 ``true``
- ``MEMORY_TOP_K``        每次召回的事实条数，默认 ``5``
- ``MEMORY_MAX_FACTS``    单个 scope 保留的最大条数，默认 ``100``
- ``MEMORY_MIN_SCORE``    召回最低分，默认 ``2``
- ``MEMORY_DECAY_DAYS``   未命中的低价值记忆留存天数，默认 ``30``（0 = 不衰减）
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Literal, Optional, Sequence

from langchain_core.messages import HumanMessage
from loguru import logger
from pydantic import BaseModel, Field

from koiagent.infra.prompts import load_prompt
from koiagent.infra.text import content_hash, keyword_overlap, truncate
from koiagent.memory.store import MemoryRecord, MemoryStore

DEFAULT_EXTRACT_PROMPT = """你是客服系统的记忆抽取器。请从下面的对话片段中，抽取**值得长期记住**的信息。

只抽取这四类：
- fact       客观事实（例如：买家是做摄影的、已有一台同型号相机）
- preference 偏好（例如：偏爱国产、希望安静一点、喜欢简洁回复）
- objection  异议或顾虑（例如：觉得贵、担心续航、怕售后麻烦）
- commitment 已作出的承诺或约定（例如：已答应包邮、答应明天发货）

严格要求：
1. **只抽取稳定、可复用**的信息。寒暄、疑问句、一次性的操作（如"你好""在吗"）一律不要。
2. 每条不超过 30 字，用第三人称陈述。
3. 不要抽取商品本身的信息，只抽取**关于这个买家**的信息。
4. 如果没有任何值得记住的内容，返回空的 memories 列表 —— **宁可少抽，不要凑数**。
5. weight 表示重要性（0.5~2.0）：涉及价格、承诺、硬性要求的给高分。

【商品信息】
{item_desc}

【买家消息】
{user_msg}

【客服回复】
{reply}
"""


class ExtractedMemory(BaseModel):
    """一条被抽取出来的长期记忆。"""

    kind: Literal["fact", "preference", "objection", "commitment"] = Field(
        description="记忆类型"
    )
    content: str = Field(description="记忆内容，第三人称陈述，不超过 30 字")
    weight: float = Field(
        default=1.0, ge=0.5, le=2.0, description="重要性权重，价格/承诺/硬性要求给高分"
    )


class MemoryExtraction(BaseModel):
    """一次抽取的结果。"""

    memories: List[ExtractedMemory] = Field(default_factory=list, description="抽取到的记忆列表")


_NORMALIZE = re.compile(r"[\s，。,.！!？?；;：:'\"“”‘’()（）\[\]【】]+")


def fingerprint_of(content: str) -> str:
    """记忆指纹：归一化后的内容哈希，用于跨会话去重。

    归一化会去掉空白与常见标点，使「已答应包邮。」与「已答应包邮」视为同一条。
    """
    normalized = _NORMALIZE.sub("", content or "").strip().lower()
    return content_hash(normalized)


class LongTermMemory:
    """长期记忆的抽取、写入、召回与淘汰。"""

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        top_k: Optional[int] = None,
        max_facts: Optional[int] = None,
        min_score: Optional[int] = None,
        decay_days: Optional[float] = None,
        enabled: Optional[bool] = None,
    ):
        self.store = store or MemoryStore()
        self.top_k = int(top_k or os.getenv("MEMORY_TOP_K", "5"))
        self.max_facts = int(max_facts or os.getenv("MEMORY_MAX_FACTS", "100"))
        self.min_score = int(min_score or os.getenv("MEMORY_MIN_SCORE", "2"))
        self.decay_days = float(os.getenv("MEMORY_DECAY_DAYS", "30"))
        if enabled is None:
            enabled = os.getenv("MEMORY_ENABLED", "true").strip().lower() != "false"
        self.enabled = bool(enabled)

    # ------------------------------------------------------------------ #
    # 抽取
    # ------------------------------------------------------------------ #
    async def extract(
        self,
        llm: Any,
        user_msg: str,
        reply: str,
        item_desc: str = "",
    ) -> List[ExtractedMemory]:
        """从一轮对话中抽取长期记忆。失败返回空列表（不影响主流程）。"""
        if not self.enabled or llm is None:
            return []
        if not (user_msg or "").strip():
            return []

        prompt = load_prompt("memory_extract", DEFAULT_EXTRACT_PROMPT).format(
            item_desc=truncate(item_desc or "（无）", 400),
            user_msg=truncate(user_msg, 500),
            reply=truncate(reply or "（未回复）", 500),
        )

        try:
            result: MemoryExtraction = await llm.ainvoke([HumanMessage(content=prompt)])
            items = list(getattr(result, "memories", []) or [])
        except Exception as e:
            logger.warning(f"[memory] 长期记忆抽取失败（跳过本轮）: {e}")
            return []

        cleaned: List[ExtractedMemory] = []
        for item in items:
            content = (item.content or "").strip()
            if len(content) < 3:
                continue
            cleaned.append(
                ExtractedMemory(kind=item.kind, content=content, weight=float(item.weight or 1.0))
            )
        if cleaned:
            logger.info(f"[memory] 抽取到 {len(cleaned)} 条长期记忆")
        return cleaned

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    def remember(
        self, scope: str, scope_id: str, items: Sequence[ExtractedMemory]
    ) -> int:
        """写入记忆，返回**新增**条数（重复项只提升权重）。

        同一批次内也会去重，避免模型重复输出同一条。
        """
        if not self.enabled or not scope_id or not items:
            return 0

        added = 0
        seen: set = set()
        for item in items:
            fingerprint = fingerprint_of(item.content)
            if not fingerprint or fingerprint in seen:
                continue
            seen.add(fingerprint)
            if self.store.upsert_memory(
                scope=scope,
                scope_id=scope_id,
                fingerprint=fingerprint,
                content=item.content,
                kind=item.kind,
                weight=item.weight,
            ):
                added += 1

        if added:
            self.prune(scope, scope_id)
        return added

    # ------------------------------------------------------------------ #
    # 召回
    # ------------------------------------------------------------------ #
    @staticmethod
    def _score(query: str, record: MemoryRecord, now: float) -> float:
        """综合评分：关键词相关度 + 权重 + 时效加成。"""
        score = keyword_overlap(query, record.content) * 2.0 if query else 0.0
        score += record.weight

        age_days = max(0.0, (now - record.created_at) / 86400.0)
        score += max(0.0, 1.0 - age_days / 30.0)  # 30 天内线性衰减的时效加成

        if record.hits:
            score += min(1.0, record.hits * 0.1)  # 被反复召回说明它确实有用
        return score

    def recall(
        self, scope: str, scope_id: str, query: str = "", k: Optional[int] = None
    ) -> List[MemoryRecord]:
        """按相关度召回记忆（同时更新命中计数与最近使用时间）。"""
        if not self.enabled or not scope_id:
            return []

        candidates = self.store.list_memories(scope, scope_id, limit=max(self.max_facts, 50))
        if not candidates:
            return []

        limit = k or self.top_k
        now = time.time()
        scored = [(self._score(query, record, now), record) for record in candidates]

        if query:
            # 有查询时按阈值过滤，避免把无关记忆塞进上下文
            scored = [item for item in scored if item[0] >= self.min_score]
        scored.sort(key=lambda item: item[0], reverse=True)

        selected = [record for _, record in scored[:limit]]
        if selected:
            self.store.touch_memories(
                [record.fingerprint for record in selected], scope, scope_id
            )
        return selected

    # ------------------------------------------------------------------ #
    # 维护
    # ------------------------------------------------------------------ #
    def prune(self, scope: str, scope_id: str) -> int:
        """按 TTL + 容量淘汰记忆，返回删除条数。"""
        if not self.enabled or not scope_id:
            return 0
        removed = self.store.prune_memories(
            scope, scope_id, max_facts=self.max_facts, decay_days=self.decay_days
        )
        if removed:
            logger.info(f"[memory] 已淘汰 {removed} 条过期/低价值记忆 ({scope}:{scope_id})")
        return removed

    def forget(self, scope: str, scope_id: str, contents: Sequence[str]) -> int:
        """按内容删除指定记忆。"""
        fingerprints = [fingerprint_of(text) for text in contents if text]
        return self.store.delete_memories(scope, scope_id, fingerprints)

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "total": self.store.count_memories(),
            "top_k": self.top_k,
            "max_facts": self.max_facts,
        }
