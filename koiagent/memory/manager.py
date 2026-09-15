"""记忆系统门面：把短期/长期/画像/工具四类记忆组合成统一的读写接口。

四类记忆的分工
--------------

=================  ================================  ==========  ==================
类型                回答什么问题                       生命周期     注入方式
=================  ================================  ==========  ==================
**短期记忆**        「这段对话刚才说了什么」            单会话       摘要 + 原文窗口
**长期记忆**        「这个买家历来是什么情况」          跨会话       按查询相关度召回
**用户画像**        「这个买家是什么样的人」            跨会话       每轮全量注入
**工具记忆**        「这个工具刚才是怎么答的」          单会话       （不进提示词）
=================  ================================  ==========  ==================

三次 LLM 调用是怎么压成一次的
------------------------------
朴素实现需要三次抽取调用（摘要 / 长期记忆 / 画像）。本模块做了两处优化：

1. **长期记忆与画像合并为一次抽取**（:class:`TurnInsight`），因为两者读的是同一段
   对话、用的是同一类提示词，合并后质量损失很小而成本减半。
2. **按输入长度预过滤**：买家只说「好的」「在吗」时直接跳过抽取（``MEMORY_MIN_INPUT_LEN``）。
   客服从大量寒暄里省下的调用数是可观的。

摘要调用是独立的（输入是历史消息而不是本轮对话），且**增量触发**，不在每轮发生。

写回为什么放在图之外
--------------------
读取（召回）必须在生成之前完成，所以是图里的 ``recall`` 节点。
写入（抽取+落库）**不影响本轮回复**，因此放在 ``agenerate_reply`` 里用
``asyncio.create_task`` 后台执行 —— 买家不必为记忆抽取多等一次 LLM 往返。

环境变量
--------
- ``MEMORY_ENABLED``        记忆系统总开关，默认 ``true``
- ``MEMORY_MIN_INPUT_LEN``  短于该长度的用户输入跳过抽取，默认 ``10``
- ``MEMORY_ASYNC_WRITE``    是否后台异步写回，默认 ``true``
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from loguru import logger
from pydantic import BaseModel, Field

from koiagent.infra.prompts import load_prompt
from koiagent.infra.text import truncate
from koiagent.memory.long_term import ExtractedMemory, LongTermMemory
from koiagent.memory.profile import ProfilePatch, ProfileStore, UserProfile
from koiagent.memory.short_term import ShortTermMemory
from koiagent.memory.store import MemoryRecord, MemoryStore
from koiagent.memory.tool_memory import ToolMemory, get_tool_memory

DEFAULT_INSIGHT_PROMPT = """你是客服系统的记忆分析器。请从下面这一轮对话中，**同时**完成两件事：
① 抽取值得长期记住的关键信息；② 更新对这位买家的画像判断。

═══ 一、长期记忆（memories）═══
只抽取这四类，且必须是**稳定、可复用**的信息：
- fact       客观事实（例如：买家是做摄影的、已有一台同型号相机）
- preference 偏好（例如：偏爱国产、希望安静一点、喜欢简洁回复）
- objection  异议或顾虑（例如：觉得贵、担心续航、怕售后麻烦）
- commitment 已作出的承诺或约定（例如：已答应包邮、答应明天发货）

严格要求：
- 不要抽取商品本身的信息，只抽取**关于这个买家**的信息
- 寒暄与一次性内容（"你好""在吗""好的"）一律不要
- 每条不超过 30 字，第三人称陈述
- weight 表示重要性（0.5~2.0）：涉及价格、承诺、硬性要求给高分
- **宁可少抽，不要凑数**；没有就返回空列表

═══ 二、用户画像（profile）═══
只填**本轮对话中明确体现出来**的字段，没提到的**必须留空**，不要猜测：
- budget_min / budget_max：买家明确说出的预算数字（元）
- intent_level：browse=随便看看；compare=比价中；ready=强意向（如问发货/下单/库存）
- bargain_style：direct=直接砍价；indirect=委婉试探；none=完全不还价
- communication_style：concise=明确表示希望简短；detailed=希望详细说明
- interests：买家明确关注的产品点（最多 3 个）
- constraints：买家的硬性要求或禁忌，如"不要黑色""必须今天发货"（最多 3 个）
- tags：概括性标签，如"价格敏感""急用"（最多 3 个）

【商品信息】
{item_desc}

【买家消息】
{user_msg}

【客服回复】
{reply}
"""


class TurnInsight(BaseModel):
    """一次抽取同时得到长期记忆与画像增量（合并调用以节省成本）。"""

    memories: List[ExtractedMemory] = Field(
        default_factory=list, description="值得长期记住的信息"
    )
    profile: ProfilePatch = Field(
        default_factory=ProfilePatch, description="本轮对话体现出的画像信息，未提及的字段留空"
    )


@dataclass
class MemoryContext:
    """一次召回的结果，用于注入 system prompt。"""

    profile: Optional[UserProfile] = None
    facts: List[MemoryRecord] = field(default_factory=list)
    scopes: List[str] = field(default_factory=list)

    @property
    def has_profile(self) -> bool:
        return self.profile is not None and not self.profile.is_empty()

    def is_empty(self) -> bool:
        return not self.has_profile and not self.facts

    def render(self) -> str:
        """渲染成可注入提示词的中文文本。"""
        parts: List[str] = []
        if self.has_profile and self.profile is not None:
            text = self.profile.render()
            if text:
                parts.append(f"【买家画像】{text}")
        if self.facts:
            lines = "\n".join(f"- {record.content}" for record in self.facts)
            parts.append(f"【关于这位买家你已经知道的信息】\n{lines}")
        return "\n".join(parts)

    def to_trace(self) -> Dict[str, Any]:
        """给可观测性用的摘要（不含记忆正文，避免 trace 膨胀）。"""
        return {
            "memory_facts": len(self.facts),
            "profile_hit": self.has_profile,
        }


class MemoryManager:
    """记忆系统统一入口。"""

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        short_term: Optional[ShortTermMemory] = None,
        enabled: Optional[bool] = None,
        tool_memory: Optional[ToolMemory] = None,
    ):
        if enabled is None:
            enabled = os.getenv("MEMORY_ENABLED", "true").strip().lower() != "false"
        self.enabled = bool(enabled)

        self.store = store or MemoryStore()
        self.short_term = short_term or ShortTermMemory()
        self.long_term = LongTermMemory(store=self.store, enabled=self.enabled)
        self.profiles = ProfileStore(store=self.store, enabled=self.enabled)
        self.tools: ToolMemory = tool_memory or get_tool_memory(store=self.store)

        self.min_input_len = int(os.getenv("MEMORY_MIN_INPUT_LEN", "10"))
        self.async_write = os.getenv("MEMORY_ASYNC_WRITE", "true").strip().lower() != "false"

        # LLM 由 KoiReplyBot 注入（复用其客户端，避免重复建连）
        self.summarizer: Optional[Any] = None
        self.extractor: Optional[Any] = None

    # ------------------------------------------------------------------ #
    # 依赖注入
    # ------------------------------------------------------------------ #
    def configure_llm(
        self,
        summarizer: Optional[Any] = None,
        extractor: Optional[Any] = None,
    ) -> None:
        """注入摘要模型与抽取模型。

        - ``summarizer``：普通文本输出（把对话压成摘要）
        - ``extractor``：结构化输出 :class:`TurnInsight`
        """
        self.summarizer = summarizer
        self.extractor = extractor

    @property
    def ready(self) -> bool:
        return self.enabled

    # ------------------------------------------------------------------ #
    # 读：召回
    # ------------------------------------------------------------------ #
    def recall(
        self,
        chat_id: Optional[str] = None,
        user_id: Optional[str] = None,
        query: str = "",
    ) -> MemoryContext:
        """召回与当前对话相关的记忆（同步、纯本地查询，开销为一次 SQLite 读）。"""
        context = MemoryContext()
        if not self.enabled:
            return context

        if user_id:
            context.profile = self.profiles.get(user_id)

        seen: set = set()
        facts: List[MemoryRecord] = []

        # 用户维度优先（跨会话），再补会话维度（本次会话特有）
        for scope, scope_id in (("user", user_id), ("chat", chat_id)):
            if not scope_id:
                continue
            for record in self.long_term.recall(scope, scope_id, query=query):
                key = record.fingerprint or record.content
                if key in seen:
                    continue
                seen.add(key)
                facts.append(record)
            context.scopes.append(f"{scope}:{scope_id}")

        context.facts = facts[: self.long_term.top_k]
        return context

    # ------------------------------------------------------------------ #
    # 写：摘要（短期记忆）
    # ------------------------------------------------------------------ #
    async def maybe_summarize(
        self,
        messages: Any,
        summary: str,
        summarized_count: int,
    ) -> Dict[str, Any]:
        """按需压缩短期记忆，返回要写回图状态的字段（无需更新时返回空字典）。"""
        if not self.enabled or self.summarizer is None:
            return {}

        total = len(messages or [])
        if not self.short_term.should_summarize(total, summarized_count):
            return {}

        start, end = self.short_term.pending_range(total, summarized_count)
        if end <= start:
            return {}

        new_summary = await self.short_term.summarize(
            self.summarizer, summary, list(messages)[start:end]
        )
        return {"summary": new_summary, "summarized_count": end}

    # ------------------------------------------------------------------ #
    # 写：长期记忆 + 画像
    # ------------------------------------------------------------------ #
    @staticmethod
    def _should_extract(user_msg: str, min_len: int) -> bool:
        """廉价预过滤：太短的输入（寒暄）不值得付一次抽取调用。"""
        text = (user_msg or "").strip()
        return len(text) >= min_len

    async def extract_insight(
        self, user_msg: str, reply: str, item_desc: str = ""
    ) -> Optional[TurnInsight]:
        """抽取一轮对话的记忆与画像增量。失败返回 None。"""
        if not self.enabled or self.extractor is None:
            return None

        prompt = load_prompt("memory_prompt", DEFAULT_INSIGHT_PROMPT).format(
            item_desc=truncate(item_desc or "（无）", 400),
            user_msg=truncate(user_msg, 500),
            reply=truncate(reply or "（未回复）", 500),
        )
        try:
            return await self.extractor.ainvoke([HumanMessage(content=prompt)])
        except Exception as e:
            logger.warning(f"[memory] 记忆抽取失败（跳过本轮）: {e}")
            return None

    async def remember(
        self,
        chat_id: Optional[str],
        user_id: Optional[str],
        user_msg: str,
        reply: str,
        item_desc: str = "",
        bargain_delta: int = 0,
    ) -> Dict[str, Any]:
        """一轮对话结束后写入记忆。返回统计信息（供日志与测试断言）。"""
        result: Dict[str, Any] = {"added": 0, "profile_fields": [], "skipped": ""}
        if not self.enabled:
            result["skipped"] = "disabled"
            return result

        # ---- 计数类字段始终更新（不依赖 LLM）----
        if user_id:
            profile = self.profiles.get(user_id)
            profile.touch()
            if bargain_delta:
                profile.record_bargain(bargain_delta)
            # 若本轮不做抽取，仍然落盘计数
            self.profiles.save(profile)

        if not self._should_extract(user_msg, self.min_input_len):
            result["skipped"] = "input_too_short"
            return result

        insight = await self.extract_insight(user_msg, reply, item_desc)
        if insight is None:
            result["skipped"] = "extraction_failed"
            return result

        # ---- 长期记忆 ----
        scope, scope_id = ("user", user_id) if user_id else ("chat", chat_id)
        if scope_id and insight.memories:
            result["added"] = self.long_term.remember(scope, scope_id, insight.memories)

        # ---- 用户画像 ----
        if user_id:
            result["profile_fields"] = self.profiles.merge(user_id, insight.profile)

        if result["added"] or result["profile_fields"]:
            logger.info(
                f"[memory] 已更新 chat={chat_id} user={user_id} "
                f"新增记忆={result['added']} 画像字段={result['profile_fields']}"
            )
        return result

    # ------------------------------------------------------------------ #
    # 维护
    # ------------------------------------------------------------------ #
    def on_knowledge_updated(self) -> int:
        """知识库热更新后调用：使检索类工具的缓存失效，避免返回旧结果。"""
        return self.tools.invalidate_tool("search_knowledge_base")

    def invalidate_session(self, chat_id: Optional[str] = None) -> int:
        return self.tools.invalidate_session(chat_id)

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "store": self.store.stats(),
            "tool_memory": self.tools.stats(),
            "long_term": self.long_term.stats(),
            "short_term": {
                "max_messages": self.short_term.max_messages,
                "trigger": self.short_term.trigger,
                "enabled": self.short_term.enabled,
            },
        }


_MANAGER_SINGLETON: Optional[MemoryManager] = None
_SINGLETON_LOCK = threading.Lock()


def get_memory_manager(refresh: bool = False) -> MemoryManager:
    """获取（惰性构建的）记忆管理器单例。"""
    global _MANAGER_SINGLETON
    with _SINGLETON_LOCK:
        if _MANAGER_SINGLETON is None or refresh:
            _MANAGER_SINGLETON = MemoryManager()
        return _MANAGER_SINGLETON


def set_memory_manager(manager: Optional[MemoryManager]) -> Optional[MemoryManager]:
    """替换全局记忆管理器，返回被替换的旧实例（供调用方恢复）。

    为什么需要这个接口：工具函数（``koiagent.agent.tools``）不知道 ``chat_id`` 之外的
    上下文，必须通过单例拿到 :class:`ToolMemory`。如果没有可注入的替换点，
    测试就只能去操作真实的 ``data/memory.db``。

    这样「工具用的缓存」与「管理器持有的缓存」保证是**同一个实例**，
    不会出现两份缓存各自计数、命中率统计对不上的情况。
    """
    global _MANAGER_SINGLETON
    with _SINGLETON_LOCK:
        previous = _MANAGER_SINGLETON
        _MANAGER_SINGLETON = manager
        return previous
