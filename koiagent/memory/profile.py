"""用户画像：结构化属性 + 增量合并。

和「长期记忆」的区别
--------------------
- **长期记忆**是**非结构化的事实列表**（「买家是做摄影的」「之前问过续航」），
  靠检索按需召回，条数不限。
- **用户画像**是**固定 schema 的结构化属性**（预算区间、意向等级、议价风格…），
  每轮**全量注入**提示词，让模型一眼知道「这个买家是什么人」。

两者互补：画像提供「快速定位」，长期记忆提供「细节回溯」。

合并策略（这是本模块的核心规则）
--------------------------------
- **标量字段**（预算、意向、风格）：**新值覆盖旧值**（非空才覆盖）。
  理由：买家需求会变化，最近一次表达最可信。
- **列表字段**（关注点、约束、标签）：**取并集**并按上限截断。
  理由：这些是累积属性，不会因为新一轮没提及就失效。
- **计数字段**（接待次数、议价次数、成交数）：**单调累加**，不减。

环境变量
--------
- ``PROFILE_ENABLED``      是否启用画像，默认 ``true``
- ``PROFILE_MAX_LIST_ITEMS`` 列表字段上限，默认 ``10``
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field

from koiagent.memory.store import MemoryStore

INTENT_LEVELS = ("browse", "compare", "ready", "unknown")
BARGAIN_STYLES = ("direct", "indirect", "none", "unknown")
COMMUNICATION_STYLES = ("concise", "detailed", "unknown")

_INTENT_LABEL = {
    "browse": "随便看看",
    "compare": "比价中",
    "ready": "强意向",
    "unknown": "未知",
}
_BARGAIN_LABEL = {
    "direct": "直接砍价",
    "indirect": "委婉试探",
    "none": "不还价",
    "unknown": "未知",
}
_STYLE_LABEL = {
    "concise": "喜欢简短",
    "detailed": "喜欢详细",
    "unknown": "未知",
}


class ProfilePatch(BaseModel):
    """由 LLM 从对话中抽取的画像增量（**全部可选**，没提到的字段留空）。"""

    budget_min: Optional[float] = Field(
        default=None, description="买家可接受的最低预算（元），未提及则留空"
    )
    budget_max: Optional[float] = Field(
        default=None, description="买家可接受的最高预算（元），未提及则留空"
    )
    intent_level: Optional[Literal["browse", "compare", "ready", "unknown"]] = Field(
        default=None, description="购买意向：browse=随便看看；compare=比价中；ready=强意向；unknown=无法判断"
    )
    bargain_style: Optional[Literal["direct", "indirect", "none", "unknown"]] = Field(
        default=None, description="议价风格：direct=直接砍价；indirect=委婉试探；none=不还价"
    )
    communication_style: Optional[Literal["concise", "detailed", "unknown"]] = Field(
        default=None, description="沟通偏好：concise=喜欢简短回复；detailed=希望说明详细"
    )
    interests: List[str] = Field(
        default_factory=list, description="买家明确关注的产品点，如『续航』『降噪』『颜色』，最多 3 个"
    )
    constraints: List[str] = Field(
        default_factory=list, description="买家的硬性约束或禁忌，如『不要黑色』『必须今天发货』，最多 3 个"
    )
    tags: List[str] = Field(
        default_factory=list, description="概括性标签，如『价格敏感』『老客户』『急用』，最多 3 个"
    )


@dataclass
class UserProfile:
    """一个买家的结构化画像。"""

    user_id: str
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    intent_level: str = "unknown"
    bargain_style: str = "unknown"
    communication_style: str = "unknown"
    interests: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    interaction_count: int = 0
    total_bargain_count: int = 0
    deals_closed: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # ------------------------------------------------------------------ #
    # 序列化
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "UserProfile":
        """从字典还原；**忽略未知字段**，保证旧数据在新版本下仍可读取。"""
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        data = {k: v for k, v in (payload or {}).items() if k in allowed}
        data.setdefault("user_id", "")
        for name in ("interests", "constraints", "tags"):
            value = data.get(name)
            data[name] = list(value) if isinstance(value, list) else []
        return cls(**data)

    # ------------------------------------------------------------------ #
    # 合并
    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge_list(old: List[str], new: List[str], limit: int) -> List[str]:
        """列表取并集，保持出现顺序，去重并截断。"""
        merged: List[str] = []
        for item in list(old or []) + list(new or []):
            text = str(item).strip()
            if text and text not in merged:
                merged.append(text)
        return merged[:limit]

    def merge(self, patch: Optional[ProfilePatch], limit: int = 10) -> List[str]:
        """把 LLM 抽取的增量合并进画像，返回**实际发生变化的字段名**。"""
        if patch is None:
            return []

        changed: List[str] = []

        for name in ("budget_min", "budget_max"):
            value = getattr(patch, name, None)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            if getattr(self, name) != value:
                setattr(self, name, value)
                changed.append(name)

        # 预算区间自洽：出现「下限 > 上限」时丢弃下限（上限更具约束力）
        if self.budget_min is not None and self.budget_max is not None:
            if self.budget_min > self.budget_max:
                self.budget_min = None
                changed.append("budget_min")

        for name, allowed in (
            ("intent_level", INTENT_LEVELS),
            ("bargain_style", BARGAIN_STYLES),
            ("communication_style", COMMUNICATION_STYLES),
        ):
            value = getattr(patch, name, None)
            if value and value in allowed and value != "unknown" and getattr(self, name) != value:
                setattr(self, name, value)
                changed.append(name)

        for name in ("interests", "constraints", "tags"):
            merged = self._merge_list(getattr(self, name), getattr(patch, name, []), limit)
            if merged != getattr(self, name):
                setattr(self, name, merged)
                changed.append(name)

        return changed

    # ------------------------------------------------------------------ #
    # 计数
    # ------------------------------------------------------------------ #
    def touch(self) -> None:
        """记录一次接待。"""
        self.interaction_count += 1
        self.last_seen = time.time()

    def record_bargain(self, count: int = 1) -> None:
        self.total_bargain_count += count

    def record_deal(self) -> None:
        self.deals_closed += 1

    # ------------------------------------------------------------------ #
    # 渲染
    # ------------------------------------------------------------------ #
    def is_empty(self) -> bool:
        """是否还没有任何有效信息（用于避免把空画像注入提示词）。"""
        return not any(
            [
                self.budget_min,
                self.budget_max,
                self.intent_level != "unknown",
                self.bargain_style != "unknown",
                self.communication_style != "unknown",
                self.interests,
                self.constraints,
                self.tags,
            ]
        )

    def render(self) -> str:
        """渲染成一段可注入 system prompt 的中文描述。"""
        if self.is_empty() and self.interaction_count <= 1:
            return ""

        parts: List[str] = []
        if self.interaction_count > 1:
            parts.append(f"第 {self.interaction_count} 次接待")
        if self.intent_level != "unknown":
            parts.append(f"意向：{_INTENT_LABEL.get(self.intent_level, self.intent_level)}")
        if self.bargain_style != "unknown":
            parts.append(f"议价风格：{_BARGAIN_LABEL.get(self.bargain_style, self.bargain_style)}")
        if self.communication_style != "unknown":
            parts.append(f"沟通偏好：{_STYLE_LABEL.get(self.communication_style, self.communication_style)}")

        if self.budget_min and self.budget_max:
            parts.append(f"预算：¥{self.budget_min:g}~¥{self.budget_max:g}")
        elif self.budget_max:
            parts.append(f"预算上限：¥{self.budget_max:g}")
        elif self.budget_min:
            parts.append(f"预算下限：¥{self.budget_min:g}")

        if self.interests:
            parts.append(f"关注点：{'、'.join(self.interests)}")
        if self.constraints:
            parts.append(f"硬性要求：{'、'.join(self.constraints)}")
        if self.tags:
            parts.append(f"标签：{'、'.join(self.tags)}")
        if self.total_bargain_count:
            parts.append(f"历史议价 {self.total_bargain_count} 次")
        if self.deals_closed:
            parts.append(f"历史成交 {self.deals_closed} 次（老客户）")

        return "｜".join(parts)


class ProfileStore:
    """用户画像的读写与增量更新。"""

    def __init__(self, store: Optional[MemoryStore] = None, enabled: Optional[bool] = None):
        self.store = store or MemoryStore()
        if enabled is None:
            enabled = os.getenv("PROFILE_ENABLED", "true").strip().lower() != "false"
        self.enabled = bool(enabled)
        self.max_items = int(os.getenv("PROFILE_MAX_LIST_ITEMS", "10"))

    def get(self, user_id: str) -> UserProfile:
        if not user_id:
            return UserProfile(user_id="")
        payload = self.store.load_profile(user_id)
        if not payload:
            return UserProfile(user_id=user_id)
        return UserProfile.from_dict(payload)

    def save(self, profile: UserProfile) -> None:
        if not self.enabled or not profile.user_id:
            return
        self.store.save_profile(profile.user_id, profile.to_dict())

    def merge(self, user_id: str, patch: Optional[ProfilePatch]) -> List[str]:
        """合并增量并落盘，返回发生变化的字段名。"""
        if not self.enabled or not user_id:
            return []
        profile = self.get(user_id)
        changed = profile.merge(patch, limit=self.max_items)
        if changed:
            self.save(profile)
            logger.info(f"[memory] 用户画像已更新 user={user_id} 字段={changed}")
        return changed

    def touch(self, user_id: str) -> UserProfile:
        """记录一次接待并落盘。"""
        profile = self.get(user_id)
        profile.touch()
        self.save(profile)
        return profile

    def bump_bargain(self, user_id: str, count: int = 1) -> None:
        if not self.enabled or not user_id:
            return
        profile = self.get(user_id)
        profile.record_bargain(count)
        self.save(profile)

    def bump_deal(self, user_id: str) -> None:
        if not self.enabled or not user_id:
            return
        profile = self.get(user_id)
        profile.record_deal()
        self.save(profile)
