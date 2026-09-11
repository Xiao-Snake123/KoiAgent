"""KoiAgent 输入侧 Prompt 注入防护（Guardrails）。

针对客服场景的「纵深防御」四层策略：

1. **归一化 normalize**：Unicode NFKC 归一 + 剥离零宽/控制字符 + 超长截断，
   防止同形字与隐形字符绕过检测。
2. **规则检测 inspect**：加权匹配已知攻击模式 —— 指令覆盖、角色扮演越狱、
   系统提示词探测、分隔符注入、输出格式劫持、编码载荷等。
3. **风险分级**：累计得分达到阈值（``GUARD_BLOCK_SCORE``，默认 5）即拦截。
4. **上下文加固 harden**：剥离伪造的角色/系统标记，配合系统提示声明
   「用户输入为不可信数据」，从结构上降低注入成功率。

环境变量
--------
- ``GUARD_ENABLED``      是否启用输入防护，默认 ``true``
- ``GUARD_BLOCK_SCORE``  拦截阈值，默认 ``5``
- ``GUARD_MAX_INPUT``    最大输入长度（超出截断），默认 ``1000``
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List

# --------------------------------------------------------------------------- #
# 归一化 / 加固
# --------------------------------------------------------------------------- #
# 零宽字符、双向控制符、C0 控制符（保留 \t \n \r）
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]")

# 伪造的对话/系统标记
_ROLE_MARKER = re.compile(r"(<\|im_(?:start|end)\|>|<<\/?SYS>>|\[/?INST\])", re.IGNORECASE)
# 行首伪造的角色名（如 "system:" / "assistant："）
_FAKE_ROLE_LINE = re.compile(r"(?im)^\s*(?:system|assistant|user)\s*[:：]\s*")


def normalize(text: str, max_len: int | None = None) -> str:
    """归一化输入：NFKC 归一 + 剥离隐形字符 + 截断超长。"""
    if not text:
        return ""
    limit = int(os.getenv("GUARD_MAX_INPUT", "1000")) if max_len is None else max_len
    text = unicodedata.normalize("NFKC", str(text))
    text = _INVISIBLE.sub("", text)
    text = text.strip()
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def harden(text: str) -> str:
    """加固处理：剥离伪造的角色与系统标记，供 LLM 使用。"""
    if not text:
        return ""
    text = _ROLE_MARKER.sub("", text)
    text = _FAKE_ROLE_LINE.sub("", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# 攻击模式（加权）
# --------------------------------------------------------------------------- #
# (正则, 权重, 分类标签)
_PATTERNS: List[tuple] = [
    # —— 指令覆盖 ——
    (r"(忽略|无视|忘记|丢弃).{0,10}(以上|之前|上面|先前|所有|全部).{0,6}(指令|提示|规则|设定|要求|内容)",
     5, "指令覆盖"),
    (r"(ignore|disregard|forget)\s+(all\s+)?(previous|above|prior|earlier)\s+(instructions?|prompts?|rules?)",
     5, "指令覆盖"),
    # —— 角色扮演越狱 ——
    (r"(从现在起|从现在开始|现在开始|接下来).{0,6}(你是|你将扮演|你要扮演|你扮演)",
     4, "角色扮演越狱"),
    (r"(假装|扮演|装作).{0,8}(你是|一位|一个)",
     3, "角色扮演越狱"),
    (r"you\s+are\s+now\s+(a|an|the)?",
     3, "角色扮演越狱"),
    (r"(开发者模式|越狱模式|jailbreak|\bDAN\b)",
     5, "越狱模式"),
    (r"(不受限制|无限制|没有限制|不受约束|无需遵守).{0,8}(AI|人工智能|助手|机器人|模型|回答)",
     3, "越狱模式"),
    # —— 系统提示词探测 ——
    (r"(输出|告诉我|复述|打印|展示|重复|泄露).{0,8}(你的|系统)?(提示词|prompt|指令|设定|规则|配置)",
     5, "提示词探测"),
    (r"(the\s+)?(full|complete|original|entire)\s+(instructions?|system\s+prompt|rules)",
     5, "提示词探测"),
    (r"(repeat|print|show|reveal|leak).{0,12}(your\s+)?(system\s+prompt|instructions?|rules)",
     5, "提示词探测"),
    (r"(output\s+as-is|without\s+any\s+rewriting|repeat\s+verbatim|verbatim\s+output)",
     5, "提示词探测"),
    # —— 输出格式劫持 ——
    (r"(按|按照|以).{0,8}(格式|要求|风格).{0,6}(输出|回复|返回|重写)",
     3, "格式劫持"),
    # —— 分隔符 / 特殊标记注入 ——
    (r"(<\|im_(?:start|end)\|>|<<\/?SYS>>|\[/?INST\])",
     5, "特殊标记注入"),
    (r"(?im)^\s*(system|assistant)\s*[:：]",
     3, "伪造角色标记"),
    # —— 身份套取 ——
    (r"(你用的?|你是|你使用).{0,6}(什么|哪个|啥).{0,4}(模型|大模型|AI|人工智能)",
     3, "身份套取"),
    # —— 疑似编码载荷 ——
    (r"[A-Za-z0-9+/]{80,}={0,2}",
     3, "疑似编码载荷"),
]


@dataclass
class GuardVerdict:
    """防护判定结果。"""

    action: str = "allow"          # allow | block
    score: int = 0
    risk: str = "low"              # low | medium | high
    reasons: List[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.action == "block"


def inspect(text: str) -> GuardVerdict:
    """对（已归一化的）文本做规则检测并给出风险判定。"""
    if not text:
        return GuardVerdict()

    score = 0
    reasons: List[str] = []
    for pattern, weight, label in _PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            score += weight
            if label not in reasons:
                reasons.append(label)

    threshold = int(os.getenv("GUARD_BLOCK_SCORE", "5"))
    action = "block" if score >= threshold else "allow"
    risk = "high" if action == "block" else ("medium" if score > 0 else "low")
    return GuardVerdict(action=action, score=score, risk=risk, reasons=reasons)


def guard_enabled() -> bool:
    return os.getenv("GUARD_ENABLED", "true").strip().lower() != "false"


def screen(raw_text: str) -> tuple:
    """一站式入口：返回 ``(安全文本, 判定结果)``。

    - 安全文本 = ``harden(normalize(raw))``，供 LLM 使用
    - 判定结果基于 ``normalize(raw)``（保留标记以便检出）
    """
    normalized = normalize(raw_text)
    cleaned = harden(normalized)
    verdict = inspect(normalized) if guard_enabled() else GuardVerdict()
    return cleaned, verdict
