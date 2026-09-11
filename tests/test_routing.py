"""意图路由与安全护栏测试。"""
import pytest

from koiagent.agent.graph import KoiReplyBot

# 规则应当直接命中的用例（不依赖 LLM）
RULE_CASES = [
    # price
    ("这个能便宜点吗", "price"),
    ("最低多少钱", "price"),
    ("能少50吗", "price"),
    ("180元可以吗", "price"),
    # tech
    ("这个参数是多少", "tech"),
    ("都有哪些规格", "tech"),
    ("是什么型号的", "tech"),
    ("支持连接蓝牙吗", "tech"),
    ("这款和那款比怎么样", "tech"),
]


@pytest.mark.parametrize("message,expect", RULE_CASES)
def test_rule_based_intent_hits(message, expect):
    assert KoiReplyBot._rule_based_intent(message) == expect


@pytest.mark.parametrize("message", ["在吗", "什么时候发货", "支持7天无理由吗", "你用的什么模型"])
def test_rule_based_intent_miss_returns_none(message):
    """未命中规则的语句应返回 None，交由 LLM 兜底。"""
    assert KoiReplyBot._rule_based_intent(message) is None


def test_safe_filter_blocks_offsite_contact():
    assert KoiReplyBot._safe_filter("加我微信详聊") == "[安全提醒]请通过平台沟通"
    assert KoiReplyBot._safe_filter("支付宝转账更便宜") == "[安全提醒]请通过平台沟通"


def test_safe_filter_passes_normal_reply():
    assert KoiReplyBot._safe_filter("今天可以发货") == "今天可以发货"


def test_extract_bargain_count():
    context = [{"role": "system", "content": "议价次数: 3"}]
    assert KoiReplyBot._extract_bargain_count(context) == 3
    assert KoiReplyBot._extract_bargain_count([]) == 0
    assert KoiReplyBot._extract_bargain_count(None) == 0


def test_format_context_filters_non_dialogue_roles():
    context = [
        {"role": "user", "content": "在吗"},
        {"role": "system", "content": "议价次数: 1"},
        {"role": "assistant", "content": "在的"},
    ]
    formatted = KoiReplyBot._format_context(context)
    assert "user: 在吗" in formatted
    assert "assistant: 在的" in formatted
    assert "议价次数" not in formatted
