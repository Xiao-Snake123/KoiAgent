"""KoiAgent 工具集（Function Calling）。

这些工具会被绑定到 Agent 的 LLM 上，由模型**自主决定**是否调用、调用哪个，
以及传入什么参数。这是本项目从「单轮 prompt 直出」升级为真正 Agent 的关键。

其中 ``search_knowledge_base`` 走 RAG 检索层（``koiagent.rag``）：
配置 ``EMBEDDING_MODEL`` 时使用**向量检索**，否则自动降级为**关键词检索**。
"""
from __future__ import annotations

from datetime import datetime

from langchain_core.tools import tool

from koiagent.rag.knowledge import get_knowledge_base


@tool
def get_bargain_policy(bargain_count: int) -> str:
    """查询给定议价轮次下应采用的阶梯让步策略与让价上限。

    当买家就价格进行砍价、讨价还价时，调用本工具获取当前轮次应当坚持的让价尺度。

    Args:
        bargain_count: 当前会话已发生的议价轮次，从 1 开始计数。
    """
    tiers = [
        (1, "首轮让价：幅度不超过标价的 5%，优先用赠品/包邮替代直接降价。"),
        (2, "次轮让价：累计让价不超过标价的 10%，强调成色与稀缺性。"),
        (3, "三轮让价：累计让价不超过标价的 15%，可给出一次性『一口价』方案。"),
    ]
    guidance = "已达让价上限，礼貌坚持底价，或引导买家关注其他在售商品。"
    for threshold, text in tiers:
        if bargain_count <= threshold:
            guidance = text
            break
    return f"当前议价轮次={bargain_count}；让步策略：{guidance}"


@tool
def search_knowledge_base(query: str) -> str:
    """在商品/技术知识库中进行检索（RAG），用于回答产品参数、规格、兼容性、售后等问题。

    当买家询问产品的技术细节、使用方法、发货售后等信息时，调用本工具检索资料后作答。

    Args:
        query: 需要检索的关键词或买家原问题。
    """
    kb = get_knowledge_base()
    if kb.size == 0:
        return "本地知识库为空（knowledge/ 目录下暂无文档），请依据商品描述谨慎作答，不确定时如实说明。"

    hits = kb.search(query)
    if not hits:
        return "知识库中未检索到相关内容，请基于商品信息作答；如无把握，请如实告知买家。"

    return "\n\n".join(f"[{source}] {text}" for source, text in hits)


@tool
def get_current_time() -> str:
    """获取当前服务器时间（Asia/Shanghai）。用于回答发货时间、时效等与时间相关的问题。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 暴露给 Agent 的工具列表
ALL_TOOLS = [get_bargain_policy, search_knowledge_base, get_current_time]
