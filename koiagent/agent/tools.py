"""KoiAgent 工具集（Function Calling）。

这些工具会被绑定到 Agent 的 LLM 上，由模型**自主决定**是否调用、调用哪个，
以及传入什么参数。这是本项目从「单轮 prompt 直出」升级为真正 Agent 的关键。

其中 ``search_knowledge_base`` 走 RAG 检索层（``koiagent.rag``）：
配置 ``EMBEDDING_MODEL`` 时使用**向量检索**，否则自动降级为**关键词检索**。

三个工具各自的记忆行为
----------------------
======================  ==================  ==========================================
工具                     是否进工具记忆       说明
======================  ==================  ==========================================
``get_bargain_policy``   ✅ 缓存 + 记录       纯函数，只依赖议价轮次；策略文件支持热更新
``search_knowledge_base``✅ 缓存 + 记录       知识库支持热更新；变更后自动让缓存失效
``get_current_time``     ❌ 仅记录            **时间每次都在变，缓存会返回错误答案**
======================  ==================  ==========================================

> 缓存是**白名单制**（见 ``koiagent.memory.tool_memory.CACHEABLE_TOOLS``）：
> 新工具默认不缓存，必须显式声明可缓存。这样不会因为「忘了排除某个工具」
> 而悄悄引入错误结果。
"""
from __future__ import annotations

from datetime import datetime

from langchain_core.tools import tool

from koiagent.agent.bargain import get_bargain_policy_store
from koiagent.memory.manager import get_memory_manager
from koiagent.memory.tool_memory import ToolMemory
from koiagent.rag.knowledge import get_knowledge_base


def _tool_memory() -> ToolMemory:
    """获取工具记忆。

    **从记忆管理器单例取**，而不是单独调 ``get_tool_memory()`` —— 否则会存在两份
    ``ToolMemory`` 实例（管理器一份、工具一份），缓存各算各的、命中率统计也对不上。
    """
    return get_memory_manager().tools


@tool
def get_bargain_policy(bargain_count: int) -> str:
    """查询给定议价轮次下应采用的阶梯让步策略与让价上限。

    当买家就价格进行砍价、讨价还价时，调用本工具获取当前轮次应当坚持的让价尺度。

    Args:
        bargain_count: 当前会话已发生的议价轮次，从 1 开始计数。
    """
    args = {"bargain_count": bargain_count}
    store = get_bargain_policy_store()

    def compute() -> str:
        # 策略文件热更新（按间隔节流，开销仅一次 stat）
        store.maybe_reload()
        return store.advise(bargain_count)

    return _tool_memory().invoke("get_bargain_policy", args, compute)


@tool
def search_knowledge_base(query: str) -> str:
    """在商品/技术知识库中进行检索（RAG），用于回答产品参数、规格、兼容性、售后等问题。

    当买家询问产品的技术细节、使用方法、发货售后等信息时，调用本工具检索资料后作答。

    Args:
        query: 需要检索的关键词或买家原问题。
    """
    args = {"query": query}
    kb = get_knowledge_base()

    # ⚠️ 热更新检查必须在缓存查询**之前**：否则知识库已经改了，
    # 但缓存里还留着旧检索结果，买家会拿到过期信息。
    if kb.maybe_reload():
        get_memory_manager().on_knowledge_updated()

    def compute() -> str:
        if kb.size == 0:
            return (
                "本地知识库为空（knowledge/ 目录下暂无文档），"
                "请依据商品描述谨慎作答，不确定时如实说明。"
            )
        hits = kb.search(query)
        if not hits:
            return "知识库中未检索到相关内容，请基于商品信息作答；如无把握，请如实告知买家。"
        return "\n\n".join(f"[{source}] {text}" for source, text in hits)

    return _tool_memory().invoke("search_knowledge_base", args, compute)


@tool
def get_current_time() -> str:
    """获取当前服务器时间（Asia/Shanghai）。用于回答发货时间、时效等与时间相关的问题。"""
    # 结果不进缓存（时间每次都变，缓存会回答错误的时间），
    # 但**仍然记录调用**，便于回放与统计
    result = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _tool_memory().record("get_current_time", {}, result, cache_hit=False)
    return result


# 暴露给 Agent 的工具列表
ALL_TOOLS = [get_bargain_policy, search_knowledge_base, get_current_time]
