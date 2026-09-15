"""KoiAgent 记忆系统：短期 / 长期 / 用户画像 / 工具记忆。

四类记忆的分工
--------------
- :mod:`koiagent.memory.short_term`  **短期记忆**：对话窗口 + 增量摘要压缩
- :mod:`koiagent.memory.long_term`   **长期记忆**：跨会话事实的抽取、召回与衰减淘汰
- :mod:`koiagent.memory.profile`     **用户画像**：结构化属性（预算/意向/议价风格…）
- :mod:`koiagent.memory.tool_memory` **工具记忆**：会话级工具结果缓存 + 调用记录

对外只需要 :class:`koiagent.memory.manager.MemoryManager` 一个入口。

设计原则
--------
1. **记忆是增强能力，不是关键路径**：任何一步失败都降级（返回旧值 / 空值），
   绝不让记忆系统的问题阻断买家收到回复。
2. **读同步、写异步**：召回在生成前必须完成；写回不影响本轮回复，因此放后台。
3. **可观测**：记忆条数、画像命中、工具缓存命中率都进 Trace 与 metrics。
"""
from __future__ import annotations

from koiagent.memory.long_term import (
    ExtractedMemory,
    LongTermMemory,
    MemoryExtraction,
)
from koiagent.memory.manager import (
    MemoryContext,
    MemoryManager,
    TurnInsight,
    get_memory_manager,
    set_memory_manager,
)
from koiagent.memory.profile import ProfilePatch, ProfileStore, UserProfile
from koiagent.memory.short_term import ShortTermMemory
from koiagent.memory.store import MemoryRecord, MemoryStore
from koiagent.memory.tool_memory import (
    CACHEABLE_TOOLS,
    ToolMemory,
    bind_chat_id,
    get_current_chat_id,
    get_tool_memory,
)

__all__ = [
    # 门面
    "MemoryManager",
    "get_memory_manager",
    "set_memory_manager",
    "MemoryContext",
    "TurnInsight",
    # 子模块
    "ShortTermMemory",
    "LongTermMemory",
    "ExtractedMemory",
    "MemoryExtraction",
    "ProfileStore",
    "UserProfile",
    "ProfilePatch",
    "ToolMemory",
    "get_tool_memory",
    "CACHEABLE_TOOLS",
    "bind_chat_id",
    "get_current_chat_id",
    # 存储
    "MemoryStore",
    "MemoryRecord",
]
