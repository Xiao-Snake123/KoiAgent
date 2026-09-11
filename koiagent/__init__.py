"""KoiAgent —— 面向电商客服场景的 AI 值守 Agent。

分层结构
--------
- ``koiagent.agent``    LangGraph 编排层：状态图、Function Calling 工具、输入侧注入防护
- ``koiagent.rag``      检索增强层：文档切分 / 向量化 / 双模式检索
- ``koiagent.platform`` 平台通信层：闲鱼 HTTP 接口与私有协议工具
- ``koiagent.infra``    基础设施层：可观测性与可靠性组件
- ``koiagent.storage``  存储层：SQLite 业务数据持久化
- ``koiagent.mcp``      MCP Server：把能力暴露给任意 MCP 客户端
- ``koiagent.ops``      运维脚本：容器健康检查
- ``koiagent.app``      应用装配：WebSocket 长连、心跳、Token 刷新、人工接管
- ``koiagent.config``   配置加载与校验
"""
from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
