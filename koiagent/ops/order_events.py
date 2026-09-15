"""订单事件：把「平台推来的订单状态变化」变成有明确扩展点的钩子。

原先 ``app.py`` 里这段是这样的：

    # 判断是否为订单消息,需要自行编写付款后的逻辑
    if message['3']['redReminder'] == '等待买家付款':
        logger.info(f'等待买家 {user_url} 付款')
        return

三种订单状态**只打日志就 return**，没有任何业务动作。问题不在于"没写"，而在于
**没有扩展点** —— 想加「成交后引导评价」的人不知道代码该写在哪一层。

本模块把这件事变成结构化的：

1. :func:`resolve_order_event` 把平台文案解析成 :class:`OrderEventKind` 枚举，
   文案匹配集中在一处（平台改文案时只改这里）。
2. :class:`OrderEventHandler` 提供**有默认行为的**钩子方法 —— 默认实现不是空的，
   而是做「记录 + 写进 Agent 记忆 + 维护画像计数」这些确定该做的事。
3. 业务方要加动作（催付、引导评价、通知卖家）时，**继承并覆盖对应方法**即可，
   不需要改 ``app.py``。

默认行为为什么不做「自动催付」「自动发货」这类动作
--------------------------------------------------
这些动作涉及**真实资金与履约**，且各店铺流程差异极大。默认实现替用户做决定是
不负责任的 —— 所以默认只做**幂等、无副作用的记录**，把有副作用的动作留给使用者显式实现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

from loguru import logger


class OrderEventKind(str, Enum):
    """订单状态类型。"""

    AWAITING_PAYMENT = "awaiting_payment"    # 等待买家付款
    CLOSED = "closed"                        # 交易关闭
    AWAITING_SHIPMENT = "awaiting_shipment"  # 等待卖家发货（付款成功）

    @property
    def label(self) -> str:
        return {
            OrderEventKind.AWAITING_PAYMENT: "等待买家付款",
            OrderEventKind.CLOSED: "交易关闭",
            OrderEventKind.AWAITING_SHIPMENT: "等待卖家发货",
        }[self]


# 平台文案 → 事件类型。平台若调整文案，**只需改这一处**。
RED_REMINDER_MAP: Dict[str, OrderEventKind] = {
    "等待买家付款": OrderEventKind.AWAITING_PAYMENT,
    "交易关闭": OrderEventKind.CLOSED,
    "等待卖家发货": OrderEventKind.AWAITING_SHIPMENT,
}


@dataclass
class OrderEvent:
    """一次订单状态变化。"""

    kind: OrderEventKind
    user_id: str = ""
    chat_id: str = ""
    item_id: str = ""
    raw: str = ""
    handled: bool = False
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def user_url(self) -> str:
        return f"https://www.goofish.com/personal?userId={self.user_id}" if self.user_id else ""


def resolve_order_event(
    red_reminder: str,
    user_id: str = "",
    chat_id: str = "",
    item_id: str = "",
) -> Optional[OrderEvent]:
    """把 ``redReminder`` 文案解析为订单事件；非订单消息返回 ``None``。"""
    kind = RED_REMINDER_MAP.get((red_reminder or "").strip())
    if kind is None:
        return None
    return OrderEvent(
        kind=kind, user_id=user_id, chat_id=chat_id, item_id=item_id, raw=red_reminder.strip()
    )


class OrderEventHandler:
    """订单事件处理器（默认实现：记录 + 写记忆 + 维护画像）。

    子类覆盖 ``on_*`` 方法即可加入业务动作，**记得调用 ``super().on_xxx(event)``**
    以保留默认的记录行为。
    """

    def __init__(
        self,
        context_manager: Optional[Any] = None,
        bot: Optional[Any] = None,
        memory: Optional[Any] = None,
    ):
        self.context_manager = context_manager
        self.bot = bot
        self.memory = memory

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _remember(self, event: OrderEvent, text: str) -> None:
        """把订单状态写进 Agent 记忆，让后续对话知道发生了什么。"""
        if self.bot is None or not event.chat_id:
            return
        try:
            self.bot.record_message(event.chat_id, "assistant", text)
        except Exception as e:
            logger.debug(f"写入订单事件记忆失败: {e}")

    # ------------------------------------------------------------------ #
    # 钩子
    # ------------------------------------------------------------------ #
    def on_awaiting_payment(self, event: OrderEvent) -> None:
        """买家已下单、等待付款。

        默认：记录 + 写进 Agent 记忆（后续买家问「我刚拍下了」时机器人有上下文）。
        **不做自动催付** —— 催付的时机与话术强依赖店铺策略。
        """
        logger.info(f"订单事件[等待付款] 买家={event.user_url or event.user_id}")
        self._remember(event, "【订单状态】买家已拍下商品，等待付款。")
        event.handled = True

    def on_closed(self, event: OrderEvent) -> None:
        """交易已关闭。

        默认：记录 + 写进 Agent 记忆 + **清理工具缓存**。
        清理缓存是必要的 —— 之前查到的价格/库存信息此时可能已经失效，
        继续用缓存会导致机器人答错。
        """
        logger.info(f"订单事件[交易关闭] 买家={event.user_url or event.user_id}")
        self._remember(event, "【订单状态】本次交易已关闭。")
        if self.memory is not None and event.chat_id:
            try:
                removed = self.memory.invalidate_session(event.chat_id)
                event.notes["cache_invalidated"] = removed
            except Exception as e:
                logger.debug(f"清理工具缓存失败: {e}")
        event.handled = True

    def on_awaiting_shipment(self, event: OrderEvent) -> None:
        """买家已付款、等待卖家发货（**成交**）。

        默认：记录 + 写进 Agent 记忆 + **累加画像成交数**（后续可识别为老客户）。
        **不做自动发货提醒** —— 发货是卖家的履约行为。
        """
        logger.info(f"订单事件[等待发货/已成交] 买家={event.user_url or event.user_id}")
        self._remember(event, "【订单状态】买家已付款，交易达成，等待卖家发货。")
        if self.memory is not None and event.user_id:
            try:
                self.memory.profiles.bump_deal(event.user_id)
                event.notes["deal_recorded"] = True
            except Exception as e:
                logger.debug(f"更新成交计数失败: {e}")
        event.handled = True

    # ------------------------------------------------------------------ #
    # 分发
    # ------------------------------------------------------------------ #
    def handle(self, event: OrderEvent) -> bool:
        """按事件类型分发；返回是否处理成功（异常会被吞掉，不影响消息主循环）。"""
        dispatcher: Dict[OrderEventKind, Callable[[OrderEvent], None]] = {
            OrderEventKind.AWAITING_PAYMENT: self.on_awaiting_payment,
            OrderEventKind.CLOSED: self.on_closed,
            OrderEventKind.AWAITING_SHIPMENT: self.on_awaiting_shipment,
        }
        handler = dispatcher.get(event.kind)
        if handler is None:
            return False
        try:
            handler(event)
            return True
        except Exception as e:
            logger.warning(f"处理订单事件失败 {event.kind}: {e}")
            return False
