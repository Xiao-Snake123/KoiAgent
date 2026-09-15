"""议价策略：从配置文件加载、支持热更新的阶梯让步策略。

为什么单独抽一层
----------------
议价策略是**确定性业务规则**（首轮让价 ≤5%、次轮累计 ≤10% …），而且**不同商品类别
需要不同策略**（3C 数码让价空间小、服装清仓可以让更多）。原先它硬编码在
``koiagent.agent.tools.get_bargain_policy`` 的函数体里，改一个数字就要改代码、重发版。

现在它是一份 **JSON 配置**，支持：

1. **文件覆盖**：``BARGAIN_POLICY_PATH``（默认 ``config/bargain_policy.json``）；
   文件缺失或格式非法时**自动回退内置默认值**，不影响启动。
2. **热更新**：按 mtime 检测文件变化，改动后**无需重启**即可生效。
3. **可测试**：``advise()`` 是纯函数式接口，输入轮次输出策略文本，便于断言。

环境变量
--------
- ``BARGAIN_POLICY_PATH``            策略文件路径，默认 ``config/bargain_policy.json``
- ``BARGAIN_POLICY_RELOAD_INTERVAL`` 热更新检查间隔（秒），默认 ``30``；``0`` 表示关闭
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from loguru import logger

DEFAULT_POLICY_PATH = "config/bargain_policy.json"

# 内置默认策略：文件缺失/损坏时的兜底，保证功能始终可用
DEFAULT_POLICY: Dict[str, Any] = {
    "version": 1,
    "description": "默认阶梯议价策略：三轮内逐步让步，之后守底价",
    "tiers": [
        {
            "max_round": 1,
            "max_discount_pct": 5,
            "guidance": "首轮让价：幅度不超过标价的 5%，优先用赠品/包邮替代直接降价。",
        },
        {
            "max_round": 2,
            "max_discount_pct": 10,
            "guidance": "次轮让价：累计让价不超过标价的 10%，强调成色与稀缺性。",
        },
        {
            "max_round": 3,
            "max_discount_pct": 15,
            "guidance": "三轮让价：累计让价不超过标价的 15%，可给出一次性『一口价』方案。",
        },
    ],
    "floor_guidance": "已达让价上限，礼貌坚持底价，或引导买家关注其他在售商品。",
}


def _validate(payload: Any) -> Optional[Dict[str, Any]]:
    """校验策略结构；非法时返回 None（由调用方回退默认值）。"""
    if not isinstance(payload, dict):
        return None
    tiers = payload.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        return None

    normalized: List[Dict[str, Any]] = []
    for index, tier in enumerate(tiers):
        if not isinstance(tier, dict):
            return None
        try:
            max_round = int(tier["max_round"])
            discount = float(tier["max_discount_pct"])
        except (KeyError, TypeError, ValueError):
            return None
        if max_round < 1 or discount < 0 or discount > 100:
            return None
        normalized.append(
            {
                "max_round": max_round,
                "max_discount_pct": discount,
                "guidance": str(tier.get("guidance") or f"第 {max_round} 轮议价策略未描述").strip(),
            }
        )

    # 按轮次升序，保证查找逻辑（取第一个 max_round >= n 的档位）正确
    normalized.sort(key=lambda item: item["max_round"])
    return {
        "version": payload.get("version", 1),
        "description": str(payload.get("description") or ""),
        "tiers": normalized,
        "floor_guidance": str(
            payload.get("floor_guidance") or DEFAULT_POLICY["floor_guidance"]
        ).strip(),
    }


class BargainPolicy:
    """阶梯议价策略的加载 / 热更新 / 查询。"""

    def __init__(self, path: Optional[str] = None, reload_interval: Optional[float] = None):
        self.path = path or os.getenv("BARGAIN_POLICY_PATH", DEFAULT_POLICY_PATH)
        if reload_interval is None:
            reload_interval = float(os.getenv("BARGAIN_POLICY_RELOAD_INTERVAL", "30"))
        self.reload_interval = float(reload_interval)

        self._lock = threading.RLock()
        self._policy: Dict[str, Any] = DEFAULT_POLICY
        self._source = "builtin"
        self._mtime: float = 0.0
        self._last_check: float = 0.0

        self.reload(force=True)

    # ---------------- 加载 ----------------
    def reload(self, force: bool = False) -> bool:
        """从文件重新加载策略，返回是否发生了变化。"""
        with self._lock:
            if not os.path.isfile(self.path):
                if force and self._source != "builtin":
                    self._policy, self._source, self._mtime = DEFAULT_POLICY, "builtin", 0.0
                    return True
                self._source = "builtin" if self._source == "builtin" else "builtin"
                return False

            try:
                mtime = os.path.getmtime(self.path)
            except OSError:
                return False

            if not force and mtime == self._mtime:
                return False

            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    payload = _validate(json.load(f))
            except Exception as e:
                logger.warning(f"议价策略文件解析失败，沿用当前策略: {self.path} ({e})")
                return False

            if payload is None:
                logger.warning(
                    f"议价策略文件结构非法（需含非空 tiers，且每项有 max_round / max_discount_pct），"
                    f"沿用当前策略: {self.path}"
                )
                return False

            self._policy = payload
            self._source = self.path
            self._mtime = mtime
            logger.info(
                f"议价策略已加载: {self.path}（{len(payload['tiers'])} 档，"
                f"最大让步 {payload['tiers'][-1]['max_discount_pct']}%）"
            )
            return True

    def maybe_reload(self) -> bool:
        """按间隔节流地检查文件是否变化（供每次工具调用前调用，开销为一次 stat）。"""
        if self.reload_interval <= 0:
            return False
        now = time.time()
        with self._lock:
            if now - self._last_check < self.reload_interval:
                return False
            self._last_check = now
        # 锁外执行 IO，避免长时间持锁
        return self.reload(force=False)

    # ---------------- 查询 ----------------
    def advise(self, bargain_count: int) -> str:
        """返回指定议价轮次下应采用的让步策略文本。"""
        with self._lock:
            tiers = self._policy["tiers"]
            floor = self._policy["floor_guidance"]
            source = self._source

        try:
            round_no = max(1, int(bargain_count))
        except (TypeError, ValueError):
            round_no = 1

        guidance = floor
        limit: Optional[float] = None
        for tier in tiers:
            if round_no <= tier["max_round"]:
                guidance = tier["guidance"]
                limit = tier["max_discount_pct"]
                break

        if limit is None:
            return f"当前议价轮次={round_no}；让步策略：{guidance}"
        return f"当前议价轮次={round_no}；累计让价上限={limit:g}%；让步策略：{guidance}"

    def describe(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "source": self._source,
                "tiers": len(self._policy["tiers"]),
                "max_discount_pct": self._policy["tiers"][-1]["max_discount_pct"],
            }

    @property
    def source(self) -> str:
        with self._lock:
            return self._source


_POLICY_SINGLETON: Optional[BargainPolicy] = None
_SINGLETON_LOCK = threading.Lock()


def get_bargain_policy_store(refresh: bool = False) -> BargainPolicy:
    """获取（惰性构建的）议价策略单例。"""
    global _POLICY_SINGLETON
    with _SINGLETON_LOCK:
        if _POLICY_SINGLETON is None or refresh:
            _POLICY_SINGLETON = BargainPolicy()
        return _POLICY_SINGLETON
