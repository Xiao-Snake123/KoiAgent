"""KoiAgent 可观测性：全链路追踪 / 指标聚合 / Token 与成本统计。

生产级 Agent 不能只有零散日志。本模块解决三个问题：

1. **可追踪**：每次 Agent 运行的完整轨迹（意图、路由方式、工具调用序列、推理步数、
   耗时、Token 用量、估算成本、异常）逐行写入 ``logs/traces.jsonl``，便于回放与排障。
2. **可度量**：累计运行数、错误数、意图分布、工具调用分布、平均/P95 耗时、
   Token 与成本总量，聚合落盘 ``logs/metrics.json``。
3. **零侵入采集**：通过 LangChain Callback 机制自动从底层 LLM 响应中提取用量，
   同时兼容新版 ``usage_metadata`` 与旧版 ``llm_output.token_usage``。

可选接入 **Langfuse**：配置 ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``
且已安装 ``langfuse`` 时，``Tracer.callbacks()`` 会自动附带其回调处理器。

环境变量
--------
- ``TRACE_ENABLED``          是否开启追踪，默认 ``true``
- ``TRACE_DIR``              输出目录，默认 ``logs``
- ``METRICS_SAMPLE_LIMIT``   延迟样本上限（用于 P95），默认 ``1000``
- ``COST_INPUT_PER_1K``      输入单价覆盖（元 / 1K tokens）
- ``COST_OUTPUT_PER_1K``     输出单价覆盖（元 / 1K tokens）
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from loguru import logger

# 示例价格表：元 / 1K tokens，**请以官方计费为准**，可用 COST_* 环境变量覆盖
_DEFAULT_PRICE_TABLE: Dict[str, tuple] = {
    "qwen-max": (0.0024, 0.0096),
    "qwen-plus": (0.0008, 0.0020),
    "qwen-turbo": (0.0003, 0.0006),
    "deepseek-chat": (0.0010, 0.0020),
    "gpt-4o-mini": (0.0011, 0.0043),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """估算调用成本（元）。优先使用 ``COST_*`` 环境变量，否则查内置示例价格表。"""
    env_in = os.getenv("COST_INPUT_PER_1K")
    env_out = os.getenv("COST_OUTPUT_PER_1K")
    if env_in or env_out:
        price_in = float(env_in or 0)
        price_out = float(env_out or 0)
    else:
        price_in, price_out = _DEFAULT_PRICE_TABLE.get((model or "").strip(), (0.0, 0.0))
    return prompt_tokens / 1000 * price_in + completion_tokens / 1000 * price_out


def extract_usage(response: Any) -> tuple:
    """从 LangChain ``LLMResult`` 中提取 ``(prompt_tokens, completion_tokens, model)``。"""
    prompt = completion = 0
    model = ""

    for gen_list in getattr(response, "generations", None) or []:
        for gen in gen_list:
            message = getattr(gen, "message", None)
            if message is None:
                continue
            usage = getattr(message, "usage_metadata", None)
            if isinstance(usage, dict):
                prompt += int(usage.get("input_tokens") or 0)
                completion += int(usage.get("output_tokens") or 0)
            meta = getattr(message, "response_metadata", None)
            if isinstance(meta, dict) and not model:
                model = meta.get("model_name") or meta.get("model") or ""

    llm_output = getattr(response, "llm_output", None) or {}
    if not (prompt or completion):
        usage = llm_output.get("token_usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
    if not model:
        model = llm_output.get("model_name") or ""

    return prompt, completion, model


class UsageCollector(BaseCallbackHandler):
    """采集单次 Agent 运行期间所有 LLM 调用的 Token 用量。"""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.models: List[str] = []

    def on_llm_start(self, serialized: Any, prompts: List[str], **kwargs: Any) -> None:
        self.llm_calls += 1

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        prompt, completion, model = extract_usage(response)
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        if model and model not in self.models:
            self.models.append(model)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def primary_model(self) -> str:
        return self.models[0] if self.models else ""


@dataclass
class TraceRecord:
    """一次 Agent 运行的完整轨迹。"""

    ts: str
    thread_id: str
    intent: str = ""
    routing: str = ""          # rule | llm | error
    guard_action: str = ""     # allow | block
    tools: List[str] = field(default_factory=list)
    steps: int = 0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    llm_calls: int = 0
    reply_len: int = 0
    reflections: int = 0
    degraded: bool = False
    error: str = ""


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


class Tracer:
    """轨迹与指标收集器（线程安全）。"""

    def __init__(self, trace_dir: Optional[str] = None, enabled: Optional[bool] = None):
        if enabled is None:
            enabled = os.getenv("TRACE_ENABLED", "true").strip().lower() != "false"
        self.enabled = enabled
        self.trace_dir = trace_dir or os.getenv("TRACE_DIR", "logs")
        self.trace_path = os.path.join(self.trace_dir, "traces.jsonl")
        self.metrics_path = os.path.join(self.trace_dir, "metrics.json")
        self.sample_limit = int(os.getenv("METRICS_SAMPLE_LIMIT", "1000"))

        self._lock = threading.Lock()
        self._langfuse_handler: Optional[Any] = None
        self._metrics: Dict[str, Any] = self._load_metrics()

    # ------------------------------ 回调 ------------------------------ #
    def callbacks(self, collector: UsageCollector) -> List[Any]:
        """返回挂载到本次运行的 callbacks 列表（采集器 + 可选 Langfuse）。"""
        handlers: List[Any] = [collector]
        langfuse = self._get_langfuse_handler()
        if langfuse is not None:
            handlers.append(langfuse)
        return handlers

    def _get_langfuse_handler(self) -> Optional[Any]:
        if self._langfuse_handler is not None:
            return self._langfuse_handler
        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
            return None
        for module_path in ("langfuse.langchain", "langfuse.callback"):
            try:
                module = __import__(module_path, fromlist=["CallbackHandler"])
                self._langfuse_handler = module.CallbackHandler()
                logger.info(f"Langfuse 追踪已启用（{module_path}）")
                return self._langfuse_handler
            except Exception:
                continue
        logger.debug("检测到 LANGFUSE_* 配置但未安装 langfuse，已跳过上报")
        return None

    # ------------------------------ 记录 ------------------------------ #
    def record(self, rec: TraceRecord) -> None:
        if not self.enabled:
            return
        with self._lock:
            try:
                self._append_trace(rec)
                self._update_metrics(rec)
                self._write_metrics()
            except Exception as e:  # 可观测性失败绝不能影响主流程
                logger.warning(f"写入追踪数据失败: {e}")

    def _append_trace(self, rec: TraceRecord) -> None:
        os.makedirs(self.trace_dir, exist_ok=True)
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def _load_metrics(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data.setdefault("latency_samples", [])
                data.setdefault("guard_blocked", 0)
                data.setdefault("degraded", 0)
                data.setdefault("critic_rejections", 0)
                return data
        except Exception as e:
            logger.warning(f"读取指标文件失败，将重新统计: {e}")
        return {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "total_runs": 0,
            "errors": 0,
            "degraded": 0,
            "critic_rejections": 0,
            "guard_blocked": 0,
            "intents": {},
            "routing": {},
            "tool_calls": {},
            "llm_calls": 0,
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "cost": 0.0,
            "latency_samples": [],
            "last_updated": "",
        }

    def _update_metrics(self, rec: TraceRecord) -> None:
        m = self._metrics
        m["total_runs"] += 1
        if rec.error:
            m["errors"] += 1
        if rec.guard_action == "block":
            m["guard_blocked"] = m.get("guard_blocked", 0) + 1
        if rec.degraded:
            m["degraded"] = m.get("degraded", 0) + 1
        if rec.reflections > 0:
            m["critic_rejections"] = m.get("critic_rejections", 0) + 1
        m["intents"][rec.intent or "unknown"] = m["intents"].get(rec.intent or "unknown", 0) + 1
        m["routing"][rec.routing or "unknown"] = m["routing"].get(rec.routing or "unknown", 0) + 1
        for name in rec.tools:
            m["tool_calls"][name] = m["tool_calls"].get(name, 0) + 1
        m["llm_calls"] += rec.llm_calls
        m["tokens"]["prompt"] += rec.prompt_tokens
        m["tokens"]["completion"] += rec.completion_tokens
        m["tokens"]["total"] += rec.total_tokens
        m["cost"] = round(m["cost"] + rec.cost, 6)

        samples: List[float] = m["latency_samples"]
        samples.append(round(rec.latency_ms, 2))
        if len(samples) > self.sample_limit:
            del samples[: len(samples) - self.sample_limit]

        m["last_updated"] = datetime.now().isoformat(timespec="seconds")

    def _write_metrics(self) -> None:
        os.makedirs(self.trace_dir, exist_ok=True)
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(self._metrics, f, ensure_ascii=False, indent=2)

    # ------------------------------ 展示 ------------------------------ #
    def snapshot(self) -> Dict[str, Any]:
        """返回指标快照（含派生指标 avg/p95）。"""
        with self._lock:
            m = json.loads(json.dumps(self._metrics))  # 深拷贝，避免外部修改
        samples = m.get("latency_samples", [])
        runs = m.get("total_runs", 0) or 1
        m["avg_latency_ms"] = round(sum(samples) / len(samples), 2) if samples else 0.0
        m["p95_latency_ms"] = round(_percentile(samples, 95), 2)
        m["error_rate"] = round(m.get("errors", 0) / runs, 4)
        return m

    def format_report(self) -> str:
        """生成可读的指标报告（用于日志或 CLI 输出）。"""
        s = self.snapshot()
        lines = [
            "──────── KoiAgent 运行指标 ────────",
            f"总运行次数   : {s.get('total_runs', 0)}",
            f"错误次数     : {s.get('errors', 0)} (错误率 {s.get('error_rate', 0):.2%})",
            f"平均耗时     : {s.get('avg_latency_ms', 0)} ms",
            f"P95 耗时     : {s.get('p95_latency_ms', 0)} ms",
            f"LLM 调用次数 : {s.get('llm_calls', 0)}",
            f"Token 用量   : 输入 {s['tokens']['prompt']} / 输出 {s['tokens']['completion']} / 合计 {s['tokens']['total']}",
            f"估算成本     : ¥{s.get('cost', 0):.4f}",
            f"降级兜底     : {s.get('degraded', 0)}",
            f"审核驳回     : {s.get('critic_rejections', 0)}",
            f"拦截注入     : {s.get('guard_blocked', 0)}",
            f"意图分布     : {s.get('intents', {})}",
            f"路由分布     : {s.get('routing', {})}",
            f"工具调用     : {s.get('tool_calls', {})}",
            "────────────────────────────────────",
        ]
        return "\n".join(lines)
