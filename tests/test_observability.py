"""可观测性模块测试：Token 提取、成本估算、Trace 与指标落盘。"""
import json
from types import SimpleNamespace

from koiagent.infra.observability import TraceRecord, Tracer, estimate_cost, extract_usage


def _llm_result(prompt, completion, model="qwen-max"):
    """构造一个带 usage_metadata 的伪 LLMResult。"""
    message = SimpleNamespace(
        usage_metadata={"input_tokens": prompt, "output_tokens": completion},
        response_metadata={"model_name": model},
    )
    return SimpleNamespace(generations=[[SimpleNamespace(message=message)]], llm_output={})


def test_extract_usage_from_usage_metadata():
    assert extract_usage(_llm_result(10, 5)) == (10, 5, "qwen-max")


def test_extract_usage_falls_back_to_llm_output():
    response = SimpleNamespace(
        generations=[[]],
        llm_output={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}, "model_name": "m"},
    )
    assert extract_usage(response) == (7, 3, "m")


def test_extract_usage_handles_empty_response():
    assert extract_usage(SimpleNamespace(generations=[[]], llm_output={})) == (0, 0, "")


def test_estimate_cost_with_env_override(monkeypatch):
    monkeypatch.setenv("COST_INPUT_PER_1K", "1")
    monkeypatch.setenv("COST_OUTPUT_PER_1K", "2")
    assert estimate_cost("any-model", 1000, 1000) == 3.0


def test_estimate_cost_unknown_model_is_zero(monkeypatch):
    monkeypatch.delenv("COST_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("COST_OUTPUT_PER_1K", raising=False)
    assert estimate_cost("unknown-model", 1000, 1000) == 0.0


def test_tracer_writes_trace_and_aggregates_metrics(tmp_path):
    tracer = Tracer(trace_dir=str(tmp_path), enabled=True)
    tracer.record(
        TraceRecord(
            ts="2026-01-01T00:00:00",
            thread_id="chat-1",
            intent="price",
            routing="rule",
            tools=["get_bargain_policy"],
            steps=2,
            latency_ms=120.5,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost=0.001,
            llm_calls=1,
            reply_len=8,
        )
    )

    lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["intent"] == "price"

    snapshot = tracer.snapshot()
    assert snapshot["total_runs"] == 1
    assert snapshot["intents"]["price"] == 1
    assert snapshot["routing"]["rule"] == 1
    assert snapshot["tool_calls"]["get_bargain_policy"] == 1
    assert snapshot["tokens"]["total"] == 15
    assert snapshot["avg_latency_ms"] == 120.5

    report = tracer.format_report()
    assert "KoiAgent 运行指标" in report
    assert "总运行次数" in report
    assert (tmp_path / "metrics.json").exists()


def test_tracer_disabled_writes_nothing(tmp_path):
    tracer = Tracer(trace_dir=str(tmp_path), enabled=False)
    tracer.record(TraceRecord(ts="t", thread_id="c1"))
    assert not (tmp_path / "traces.jsonl").exists()


def test_tracer_records_error_run(tmp_path):
    tracer = Tracer(trace_dir=str(tmp_path), enabled=True)
    tracer.record(TraceRecord(ts="t", thread_id="c1", error="TimeoutError: boom"))
    snapshot = tracer.snapshot()
    assert snapshot["errors"] == 1
    assert snapshot["error_rate"] == 1.0
    assert snapshot["routing"]["unknown"] == 1


def test_tracer_counts_degraded_runs(tmp_path):
    tracer = Tracer(trace_dir=str(tmp_path), enabled=True)
    tracer.record(TraceRecord(ts="t", thread_id="c1", degraded=True, error="circuit_open"))
    snapshot = tracer.snapshot()
    assert snapshot["degraded"] == 1
    assert snapshot["errors"] == 1
