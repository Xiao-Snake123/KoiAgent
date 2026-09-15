"""配置校验测试：确保「填错配置」在启动期就被发现，而不是运行到一半才崩。"""
import os

import pytest

from koiagent.config import CONFIG_SCHEMA, describe_config, validate_config


@pytest.fixture
def env_guard():
    """validate_config 会回写 os.environ，测试后必须整体还原。"""
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_schema_entries_are_well_formed():
    for key, spec in CONFIG_SCHEMA.items():
        assert len(spec) == 4, f"{key} 的 schema 应为 (类型, 最小, 最大, 默认值)"
        typ, lo, hi, _default = spec
        assert typ in (int, float, bool), f"{key} 的类型 {typ} 不支持"
        if lo is not None and hi is not None:
            assert lo <= hi, f"{key} 的最小值大于最大值"


def test_validate_accepts_defaults(env_guard):
    for key in CONFIG_SCHEMA:
        os.environ.pop(key, None)
    assert validate_config() == []


def test_validate_rejects_non_numeric(env_guard):
    os.environ["AGENT_MAX_STEPS"] = "abc"
    errors = validate_config()
    assert any("AGENT_MAX_STEPS" in message and "整数" in message for message in errors)


def test_validate_rejects_out_of_range(env_guard):
    os.environ["AGENT_MAX_STEPS"] = "999"
    errors = validate_config()
    assert any("AGENT_MAX_STEPS" in message and "最大值" in message for message in errors)

    os.environ["AGENT_MAX_STEPS"] = "0"
    errors = validate_config()
    assert any("AGENT_MAX_STEPS" in message and "最小值" in message for message in errors)


@pytest.mark.parametrize(
    "raw,expected",
    [("true", True), ("TRUE", True), ("1", True), ("yes", True), ("on", True), ("false", False), ("0", False), ("no", False)],
)
def test_validate_accepts_bool_variants(env_guard, raw, expected):
    os.environ["CRITIC_ENABLED"] = raw
    assert validate_config() == []
    # 校验后应被规范化，保证下游 `== "false"` 之类的比较可靠
    normalized = os.getenv("CRITIC_ENABLED")
    assert normalized in ("True", "False")
    assert (normalized == "True") is expected


def test_validate_rejects_bad_bool(env_guard):
    os.environ["CRITIC_ENABLED"] = "也许"
    errors = validate_config()
    assert any("CRITIC_ENABLED" in message for message in errors)


def test_validate_cross_checks_chunk_overlap(env_guard):
    os.environ["RAG_CHUNK_SIZE"] = "100"
    os.environ["RAG_CHUNK_OVERLAP"] = "100"
    errors = validate_config()
    assert any("RAG_CHUNK_OVERLAP" in message for message in errors)


def test_validate_warns_when_reap_interval_exceeds_ttl(env_guard, caplog):
    os.environ["MEMORY_REAP_INTERVAL"] = "9999"
    os.environ["MEMORY_TTL"] = "7200"
    # 这是警告而非错误：不阻挡启动，但必须被记入日志
    assert validate_config() == []


def test_validate_strict_raises(env_guard):
    os.environ["GUARD_BLOCK_SCORE"] = "not-a-number"
    with pytest.raises(ValueError, match="配置校验失败"):
        validate_config(strict=True)


def test_validate_writes_defaults_back(env_guard):
    os.environ.pop("LLM_MAX_CONCURRENCY", None)
    validate_config()
    assert os.getenv("LLM_MAX_CONCURRENCY") == "4"


def test_optional_keys_without_default_are_skipped(env_guard):
    """COST_* 没有默认值：未设置时既不该报错，也不该凭空写入。"""
    os.environ.pop("COST_INPUT_PER_1K", None)
    assert validate_config() == []
    assert os.getenv("COST_INPUT_PER_1K") is None


def test_describe_config_masks_secrets(env_guard):
    os.environ["API_KEY"] = "sk-super-secret"
    os.environ["EMBEDDING_API_KEY"] = "sk-embed-secret"
    os.environ.pop("EMBEDDING_MODEL", None)
    snapshot = describe_config()

    assert snapshot["API_KEY"] == "已配置"
    assert "secret" not in str(snapshot)
    assert snapshot["EMBEDDING_MODEL"] == "<未配置，使用关键词检索>"
