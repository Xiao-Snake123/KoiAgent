"""pytest 全局配置：注入离线可用环境变量，保证测试不依赖真实 API 与当前工作目录。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 构造 ChatOpenAI 需要 api_key；测试不会真正发起请求
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
os.environ.setdefault("MODEL_NAME", "qwen-max")

# 测试默认走「关键词检索」，避免联网调用 Embedding
os.environ.pop("EMBEDDING_MODEL", None)

# 使用绝对路径，保证从任意目录执行 pytest 都稳定
os.environ.setdefault("KNOWLEDGE_DIR", os.path.join(ROOT, "knowledge"))
os.environ.setdefault("PROMPT_DIR", os.path.join(ROOT, "prompts"))
