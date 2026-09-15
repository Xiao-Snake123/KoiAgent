"""提示词文件加载：``PROMPT_DIR`` 下的用户自定义文件优先于内置默认值。

约定（与 ``koiagent.agent.graph`` 保持一致）：
    ``{PROMPT_DIR}/{name}.txt``         ← 用户自定义，**已被 .gitignore 忽略**
    ``{PROMPT_DIR}/{name}_example.txt`` ← 仓库内的示例/默认值
    内置常量                            ← 兜底，保证文件缺失也能运行
"""
from __future__ import annotations

import os
from typing import Optional

from loguru import logger


def load_prompt(name: str, default: str, prompt_dir: Optional[str] = None) -> str:
    """按「自定义 → 示例 → 内置默认」的顺序加载提示词。"""
    directory = prompt_dir or os.getenv("PROMPT_DIR", "prompts")

    for filename in (f"{name}.txt", f"{name}_example.txt"):
        path = os.path.join(directory, filename)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"读取提示词失败，继续尝试下一个来源: {path} ({e})")
            continue

        if content.strip():
            logger.debug(f"已加载提示词 {name} <- {path}（{len(content)} 字符）")
            return content

    logger.debug(f"提示词 {name} 使用内置默认值（{directory} 下未找到文件）")
    return default
