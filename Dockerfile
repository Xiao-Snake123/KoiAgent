FROM python:3.10-alpine AS builder

WORKDIR /app

# 只安装构建所需的依赖
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    musl-dev \
    libffi-dev \
    build-base

# 创建虚拟环境并安装依赖
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 第二阶段：最终镜像
FROM python:3.10-alpine

# 添加元数据标签
LABEL maintainer="Xiao-Snake123"
LABEL description="KoiAgent - 智能客服值守机器人"
LABEL version="1.0"

# 设置时区和编码
ENV TZ=Asia/Shanghai \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 只安装运行时必要的包
RUN apk add --no-cache \
    tzdata \
    && ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo Asia/Shanghai > /etc/timezone \
    # 清理apk缓存
    && rm -rf /var/cache/apk/*

# 设置工作目录
WORKDIR /app

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 创建必要的目录
RUN mkdir -p data prompts knowledge

# 复制示例提示词文件并重命名为正式文件
COPY prompts/classify_prompt_example.txt prompts/classify_prompt.txt
COPY prompts/price_prompt_example.txt prompts/price_prompt.txt
COPY prompts/tech_prompt_example.txt prompts/tech_prompt.txt
COPY prompts/default_prompt_example.txt prompts/default_prompt.txt
COPY prompts/critic_prompt_example.txt prompts/critic_prompt.txt
COPY prompts/memory_prompt_example.txt prompts/memory_prompt.txt
COPY prompts/memory_extract_example.txt prompts/memory_extract.txt

# 复制应用代码（包结构）
COPY koiagent/ koiagent/
COPY main.py ./
COPY knowledge/ knowledge/
# 业务配置（议价策略）。不拷的话容器内会回退到内置默认策略，
# 导致改了 config/bargain_policy.json 却不生效
COPY config/ config/

# 健康检查：worker 类容器无 HTTP 端口，用「心跳文件」探活
HEALTHCHECK --interval=60s --timeout=10s --start-period=90s --retries=3 CMD ["python", "-m", "koiagent.ops.healthcheck"]

# 容器启动时运行的命令
CMD ["python", "-m", "koiagent"]
