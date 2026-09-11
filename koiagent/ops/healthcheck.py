"""容器健康检查：校验心跳文件是否新鲜。

worker 类容器没有 HTTP 端口，因此用「心跳文件」探活：
主程序在连接建立与每次心跳响应成功时都会刷新该文件，
本脚本判断其时间戳是否在允许的时间内，超时即判定不健康。

环境变量
--------
- ``HEALTH_FILE``     心跳文件路径，默认 ``data/health.json``
- ``HEALTH_MAX_AGE``  允许的最大静默秒数，默认 ``180``
"""
import json
import os
import sys
import time


def main() -> int:
    path = os.getenv("HEALTH_FILE", "data/health.json")
    max_age = float(os.getenv("HEALTH_MAX_AGE", "180"))

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        age = time.time() - float(payload.get("ts", 0))
    except Exception as e:
        print(f"unhealthy: 无法读取心跳文件 {path}: {e}")
        return 1

    if age > max_age:
        print(f"unhealthy: 心跳已过期 {age:.0f}s > {max_age:.0f}s")
        return 1

    print(f"healthy: {age:.0f}s 前有心跳 (status={payload.get('status')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
