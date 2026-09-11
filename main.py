"""兼容入口：等价于 ``python -m koiagent``。

保留此文件是为了让 ``python main.py`` 这一常见启动方式继续可用。
"""
import sys

from koiagent.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
