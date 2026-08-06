# pytest 收集入口: 确保 repro/ 目录在 sys.path（根目录/本目录两种运行方式均生效）
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
