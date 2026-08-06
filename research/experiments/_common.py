# -*- coding: utf-8 -*-
"""research/experiments 共享工具: 配置加载 / 环境探测 / 指标对比 / txt 解析"""
import json
import os
import subprocess


def repo_root():
    # experiments/_common.py -> research -> quant_system_v2
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        raise SystemExit("缺少 pyyaml: 请先 `pip install pyyaml`")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def probe_environment():
    """探测运行时环境: python/核心依赖版本 + 上游仓库 commit。"""
    import platform
    env = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "scipy", "pyarrow", "matplotlib"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "?")
        except ImportError:
            env[mod] = "missing"
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root(), capture_output=True, text=True)
        env["upstream_commit"] = out.stdout.strip() or "n/a"
    except Exception:
        env["upstream_commit"] = "n/a"
    return env


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def compare_metrics(actual, expected, rtol=0.01, atol=0.0):
    """递归对比 actual/expected, 返回差异字符串列表; 空列表 = 全部通过。"""
    diffs = []

    def walk(a, e, prefix=""):
        if not isinstance(a, dict):
            a = {}
        for k, ev in e.items():
            key = f"{prefix}{k}"
            if isinstance(ev, dict):
                walk(a.get(k, {}), ev, key + ".")
            elif k not in a:
                diffs.append(f"missing {key}")
            else:
                av = a[k]
                if av is None or (isinstance(av, float) and av != av):
                    diffs.append(f"{key}: actual NaN")
                    continue
                tol = rtol * abs(ev) + atol
                if abs(av - ev) > tol:
                    diffs.append(f"{key}: {av:.6f} vs expected {ev:.6f} (tol {tol:.6f})")

    walk(actual, expected)
    return diffs


def parse_summary_txt(path):
    """解析 run_validation 的 summary_<factor>.txt (key=value 行)。"""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def parse_risk_control_txt(path, labels):
    """解析 risk_control_bt.txt 固定宽度表, 按 label 前缀取行。

    列: 策略 年化 Sharpe MaxDD 月胜率 超额vETF 卡玛 强段均仓
    返回 {label: {cagr, sharpe, mdd, win, excess_v_etf, calmar, avg_weight}}
    """
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.rstrip("\n")
            for lb in labels:
                if s.startswith(lb):
                    parts = s[len(lb):].split()
                    if len(parts) >= 7:
                        rows[lb] = {
                            "cagr": float(parts[0].rstrip("%")) / 100.0,
                            "sharpe": float(parts[1]),
                            "mdd": float(parts[2].rstrip("%")) / 100.0,
                            "win": float(parts[3].rstrip("%")) / 100.0,
                            "excess_v_etf": float(parts[4].rstrip("%")) / 100.0,
                            "calmar": float(parts[5]),
                            "avg_weight": float(parts[6].rstrip("%")) / 100.0,
                        }
    return rows
