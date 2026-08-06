# -*- coding: utf-8 -*-
"""research/experiments 共享工具: 配置加载 / 环境探测 / 指标对比 / txt 解析 / 数据快照检测"""
import datetime
import hashlib
import json
import os
import subprocess

MANIFEST_DEFAULT = None  # 延迟解析: 依赖 repo_root


def repo_root():
    # experiments/_common.py -> research -> quant_system_v2
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _default_manifest_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_manifest.json")


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


def _sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _scan_light(path):
    """轻量扫描: 每文件 rel -> (size, mtime_str) + 聚合指纹（不做 hash）。"""
    files = {}
    total = 0
    m_min = m_max = None
    fmt = "%Y-%m-%d %H:%M:%S"
    for root, _dirs, names in os.walk(path):
        for fn in names:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, path)
            st = os.stat(full)
            files[rel] = (st.st_size, datetime.datetime.fromtimestamp(st.st_mtime).strftime(fmt))
            total += st.st_size
            m_min = st.st_mtime if m_min is None else min(m_min, st.st_mtime)
            m_max = st.st_mtime if m_max is None else max(m_max, st.st_mtime)
    fp = {
        "file_count": len(files),
        "total_size": total,
        "mtime_min": datetime.datetime.fromtimestamp(m_min).strftime(fmt) if m_min is not None else None,
        "mtime_max": datetime.datetime.fromtimestamp(m_max).strftime(fmt) if m_max is not None else None,
    }
    return fp, files


def check_data_manifest(manifest_path=None):
    """检测数据漂移: 对比当前数据源与基线 manifest（审查 P0-2 可复现性）。

    基线: research/experiments/data_manifest.json（make_data_manifest.py 生成）
    返回 dict:
        ok          True=可安全运行
        drift_msgs  漂移（既有文件被修改/删除）→ 必须阻断, 先重建 manifest
        update_msgs 仅新增文件/数据更新 → 提示, 不阻断
        baseline_generated_at, mode
    """
    path = manifest_path or _default_manifest_path()
    base = {"drift_msgs": [], "update_msgs": [], "ok": True,
            "baseline_generated_at": None, "mode": None}
    if not os.path.isfile(path):
        base["drift_msgs"].append(f"缺少数据快照 {os.path.basename(path)}: "
                                  f"先运行 make_data_manifest.py 生成基线")
        base["ok"] = False
        return base
    with open(path, encoding="utf-8") as fh:
        m = json.load(fh)
    base["baseline_generated_at"] = m.get("generated_at")
    base["mode"] = m.get("mode")

    for name, bl in m.get("sources", {}).items():
        p = bl.get("path")
        if not (p and os.path.isdir(p)):
            base["drift_msgs"].append(f"{name}: 数据目录不存在/路径变化 ({p})")
            base["ok"] = False
            continue
        cur_fp, cur_files = _scan_light(p)

        # 1) 聚合指纹对比
        if cur_fp["file_count"] != bl["file_count"]:
            d = cur_fp["file_count"] - bl["file_count"]
            (base["update_msgs"] if d > 0 else base["drift_msgs"]).append(
                f"{name}: 文件数 {bl['file_count']} -> {cur_fp['file_count']} ({'+' if d > 0 else ''}{d})")
            if d < 0:
                base["ok"] = False
        if cur_fp["total_size"] != bl["total_size"]:
            base["update_msgs"].append(
                f"{name}: 总大小 {bl['total_size'] / 1e6:.1f}MB -> {cur_fp['total_size'] / 1e6:.1f}MB")
        if cur_fp["mtime_max"] != bl["mtime_max"]:
            base["update_msgs"].append(
                f"{name}: 最近写入 {bl['mtime_max']} -> {cur_fp['mtime_max']}")

        # 2) 逐文件核验（full 基线才有 hashes; 用 size/mtime 快速跳过, 异常才 sha256）
        hashes = bl.get("hashes")
        if hashes:
            for rel, (size, mt) in cur_files.items():
                entry = hashes.get(rel)
                if entry is None:
                    continue  # 新增文件已由 file_count 差异覆盖
                if entry.get("size") == size and entry.get("mtime") == mt:
                    continue  # 未变
                # size/mtime 异常 → sha256 确认是否内容漂移
                if entry.get("sha256") != _sha256_file(os.path.join(p, rel)):
                    base["drift_msgs"].append(f"{name}: 文件内容变化 {rel}")
                    base["ok"] = False
                else:
                    base["update_msgs"].append(f"{name}: 元数据刷新（内容一致）{rel}")

    return base
