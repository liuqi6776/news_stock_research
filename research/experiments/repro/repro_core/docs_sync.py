# -*- coding: utf-8 -*-
"""文档指标同步（从 experiments/_common.py 提取并扩展）: 把实际运行指标写回结论库文档,
并校验与冻结期望一致 —— 防止"文档表格与输出文件漂移"（复审 P0 主线）。

约定:
  - 结论库文档表格行由指标渲染, 数字来源 = actual_metrics.json;
  - 冻结期望 = expected_metrics.json（含冻结时 data_snapshot）;
  - compare_metrics 数值键做容差比较, 非数值键仅存在性检查。
"""
from __future__ import annotations

import json
import os


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def compare_metrics(actual: dict, expected: dict, rtol: float = 0.01, atol: float = 0.0) -> list[str]:
    """递归对比 actual/expected, 返回差异字符串列表; 空列表 = 全部通过。

    数值键做容差比较 (tol = rtol*|expected| + atol); 非数值键仅要求 actual 存在。
    """
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
            elif isinstance(ev, (int, float)):
                av = a[k]
                if av is None or (isinstance(av, float) and av != av):
                    diffs.append(f"{key}: actual NaN")
                    continue
                tol = rtol * abs(ev) + atol
                if abs(av - ev) > tol:
                    diffs.append(f"{key}: {av:.6f} vs expected {ev:.6f} (tol {tol:.6f})")

    walk(actual, expected)
    return diffs


def parse_summary_txt(path: str) -> dict:
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


def parse_risk_control_txt(path: str, labels: list[str]) -> dict:
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


def render_pct(v, digits: int = 2) -> str:
    """指标 -> 文档展示字符串（如 0.1383 -> '13.83%'）。"""
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v * 100:.{digits}f}%"


def sync_doc_table(md_path: str, table_title: str, metrics: dict, label: str,
                   snapshot_note: str) -> bool:
    """把一组指标同步进结论库 markdown 表格行（防文档-输出漂移）。

    md_path: 结论库文档路径; table_title: 表格标题(用于定位首个 `| ` 表头行之后的位置);
    metrics: {列名: 数值}; label: 该行第一个单元格(策略名); snapshot_note: 追加注记(如快照 ID)。

    策略: 定位文档中 `| <label>` 开头的行并整行替换; 不存在则追加在表头下。
    返回是否发生替换。
    """
    if not os.path.isfile(md_path):
        return False
    with open(md_path, encoding="utf-8") as fh:
        lines = fh.readlines()

    cols = list(metrics.keys())
    cells = [label] + [render_pct(metrics[c]) if "rate" in c or "cagr" in c or "mdd" in c or "win" in c or "excess" in c
                       else (f"{metrics[c]:.2f}" if isinstance(metrics[c], (int, float)) and metrics[c] == metrics[c]
                             else str(metrics[c]))
                       for c in cols]
    new_line = "| " + " | ".join(cells) + " |" + (f"  <!-- {snapshot_note} -->" if snapshot_note else "") + "\n"

    replaced = False
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(f"| {label}") and "|" in ln:
            lines[i] = new_line
            replaced = True
            break
    if not replaced:
        # 找不到对应行 -> 在表头分隔行后插入
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith("|---") or ln.lstrip().startswith("| ---"):
                lines.insert(i + 1, new_line)
                replaced = True
                break
    if replaced:
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
    return replaced
