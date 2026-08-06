# -*- coding: utf-8 -*-
"""数据漂移检测（从 experiments/_common.py 提取, 参数显式化以便外部独立使用）。

策略: 轻量扫描(文件数/总大小/mtime 范围)快速比对, 逐文件用 size/mtime 快速跳过,
仅在异常时 sha256 核验 —— 日常运行零 hash 开销; 快照文件自身有指针哈希防篡改。
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os


def sha256_file(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def scan_light(path: str) -> tuple[dict, dict]:
    """轻量扫描: 返回 (聚合指纹, {rel: (size, mtime_str)})。不做 hash。"""
    fmt = "%Y-%m-%d %H:%M:%S"
    files, total = {}, 0
    m_min = m_max = None
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


def load_json_checked(path: str) -> dict:
    """读取 JSON; 损坏 -> ValueError（由调用方转为漂移消息）。"""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def check_manifest(manifest_path: str) -> dict:
    """检测数据漂移（与 _common.check_data_manifest 逻辑一致, 显式传指针路径）。

    返回 {ok, drift_msgs, update_msgs, snapshot_id, manifest_sha256,
          baseline_generated_at, mode}:
      - ok=False / drift_msgs 非空 -> 既有文件被修改/删除/指针损坏, 必须阻断（退出码 2）;
      - update_msgs 非空 -> 仅新增文件/元数据刷新, 提示不阻断。
    """
    base = {"drift_msgs": [], "update_msgs": [], "ok": True,
            "snapshot_id": None, "manifest_sha256": None,
            "baseline_generated_at": None, "mode": None}
    if not os.path.isfile(manifest_path):
        base["drift_msgs"].append(f"缺少数据快照指针 {os.path.basename(manifest_path)}")
        base["ok"] = False
        return base
    try:
        ptr = load_json_checked(manifest_path)
    except json.JSONDecodeError as e:
        base["drift_msgs"].append(f"数据快照指针损坏（{os.path.basename(manifest_path)} 非合法 JSON: {e}）")
        base["ok"] = False
        return base

    # 旧格式兼容: 顶层含 sources 直接当 manifest
    if "sources" in ptr:
        m, base["snapshot_id"], base["manifest_sha256"] = ptr, "legacy", None
    else:
        snap_path = os.path.join(os.path.dirname(manifest_path), ptr.get("path", ""))
        if not os.path.isfile(snap_path):
            base["drift_msgs"].append(f"快照文件缺失: {snap_path}")
            base["ok"] = False
            return base
        try:
            m = load_json_checked(snap_path)
        except json.JSONDecodeError as e:
            base["drift_msgs"].append(f"快照文件损坏（{snap_path} 非合法 JSON: {e}）")
            base["ok"] = False
            return base
        digest = sha256_file(snap_path)
        if ptr.get("manifest_sha256") and digest != ptr["manifest_sha256"]:
            base["drift_msgs"].append(f"快照文件与指针哈希不一致（{snap_path} 疑似被修改/损坏）")
            base["ok"] = False
            return base
        base["snapshot_id"] = ptr.get("snapshot_id") or m.get("snapshot_id")
        base["manifest_sha256"] = ptr.get("manifest_sha256")
    base["baseline_generated_at"] = m.get("generated_at")
    base["mode"] = m.get("mode")

    for name, bl in m.get("sources", {}).items():
        p = bl.get("path")
        if not (p and os.path.isdir(p)):
            base["drift_msgs"].append(f"{name}: 数据目录不存在/路径变化 ({p})")
            base["ok"] = False
            continue
        cur_fp, cur_files = scan_light(p)

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
            base["update_msgs"].append(f"{name}: 最近写入 {bl['mtime_max']} -> {cur_fp['mtime_max']}")

        # 2) 逐文件核验（full 基线才有 hashes; size/mtime 相同则跳过）
        hashes = bl.get("hashes")
        if hashes:
            for rel, (size, mt) in cur_files.items():
                entry = hashes.get(rel)
                if entry is None:
                    continue  # 新增文件已由 file_count 差异覆盖
                if entry.get("size") == size and entry.get("mtime") == mt:
                    continue
                if entry.get("sha256") != sha256_file(os.path.join(p, rel)):
                    base["drift_msgs"].append(f"{name}: 文件内容变化 {rel}")
                    base["ok"] = False
                else:
                    base["update_msgs"].append(f"{name}: 元数据刷新（内容一致）{rel}")

    return base
