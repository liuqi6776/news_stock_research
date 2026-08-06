# -*- coding: utf-8 -*-
"""数据快照检测（check_data_manifest）逻辑测试: 无变化 / 内容修改 / 新增 / 删除 /
指针哈希校验 / 旧格式兼容。

运行:
    C:\\Users\\liuqi\\anaconda3\\python.exe -m pytest research/experiments/tests/test_data_manifest.py
"""
import datetime
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXPS = os.path.dirname(HERE)
sys.path.insert(0, EXPS)

import _common  # noqa: E402


def _mkfile(d, rel, content, mtime=None):
    full = os.path.join(d, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as fh:
        fh.write(content)
    if mtime is not None:
        os.utime(full, (mtime, mtime))
    return full


def _make_manifest(d, files, meta_dir, snapshot_id="data_test-v1"):
    """构造指针 + 快照文件（快照写入 meta_dir/snapshots/, 指针在 meta_dir/ 顶层）。"""
    fmt = "%Y-%m-%d %H:%M:%S"
    hashes, total, mtimes = {}, 0, []
    for rel, content in files.items():
        full = os.path.join(d, rel)
        st = os.stat(full)
        hashes[rel] = {
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "size": st.st_size,
            "mtime": datetime.datetime.fromtimestamp(st.st_mtime).strftime(fmt),
        }
        total += st.st_size
        mtimes.append(st.st_mtime)
    mani = {
        "snapshot_id": snapshot_id,
        "generated_at": "2026-08-06 00:00:00",
        "mode": "full",
        "sources": {
            "t": {
                "path": d, "exists": True,
                "file_count": len(files), "total_size": total,
                "mtime_min": datetime.datetime.fromtimestamp(min(mtimes)).strftime(fmt),
                "mtime_max": datetime.datetime.fromtimestamp(max(mtimes)).strftime(fmt),
                "hashes": hashes,
            }
        },
    }
    snap_dir = os.path.join(meta_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    snap_path = os.path.join(snap_dir, f"data_manifest_{snapshot_id}.json")
    with open(snap_path, "w", encoding="utf-8") as fh:
        json.dump(mani, fh, ensure_ascii=False, indent=2)
    digest = _common._sha256_file(snap_path)
    ptr = {
        "snapshot_id": snapshot_id,
        "generated_at": mani["generated_at"],
        "mode": "full",
        "manifest_sha256": digest,
        "path": os.path.relpath(snap_path, meta_dir),
    }
    mp = os.path.join(meta_dir, "data_manifest.json")
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(ptr, fh, ensure_ascii=False, indent=2)
    return mp


def test_no_change_ok():
    d = tempfile.mkdtemp()
    meta = tempfile.mkdtemp()
    _mkfile(d, "a.txt", "hello")
    mp = _make_manifest(d, {"a.txt": "hello"}, meta)
    r = _common.check_data_manifest(mp)
    assert r["ok"] is True, r
    assert r["snapshot_id"] == "data_test-v1", r
    assert r["manifest_sha256"], r
    assert r["drift_msgs"] == [] and r["update_msgs"] == []


def test_content_change_drift():
    d = tempfile.mkdtemp()
    meta = tempfile.mkdtemp()
    _mkfile(d, "a.txt", "hello")
    mp = _make_manifest(d, {"a.txt": "hello"}, meta)
    _mkfile(d, "a.txt", "world!")  # 内容 + 尺寸变化
    r = _common.check_data_manifest(mp)
    assert r["ok"] is False, r
    assert any("内容变化" in m for m in r["drift_msgs"]), r


def test_content_change_same_size_drift():
    """同尺寸内容修改: size 不变 mtime 变 → 走 sha256 核验 → 必须判漂移。"""
    d = tempfile.mkdtemp()
    meta = tempfile.mkdtemp()
    _mkfile(d, "a.txt", "hello")
    mp = _make_manifest(d, {"a.txt": "hello"}, meta)
    # Windows mtime 粒度粗, 显式偏移时间戳确保 mtime 变化
    _mkfile(d, "a.txt", "heLlo", mtime=os.stat(os.path.join(d, "a.txt")).st_mtime + 100)
    r = _common.check_data_manifest(mp)
    assert r["ok"] is False, r
    assert any("内容变化" in m for m in r["drift_msgs"]), r


def test_new_file_update_only():
    d = tempfile.mkdtemp()
    meta = tempfile.mkdtemp()
    _mkfile(d, "a.txt", "hello")
    mp = _make_manifest(d, {"a.txt": "hello"}, meta)
    _mkfile(d, "b.txt", "new data")  # 仅新增
    r = _common.check_data_manifest(mp)
    assert r["ok"] is True, r
    assert r["drift_msgs"] == [], r
    assert any("文件数" in m for m in r["update_msgs"]), r


def test_missing_file_drift():
    d = tempfile.mkdtemp()
    meta = tempfile.mkdtemp()
    _mkfile(d, "a.txt", "hello")
    _mkfile(d, "b.txt", "bye")
    mp = _make_manifest(d, {"a.txt": "hello", "b.txt": "bye"}, meta)
    os.remove(os.path.join(d, "b.txt"))  # 删除
    r = _common.check_data_manifest(mp)
    assert r["ok"] is False, r
    assert any("文件数" in m for m in r["drift_msgs"]), r


def test_missing_manifest_blocked():
    d = tempfile.mkdtemp()
    r = _common.check_data_manifest(os.path.join(d, "none.json"))
    assert r["ok"] is False
    assert any("缺少数据快照" in m for m in r["drift_msgs"]), r


def test_pointer_hash_mismatch_drift():
    """指针哈希核验: 快照文件被修改 → 必须判异常（防篡改/损坏）。"""
    d = tempfile.mkdtemp()
    meta = tempfile.mkdtemp()
    _mkfile(d, "a.txt", "hello")
    mp = _make_manifest(d, {"a.txt": "hello"}, meta)
    snap_path = os.path.join(meta, "snapshots", "data_manifest_data_test-v1.json")
    with open(snap_path, "r", encoding="utf-8") as fh:
        data = fh.read()
    with open(snap_path, "w", encoding="utf-8") as fh:
        fh.write(data + "\n# tampered\n")  # 快照文件被改动（损坏或哈希不一致均可判定）
    r = _common.check_data_manifest(mp)
    assert r["ok"] is False, r
    assert any("损坏" in m or "哈希不一致" in m for m in r["drift_msgs"]), r


def test_missing_snapshot_file_blocked():
    """指针指向的快照文件缺失 → 必须阻断。"""
    d = tempfile.mkdtemp()
    meta = tempfile.mkdtemp()
    _mkfile(d, "a.txt", "hello")
    mp = _make_manifest(d, {"a.txt": "hello"}, meta)
    os.remove(os.path.join(meta, "snapshots", "data_manifest_data_test-v1.json"))
    r = _common.check_data_manifest(mp)
    assert r["ok"] is False, r
    assert any("快照文件缺失" in m for m in r["drift_msgs"]), r


def test_legacy_format_compat():
    """旧格式兼容: 顶层含 sources 直接当 manifest（snapshot_id=legacy）。"""
    d = tempfile.mkdtemp()
    meta = tempfile.mkdtemp()
    _mkfile(d, "a.txt", "hello")
    mp = _make_manifest(d, {"a.txt": "hello"}, meta)
    with open(mp, "r", encoding="utf-8") as fh:
        ptr = json.load(fh)
    snap_path = os.path.join(meta, ptr["path"])
    with open(snap_path, "r", encoding="utf-8") as fh:
        mani = json.load(fh)
    with open(mp, "w", encoding="utf-8") as fh:  # 覆盖为旧格式（顶层 sources）
        json.dump(mani, fh, ensure_ascii=False, indent=2)
    r = _common.check_data_manifest(mp)
    assert r["ok"] is True, r
    assert r["snapshot_id"] == "legacy", r


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
