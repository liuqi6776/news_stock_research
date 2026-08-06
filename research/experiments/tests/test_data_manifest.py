# -*- coding: utf-8 -*-
"""数据快照检测（check_data_manifest）逻辑测试: 无变化 / 内容修改 / 新增 / 删除。

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


def _make_manifest(d, files, meta_dir):
    """files: {rel: content}; 生成 full 模式 manifest（含 per-file hash/size/mtime）。

    manifest 写入 meta_dir（必须在被扫描目录之外, 避免自身混入 file_count）。
    """
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
    mp = os.path.join(meta_dir, "manifest.json")
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(mani, fh, ensure_ascii=False, indent=2)
    return mp


def test_no_change_ok():
    d = tempfile.mkdtemp()
    meta = tempfile.mkdtemp()
    _mkfile(d, "a.txt", "hello")
    mp = _make_manifest(d, {"a.txt": "hello"}, meta)
    r = _common.check_data_manifest(mp)
    assert r["ok"] is True, r
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


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
