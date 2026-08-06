# -*- coding: utf-8 -*-
"""drift 测试: 数据漂移检测（无变化/内容修改/新增/删除/指针缺失/哈希篡改）。"""
import json
import os

import pytest

from repro_core import drift


def _write(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _make_snapshot(tmp_path):
    """构造: tmp_path/daily/{a.parquet, b.parquet} + 指针 -> 快照。"""
    daily = tmp_path / "daily"
    daily.mkdir()
    _write(daily / "a.parquet", "data-a-v1")
    _write(daily / "b.parquet", "data-b-v1")

    files = {}
    for fn in ("a.parquet", "b.parquet"):
        full = os.path.join(str(daily), fn)
        st = os.stat(full)
        files[fn] = {"size": st.st_size,
                     "mtime": __import__("datetime").datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                     "sha256": drift.sha256_file(full)}
    snap = {"snapshot_id": "data_20260806-v9", "generated_at": "2026-08-06 00:00:00",
            "mode": "full", "sources": {"daily": {"path": str(daily), "file_count": 2,
                                                   "total_size": sum(v["size"] for v in files.values()),
                                                   "mtime_min": None, "mtime_max": None,
                                                   "hashes": files}}}
    snap_path = tmp_path / "data_manifest_data_20260806-v9.json"
    _write(snap_path, json.dumps(snap))
    ptr = {"snapshot_id": "data_20260806-v9", "path": os.path.basename(str(snap_path)),
           "manifest_sha256": drift.sha256_file(str(snap_path))}
    ptr_path = tmp_path / "data_manifest.json"
    _write(ptr_path, json.dumps(ptr))
    return str(ptr_path), daily


def test_drift_no_change_ok(tmp_path):
    ptr, _ = _make_snapshot(tmp_path)
    r = drift.check_manifest(ptr)
    assert r["ok"] is True
    assert r["drift_msgs"] == []
    assert r["snapshot_id"] == "data_20260806-v9"


def test_drift_content_modified(tmp_path):
    ptr, daily = _make_snapshot(tmp_path)
    _write(daily / "a.parquet", "data-a-v2-CHANGED")   # 内容变化
    r = drift.check_manifest(ptr)
    assert r["ok"] is False
    assert any("内容变化" in m for m in r["drift_msgs"])


def test_drift_file_removed(tmp_path):
    ptr, daily = _make_snapshot(tmp_path)
    os.remove(daily / "b.parquet")
    r = drift.check_manifest(ptr)
    assert r["ok"] is False
    assert any("文件数" in m for m in r["drift_msgs"])


def test_drift_file_added_is_update_only(tmp_path):
    ptr, daily = _make_snapshot(tmp_path)
    _write(daily / "c.parquet", "new-file")             # 新增 -> 不阻断
    r = drift.check_manifest(ptr)
    assert r["ok"] is True
    assert any("文件数" in m for m in r["update_msgs"])


def test_drift_missing_pointer(tmp_path):
    r = drift.check_manifest(str(tmp_path / "nope.json"))
    assert r["ok"] is False
    assert r["drift_msgs"]


def test_drift_pointer_hash_tamper(tmp_path):
    ptr, _ = _make_snapshot(tmp_path)
    _write(ptr, '{"snapshot_id": "x", "path": "whatever", "manifest_sha256": "deadbeef"}')
    r = drift.check_manifest(ptr)
    assert r["ok"] is False
    assert any("哈希不一致" in m or "缺失" in m for m in r["drift_msgs"])


def test_drift_manifest_json_corrupt(tmp_path):
    ptr, _ = _make_snapshot(tmp_path)
    _write(ptr, "{not json")
    r = drift.check_manifest(ptr)
    assert r["ok"] is False
