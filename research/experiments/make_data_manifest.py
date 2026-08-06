# -*- coding: utf-8 -*-
"""
数据快照 manifest 生成器（审查 P0-2 / 复审: 快照 ID + 历史不覆盖）

用法:
    python research/experiments/make_data_manifest.py            # 全量 sha256（慢）
    python research/experiments/make_data_manifest.py --quick    # 仅轻量指纹
    python research/experiments/make_data_manifest.py --no-pointer  # 仅生成快照文件, 不更新指针

覆盖数据源（两最小可复现实验所需）:
    daily        D:/iquant_data/data_v2/data_day1   (settings.daily_data_path)
    index_weight research/chip_momentum/data/index_weight
    index_daily  research/chip_momentum/data/index_daily
    factor_data  research/factor_dic/data

快照管理（复审建议）:
    - 每次运行生成 snapshots/data_manifest_<snapshot_id>.json, **历史快照永不覆盖**;
    - data_manifest.json 仅作指针: {snapshot_id, generated_at, mode, manifest_sha256, path};
    - 实验 run.py 通过指针解析活动快照; 结果绑定 snapshot_id + manifest_sha256;
    - 数据漂移流程: 停实验 → 生成新快照 → 重跑全部实验 → old-vs-new 指标差异 → 人工批准（勿直接改期望值硬过）。
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def _rel(parts):
    return os.path.join(ROOT, *parts)


def get_sources():
    from config.settings import Settings
    s = Settings()
    return {
        "daily": s.daily_data_path,
        "index_weight": _rel(["research", "chip_momentum", "data", "index_weight"]),
        "index_daily": _rel(["research", "chip_momentum", "data", "index_daily"]),
        "factor_data": _rel(["research", "factor_dic", "data"]),
    }


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def fingerprint_dir(path, with_hash):
    """返回目录的轻量指纹 (+ 可选全量 sha256)"""
    if not os.path.isdir(path):
        return {"exists": False}
    fp = {"exists": True, "file_count": 0, "total_size": 0, "mtime_min": None, "mtime_max": None}
    if with_hash:
        fp["hashes"] = {}
    for root, _dirs, files in os.walk(path):
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, path)
            st = os.stat(full)
            fp["file_count"] += 1
            fp["total_size"] += st.st_size
            mt = st.st_mtime
            fp["mtime_min"] = mt if fp["mtime_min"] is None else min(fp["mtime_min"], mt)
            fp["mtime_max"] = mt if fp["mtime_max"] is None else max(fp["mtime_max"], mt)
            if with_hash:
                fp["hashes"][rel] = {
                    "sha256": sha256_file(full),
                    "size": st.st_size,
                    "mtime": datetime.datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S"),
                }
    if fp["mtime_min"] is not None:
        fmt = "%Y-%m-%d %H:%M:%S"
        fp["mtime_min"] = datetime.datetime.fromtimestamp(fp["mtime_min"]).strftime(fmt)
        fp["mtime_max"] = datetime.datetime.fromtimestamp(fp["mtime_max"]).strftime(fmt)
    return fp


def next_snapshot_id(snap_dir):
    """snapshot_id = data_YYYYMMDD-vN（当天第 N 个, 历史保留不覆盖）。"""
    stamp = datetime.date.today().strftime("%Y%m%d")
    n = 1
    while os.path.exists(os.path.join(snap_dir, f"data_manifest_data_{stamp}-v{n}.json")):
        n += 1
    return f"data_{stamp}-v{n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="仅轻量指纹（不逐文件 hash）")
    ap.add_argument("--no-pointer", action="store_true",
                    help="仅生成快照文件到 snapshots/, 不更新 data_manifest.json 指针")
    args = ap.parse_args()

    snap_dir = os.path.join(HERE, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    sid = next_snapshot_id(snap_dir)
    out = os.path.join(snap_dir, f"data_manifest_{sid}.json")

    sources = get_sources()
    manifest = {
        "snapshot_id": sid,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "quick" if args.quick else "full",
        "sources": {},
    }
    for name, path in sources.items():
        print(f"[scan] {name}: {path}")
        fp = fingerprint_dir(path, with_hash=not args.quick)
        manifest["sources"][name] = {"path": path, **fp}
        n = fp.get("file_count", 0)
        print(f"        files={n} size={fp.get('total_size', 0) / 1e6:.1f}MB "
              f"mtime={fp.get('mtime_min')}~{fp.get('mtime_max')}")

    if args.no_pointer:
        # 纯巡检: 不落盘、不更新指针（与活动快照的漂移对比由实验 run.py 完成）
        print("\n[巡检] --no-pointer: 仅显示当前数据指纹, 未落盘; "
              "漂移检测请运行实验 run.py (check_data_manifest)")
        return

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    digest = sha256_file(out)
    print(f"\n[save] {out}")
    print(f"       snapshot_id={sid}  sha256={digest[:16]}…")

    if not args.no_pointer:
        pointer = {
            "snapshot_id": sid,
            "generated_at": manifest["generated_at"],
            "mode": manifest["mode"],
            "manifest_sha256": digest,
            "path": os.path.relpath(out, HERE),
        }
        ptr_path = os.path.join(HERE, "data_manifest.json")
        with open(ptr_path, "w", encoding="utf-8") as fh:
            json.dump(pointer, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"[pointer] {ptr_path} -> {sid} (历史快照保留于 snapshots/)")


if __name__ == "__main__":
    main()
