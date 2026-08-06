# -*- coding: utf-8 -*-
"""
数据快照 manifest 生成器（审查 P0-2: data_manifest.json）

用法:
    python research/experiments/make_data_manifest.py            # 全量 sha256（慢）
    python research/experiments/make_data_manifest.py --quick    # 仅轻量指纹

覆盖数据源（两最小可复现实验所需）:
    daily        D:/iquant_data/data_v2/data_day1   (settings.daily_data_path)
    index_weight research/chip_momentum/data/index_weight
    index_daily  research/chip_momentum/data/index_daily
    factor_data  research/factor_dic/data

输出: research/experiments/data_manifest.json
用途: 实验 run.py 运行前调用 _common.check_data_manifest 检测数据漂移;
      manifest 的 hashes 记录"实验结果对应的数据快照"。
"""
import argparse
import hashlib
import json
import os
import sys
import datetime

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="仅轻量指纹（不逐文件 hash）")
    ap.add_argument("--out", default=os.path.join(HERE, "data_manifest.json"))
    args = ap.parse_args()

    sources = get_sources()
    manifest = {
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

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n[保存] {args.out}")


if __name__ == "__main__":
    main()
