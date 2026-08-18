# -*- coding: utf-8 -*-
"""
fund_research 数据加载模块
==========================
本地场内基金数据: D:\\iquant_data\\data_v2\\fund1
  - fund_basic_E.parquet           基金基础信息 (2844 只)
  - {YYYYMMDD}.parquet             每日全部场内基金日线 (ts_code 维度, 2014 至今)
本地场外基金数据: D:\\iquant_data\\data_v2\\fund2
  - fund_basic_O.parquet           场外基金基础信息 (27449 只, 含类型)
  - nav/{code}.parquet             每只场外基金净值: date/unit_nav/acc_nav/pct_chg

主要接口:
  场内:
    load_fund_basic()            -> DataFrame(ts_code, name, management, list_date...)
    load_panel(rebuild=False)    -> 宽表面板 index=trade_date, columns=(ts_code, field)
    get_fund_returns(...)        -> 便捷取收益序列
    list_funds(by=...)           -> 按名称/类型浏览场内基金
  场外:
    load_otc_basic()             -> 场外基金列表(代码/名称/类型)
    load_otc_nav(code)           -> 单只场外基金净值序列
    load_otc_nav_panel(...)      -> 多只场外基金净值面板(unit_nav 单位净值)
    list_otc_funds(keyword=...)  -> 按名称/类型浏览场外基金

注意:
  - fund1 是"场内"价格(ETF/LOF 交易所成交价), fund2 是"场外"净值(申购赎回口径).
  - 运行环境: 用 base python (C:\\Users\\liuqi\\anaconda3\\python.exe),
    iquant conda 环境当前 numpy/pyarrow 不兼容无法读 parquet.
"""

import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    FUND_DATA_DIR,
    FUND_BASIC_PATH,
    PANEL_CACHE_PATH,
    OTC_DATA_DIR,
    OTC_NAV_DIR,
    OTC_BASIC_PATH,
    OTC_PANEL_CACHE_PATH,
    DEFAULT_START,
    DEFAULT_END,
    VERBOSE,
)


def log(msg):
    if VERBOSE:
        print(f"[data_loader] {msg}")


# ---------------------------------------------------------------------------
# 基础信息
# ---------------------------------------------------------------------------
def load_fund_basic():
    """场内基金基础信息: ts_code, name, management, found_date, list_date ..."""
    if not os.path.exists(FUND_BASIC_PATH):
        raise FileNotFoundError(f"缺少基金基础信息文件: {FUND_BASIC_PATH}")
    df = pd.read_parquet(FUND_BASIC_PATH)
    # 统一日期列为 str 的 YYYYMMDD
    for col in ("found_date", "due_date", "list_date", "delist_date", "issue_date"):
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"None": np.nan, "nan": np.nan})
    return df


# ---------------------------------------------------------------------------
# 日线面板
# ---------------------------------------------------------------------------
def _iter_daily_files():
    """按日期排序遍历日线 parquet 文件"""
    files = glob.glob(os.path.join(FUND_DATA_DIR, "????????.parquet"))
    files = [f for f in files if os.path.basename(f)[:8].isdigit()]
    return sorted(files)


def _build_panel(start=DEFAULT_START, end=None):
    """合并全部日线 parquet -> 宽表面板 (index=trade_date, columns=(ts_code, field))"""
    if end is None:
        end = datetime.now().strftime("%Y%m%d")
    files = _iter_daily_files()
    files = [f for f in files if start <= os.path.basename(f)[:8] <= end]
    log(f"合并 {len(files)} 个交易日文件 ({start} ~ {end}) ...")

    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            log(f"跳过 {os.path.basename(f)}: {e}")
            continue
        if df.empty:
            continue
        df = df.set_index(["ts_code", "trade_date"])
        frames.append(df)
    if not frames:
        raise RuntimeError("未读取到任何日线数据")

    long = pd.concat(frames).sort_index()
    # long: MultiIndex(ts_code, trade_date) x columns(open/high/low/close/vol/...)
    panel = long.unstack(level=0)  # index=trade_date, columns=(field, ts_code)
    panel.index = pd.to_datetime(panel.index, format="%Y%m%d")
    panel.index.name = "trade_date"
    return panel


def load_panel(rebuild=False):
    """宽表日线面板, 带 parquet 缓存。首次构建较慢, 之后秒级加载。"""
    if not rebuild and os.path.exists(PANEL_CACHE_PATH):
        log(f"从缓存加载面板 {PANEL_CACHE_PATH}")
        return pd.read_parquet(PANEL_CACHE_PATH)

    panel = _build_panel()
    panel.to_parquet(PANEL_CACHE_PATH)
    log(f"面板构建完成, 写入缓存: {PANEL_CACHE_PATH}")
    return panel


# ---------------------------------------------------------------------------
# 便捷取数
# ---------------------------------------------------------------------------
def get_close_panel(rebuild=False):
    """收盘价面板: index=trade_date, columns=ts_code"""
    panel = load_panel(rebuild=rebuild)
    return panel["close"]


def get_fund_returns(rebuild=False):
    """日收益率面板 (前收口径): r_t = close_t / close_{t-1} - 1"""
    close = get_close_panel(rebuild=rebuild)
    return close.pct_change(fill_method=None)


def get_fund_nav_series(ts_code, rebuild=False):
    """单只基金收盘价序列 (场内价)。场外净值请用 OTC 抓取接口。"""
    close = get_close_panel(rebuild=rebuild)
    if ts_code not in close.columns:
        raise KeyError(f"{ts_code} 不在面板中, 可用 list_funds() 查看")
    return close[ts_code].dropna()


def list_funds(keyword=None, by="name", head=None):
    """浏览基金: 按名称关键词或管理公司过滤"""
    basic = load_fund_basic()
    if keyword:
        if by == "name":
            mask = basic["name"].astype(str).str.contains(keyword, na=False)
        elif by == "management":
            mask = basic["management"].astype(str).str.contains(keyword, na=False)
        else:
            mask = basic[by].astype(str).str.contains(keyword, na=False)
        basic = basic[mask]
    if head:
        basic = basic.head(head)
    return basic


# ---------------------------------------------------------------------------
# 场外基金 (fund2)
# ---------------------------------------------------------------------------
def load_otc_basic():
    """场外基金列表: code, pinyin, name, fund_type, pinyin_full"""
    if not os.path.exists(OTC_BASIC_PATH):
        raise FileNotFoundError(f"缺少场外基金列表: {OTC_BASIC_PATH}, 先运行 data/fetch_fund2.py")
    return pd.read_parquet(OTC_BASIC_PATH)


def load_otc_nav(code):
    """单只场外基金净值: index=date, columns=[unit_nav, acc_nav, pct_chg]"""
    path = os.path.join(OTC_NAV_DIR, f"{code}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"无 {code} 净值文件: {path}")
    df = pd.read_parquet(path)
    df = df.set_index("date").sort_index()
    return df


def load_otc_nav_panel(codes=None, field="unit_nav", rebuild=False, start=None, end=None):
    """
    多只场外基金净值面板: index=date, columns=code
    field: unit_nav(单位净值) / acc_nav(累计净值) / pct_chg(日增长率)
    首次构建后缓存到 cache/otc_nav_panel.parquet
    """
    if codes is None:
        codes = sorted(
            f[:-8] for f in glob.glob(os.path.join(OTC_NAV_DIR, "????????.parquet"))
        )
    cache_key = f"{field}_panel"
    cache_path = os.path.join(os.path.dirname(OTC_PANEL_CACHE_PATH),
                              f"otc_{field}_panel.parquet")
    if not rebuild and os.path.exists(cache_path):
        log(f"从缓存加载场外{field}面板: {cache_path}")
        panel = pd.read_parquet(cache_path)
    else:
        frames = []
        for c in codes:
            try:
                df = load_otc_nav(c)
                frames.append(df[field].rename(c))
            except FileNotFoundError:
                continue
        if not frames:
            raise RuntimeError("无任何净值数据")
        panel = pd.concat(frames, axis=1).sort_index()
        panel.to_parquet(cache_path)
        log(f"场外{field}面板构建完成: {panel.shape} -> {cache_path}")
    if start:
        panel = panel[panel.index >= pd.to_datetime(start)]
    if end:
        panel = panel[panel.index <= pd.to_datetime(end)]
    return panel


def list_otc_funds(keyword=None, by="name", head=None):
    """浏览场外基金: 按名称/类型关键词过滤"""
    basic = load_otc_basic()
    if keyword:
        mask = basic[by].astype(str).str.contains(keyword, na=False)
        basic = basic[mask]
    if head:
        basic = basic.head(head)
    return basic


if __name__ == "__main__":
    # 冒烟测试
    basic = load_fund_basic()
    log(f"场内基金基础信息: {len(basic)} 只")
    log(basic.head(3).to_string())

    close = get_close_panel()
    log(f"场内收盘价面板: {close.shape} (日期 x 基金), 时间范围 {close.index.min()} ~ {close.index.max()}")
    sample = get_fund_nav_series("159915.SZ")  # 创业板ETF 示例
    log(f"159915.SZ 近5日: {sample.tail(5).to_dict()}")

    # 场外
    otc = load_otc_basic()
    log(f"场外基金列表: {len(otc)} 只")
    log(list_otc_funds("华夏成长").head(3).to_string())
    try:
        nav = load_otc_nav("000001")
        log(f"000001 净值序列: {len(nav)} 条, 最新: {nav.tail(1).to_dict()}")
    except FileNotFoundError as e:
        log(f"场外净值未就绪: {e} (等 fetch_fund2 抓取完成)")
