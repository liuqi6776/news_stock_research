# -*- coding: utf-8 -*-
"""搜索各板块代表性基金, 找数据最长的那只"""
import pandas as pd, os

df = pd.read_parquet(r"D:\iquant_data\data_v2\fund2\fund_basic_O.parquet")
NAV_DIR = r"D:\iquant_data\data_v2\fund2\nav"

keywords = {
    "纯债":     ["纯债"],
    "红利":     ["红利"],
    "沪深300":  ["沪深300"],
    "中证500":  ["中证500"],
    "创业板":   ["创业板"],
    "科创50":   ["科创50"],
    "恒生指数": ["恒生指数"],
    "恒生科技": ["恒生科技"],
    "纳指":     ["纳指"],
    "标普":     ["标普500"],
    "黄金":     ["黄金"],
    "消费":     ["消费"],
    "医药":     ["医药"],
    "医疗":     ["医疗"],
    "军工":     ["军工"],
    "新能源":   ["新能源"],
    "半导体":   ["半导体"],
    "芯片":     ["芯片"],
    "金融":     ["金融"],
    "券商":     ["券商"],
    "地产":     ["地产"],
    "原油":     ["原油"],
    "油气":     ["油气"],
    "可转债":   ["可转债"],
    "国债":     ["国债"],
    "REITs":    ["REIT"],
}

idx = df[df["fund_type"].str.contains("指数", na=False)]
print(f"指数型基金总数: {len(idx)}\n")

for cat, kws in keywords.items():
    mask = idx["name"].str.contains("|".join(kws), na=False)
    mask = mask & ~idx["name"].str.contains("C|后端|D|E|I|H", na=False)
    sub = idx[mask].copy()
    if len(sub) == 0:
        mask2 = df["name"].str.contains("|".join(kws), na=False) & ~df["name"].str.contains("C|后端|D|E|I|H", na=False)
        sub = df[mask2].copy()
    if len(sub) == 0:
        print(f"{cat:8s}: 无匹配")
        continue
    hits = []
    for _, row in sub.head(30).iterrows():
        p = os.path.join(NAV_DIR, f"{row['code']}.parquet")
        if os.path.exists(p):
            try:
                d = pd.read_parquet(p, columns=["date"])
                if len(d) > 500:
                    hits.append((row["code"], row["name"], d["date"].min(), d["date"].max(), len(d)))
            except:
                pass
    if hits:
        hits.sort(key=lambda x: x[4], reverse=True)
        top = hits[0]
        print(f"{cat:8s}: {top[0]} {top[1][:25]:25s} 起{top[2]} 止{top[3]} {top[4]}条")
    else:
        print(f"{cat:8s}: 有名称匹配但无数据")
