# -*- coding: utf-8 -*-
"""第二轮搜索: 含QDII/非指数型, 要求2018前有数据"""
import pandas as pd, os

df = pd.read_parquet(r"D:\iquant_data\data_v2\fund2\fund_basic_O.parquet")
NAV_DIR = r"D:\iquant_data\data_v2\fund2\nav"

keywords = {
    "恒生指数":   ["恒生指数"],
    "恒生科技":   ["恒生科技"],
    "港股":       ["港股"],
    "纳指":       ["纳指"],
    "标普":       ["标普"],
    "黄金":       ["黄金"],
    "白银":       ["白银"],
    "原油":       ["原油"],
    "油气":       ["油气"],
    "商品":       ["商品"],
    "大宗":       ["大宗"],
    "REITs":      ["REIT"],
    "银行":       ["银行"],
    "煤炭":       ["煤炭"],
    "能源":       ["能源"],
    "环保":       ["环保"],
    "传媒":       ["传媒"],
    "食品":       ["食品"],
    "农业":       ["农业"],
    "有色":       ["有色"],
    "钢铁":       ["钢铁"],
    "基建":       ["基建"],
    "中证1000":   ["中证1000"],
    "国证2000":   ["国证2000"],
    "价值":       ["价值"],
    "信息":       ["信息"],
    "科技":       ["科技"],
    "QDII债":     ["QDII.*债|境外债|全球债"],
}

# 全类型搜索, 排除C/D/E/I/H/后端
for cat, kws in keywords.items():
    pat = "|".join(kws)
    mask = df["name"].str.contains(pat, na=False, regex=True)
    mask = mask & ~df["name"].str.contains("C|后端|D|E|I|H|B", na=False)
    sub = df[mask].copy()
    if len(sub) == 0:
        print(f"{cat:8s}: 无匹配")
        continue
    hits = []
    for _, row in sub.head(40).iterrows():
        p = os.path.join(NAV_DIR, f"{row['code']}.parquet")
        if os.path.exists(p):
            try:
                d = pd.read_parquet(p, columns=["date"])
                if len(d) > 500:
                    dmin = pd.to_datetime(d["date"]).min()
                    if dmin < pd.Timestamp("2018-01-01"):
                        hits.append((row["code"], row["name"][:25], d["date"].min(), d["date"].max(), len(d)))
            except:
                pass
    if hits:
        hits.sort(key=lambda x: x[4], reverse=True)
        top = hits[0]
        print(f"{cat:8s}: {top[0]} {top[1]:25s} 起{top[2]} {top[4]}条")
    else:
        print(f"{cat:8s}: 有匹配但无2018前数据")
