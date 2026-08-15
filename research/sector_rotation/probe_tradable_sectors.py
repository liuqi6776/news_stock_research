# -*- coding: utf-8 -*-
"""盘点可交易板块池:
  1. 从fund_basic筛出 指数型-股票 基金 (行业ETF/联接/指数基金)
  2. 按名字关键词分到板块
  3. 检查每个板块的基金数、净值覆盖、代表基金
  4. 映射板块→细分行业(用industry_pe.csv的行业名匹配关键词)
"""
import os, json
import pandas as pd

DATA = r"D:\iquant_data\data_v2"
ROOT = r"c:\Users\liuqi\quant_system_v2"
NAV_DIR = os.path.join(DATA, "fund2", "nav")

fb = pd.read_parquet(os.path.join(DATA, "fund2", "fund_basic_O.parquet"))
print(f"总基金: {len(fb)}")
idx_equity = fb[fb["fund_type"] == "指数型-股票"].copy()
print(f"指数型-股票: {len(idx_equity)}")

# 板块关键词 → 板块名
SECT_KW = {
    "医药医疗": ["医药", "医疗", "生物", "创新药", "中药", "医疗器械", "医美"],
    "白酒消费": ["白酒", "食品", "饮料", "消费", "酒"],
    "银行": ["银行", "银行ETF"],
    "证券保险": ["证券", "券商", "保险", "非银"],
    "地产": ["地产", "房地产", "房地产"],
    "煤炭": ["煤炭"],
    "钢铁": ["钢铁"],
    "有色金属": ["有色", "黄金", "稀土", "矿业"],
    "石油石化": ["石油", "石化", "油气"],
    "化工": ["化工"],
    "电力": ["电力", "绿电"],
    "公用事业": ["公用事业"],
    "新能源": ["新能源", "光伏", "风电", "电池", "储能", "锂电", "氢能"],
    "半导体芯片": ["半导体", "芯片", "集成电路"],
    "电子": ["电子", "消费电子"],
    "计算机软件": ["计算机", "软件", "云计算", "大数据", "信创"],
    "通信": ["通信", "5G"],
    "传媒": ["传媒", "游戏", "影视"],
    "人工智能": ["人工智能", "AI", "机器人"],
    "军工": ["军工", "国防", "航天", "航空"],
    "汽车": ["汽车"],
    "家电": ["家电"],
    "建材": ["建材", "水泥"],
    "建筑": ["建筑"],
    "机械": ["机械", "智能制造"],
    "农业": ["农业", "养殖", "畜牧", "种业", "种植"],
    "纺织服装": ["纺织", "服装"],
    "交通运输": ["交通运输", "物流", "港口"],
    "环保": ["环保"],
    "基建": ["基建"],
    "一带一路": ["一带一路"],
}
# 反向: 行业名 → 板块
rev = {}
for sect, kws in SECT_KW.items():
    for kw in kws:
        rev.setdefault(kw, sect)

# 给基金打板块标签 (一个基金可能多个板块, 取第一个命中)
def tag_fund(name):
    hits = [sect for kw, sect in rev.items() if kw in name]
    return hits[0] if hits else None

idx_equity["sector"] = idx_equity["name"].apply(tag_fund)
tagged = idx_equity[idx_equity["sector"].notna()]
print(f"能匹配到板块的指数基金: {len(tagged)} / {len(idx_equity)}")

# 检查净值覆盖
nav_files = set(os.listdir(NAV_DIR))
def nav_len(code):
    fp = f"{code}.parquet"
    if fp in nav_files:
        import pandas as pd
        try:
            return len(pd.read_parquet(os.path.join(NAV_DIR, fp)))
        except Exception:
            return 0
    return 0

# 按板块统计: 基金数, 有净值数, 最早净值
sect_stats = {}
for sect, g in tagged.groupby("sector"):
    codes = g["code"].astype(str).tolist()
    has_nav = [c for c in codes if f"{c}.parquet" in nav_files]
    sect_stats[sect] = {
        "funds": len(codes), "with_nav": len(has_nav),
        "sample": g.sort_values("code")["code"].astype(str).iloc[0] if len(g) else None,
        "sample_name": g.sort_values("code")["name"].iloc[0] if len(g) else None,
    }
    # 检查样本净值长度
    if has_nav:
        import pandas as pd
        first = has_nav[0]
        df = pd.read_parquet(os.path.join(NAV_DIR, f"{first}.parquet"))
        sect_stats[sect]["first_date"] = str(df["date"].iloc[0]) if len(df) else None
        sect_stats[sect]["last_date"] = str(df["date"].iloc[-1]) if len(df) else None
        sect_stats[sect]["rows"] = len(df)

print("\n=== 可交易板块池 (指数型股票基金覆盖) ===")
print(f"{'板块':<12}{'基金数':<6}{'有净值':<6}{'样本基金':<30}{'净值起始':<12}")
for sect, st in sorted(sect_stats.items()):
    print(f"{sect:<12}{st['funds']:<6}{st['with_nav']:<6}"
          f"{str(st['sample_name'])[:26]:<30}{st.get('first_date',''):<12}")

# 保存板块池
with open(os.path.join(ROOT, "research", "sector_rotation", "results", "tradable_sector_pool.json"), "w", encoding="utf-8") as f:
    json.dump(sect_stats, f, ensure_ascii=False, indent=2, default=str)
print(f"\n已保存 tradable_sector_pool.json")
