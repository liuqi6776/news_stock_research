# -*- coding: utf-8 -*-
import pandas as pd, os
df = pd.read_parquet(r"D:\iquant_data\data_v2\fund2\fund_basic_O.parquet")
NAV_DIR = r"D:\iquant_data\data_v2\fund2\nav"
known = ["000834","050025","000071","000948","000216","002610","100032",
         "050002","000051","110020","160119","002683","270042","161125",
         "004098","160719","161226","501018","162411","161715","161724",
         "160222","165520","168203","003646","080012","004998","100051",
         "001512","000015","005051","000312","000478","001879","165519",
         "161024","164905","001469","161211"]
for c in known:
    p = os.path.join(NAV_DIR, c + ".parquet")
    row = df[df["code"] == c]
    name = row["name"].values[0] if len(row) else "?"
    ftype = row["fund_type"].values[0] if len(row) else "?"
    if os.path.exists(p):
        d = pd.read_parquet(p, columns=["date"])
        dmin = d["date"].min()
        dmax = d["date"].max()
        n = len(d)
        print(f"{c} {name[:22]:22s} {ftype:14s} {dmin}~{dmax} {n}")
    else:
        print(f"{c} {name[:22]:22s} {ftype:14s} NO_FILE")
