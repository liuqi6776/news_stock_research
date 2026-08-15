# -*- coding: utf-8 -*-
"""构建 Alpha101 (WorldQuant 101 Formulaic Alphas) 因子库

输入:
  - D:/iquant_data/data_v2/data_day1/*.parquet      日频行情 (全市场)
  - D:/iquant_data/data_v2/index_weight/*.parquet   中证1000 月度成分快照 (000852)
  - research/studies/study_008_enhancements/data/industry_map.parquet 行业映射

输出:
  - research/sector_rotation/alpha101_factor_panel.parquet
    列: [trade_date(月末), ts_code, industry, fwd_5, fwd_20, fwd_60, alpha_xxx...]

说明:
  - 只计算不依赖 market cap (cap) 和行业中性化 (IndNeutralize) 的 Alpha101 因子
  - vwap = 10 * amount / vol (amount单位千元, vol单位手 -> 元/股)
  - 因子只用 T 及之前数据; fwd_k = close[T+k]/close[T]-1 (前视标签, 仅用于 IC 验证)
  - 小数窗口因子(61+) 窗口四舍五入到整数, 因其多为 rank/布尔型对窗口不敏感
"""
import os
import glob
import time
import numpy as np
import pandas as pd

t0 = time.time()
ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
OUT = os.path.join(ROOT, "research", "sector_rotation", "alpha101_factor_panel.parquet")

# ---------- 1. 中证1000 成分历史 ----------
iw_files = sorted(glob.glob(os.path.join(DATA, "index_weight", "*.parquet")))
iw = pd.concat([pd.read_parquet(f) for f in iw_files], ignore_index=True)
iw = iw[iw["index_code"] == "000852.SH"].copy()
iw["iw_date"] = iw["trade_date"].astype(int)
iw = iw[["iw_date", "con_code"]].drop_duplicates()
iw_dates = sorted(iw["iw_date"].unique())
member_codes = set(iw["con_code"].unique())
print(f"[1] 中证1000成分: {len(iw_dates)} 期快照, {len(member_codes)} 只历史成分")

# ---------- 2. 行情 (2018-07 起, 留 ~1.5 年预热期给最长 250 日窗口) ----------
px_files = sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet")))
parts = []
for f in px_files:
    if os.path.getsize(f) <= 1024:
        continue
    d = os.path.basename(f)[:8]
    if d < "20180701":
        continue
    df = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low",
                                     "close", "vol", "amount"])
    df = df[df["ts_code"].isin(member_codes)]
    if len(df):
        parts.append(df)
px = pd.concat(parts, ignore_index=True)
px["trade_date"] = px["trade_date"].astype(int)
px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
print(f"[2] 行情面板: {len(px):,} 行, {px['ts_code'].nunique()} 只, "
      f"{px['trade_date'].min()}~{px['trade_date'].max()}, 耗时{time.time()-t0:.0f}s")

# ---------- 3. 基础字段 ----------
g = px.groupby("ts_code", sort=False)
px["returns"] = g["close"].pct_change()
px["vwap"] = np.where(px["vol"] > 0, px["amount"] / px["vol"] * 10.0, np.nan)
for d in (5, 10, 15, 20, 30, 40, 50, 60, 81, 120, 150, 180):
    px[f"adv{d}"] = g["amount"].transform(lambda x, dd=d: x.rolling(dd, min_periods=dd).mean())
print(f"[3] 基础字段完成, 耗时{time.time()-t0:.0f}s")

# 基础数组引用 (统一用 numpy array, 与 px 行序一致)
C = px["close"].to_numpy(float)
O = px["open"].to_numpy(float)
H = px["high"].to_numpy(float)
L = px["low"].to_numpy(float)
V = px["vol"].to_numpy(float)
R = px["returns"].to_numpy(float)
VW = px["vwap"].to_numpy(float)
ADV = {d: px[f"adv{d}"].to_numpy(float) for d in (5, 10, 15, 20, 30, 40, 50, 60, 81, 120, 150, 180)}
_code = px["ts_code"].to_numpy()
_td = px["trade_date"].to_numpy()

# ---------- 4. 算子库 (统一返回 numpy array) ----------
def _arr(x):
    return np.asarray(x, dtype=float)

def _ser(x):
    if isinstance(x, pd.Series):
        return x
    return pd.Series(_arr(x), index=px.index)

def _ts(s, fn):
    return _ser(s).groupby(_code, sort=False).transform(fn).to_numpy(float)

def ts_sum(s, d):
    return _ts(s, lambda x: x.rolling(d).sum())
def ts_mean(s, d):
    return _ts(s, lambda x: x.rolling(d).mean())
def ts_std(s, d):
    return _ts(s, lambda x: x.rolling(d).std())
def ts_min(s, d):
    return _ts(s, lambda x: x.rolling(d).min())
def ts_max(s, d):
    return _ts(s, lambda x: x.rolling(d).max())
def ts_rank(s, d):
    return _ts(s, lambda x: x.rolling(d).apply(
        lambda a: (a < a[-1]).mean() + 0.5 * (a == a[-1]).mean(), raw=True))
def ts_argmax(s, d):
    return _ts(s, lambda x: x.rolling(d).apply(
        lambda a: float(np.argmax(a)) + 1.0, raw=True))
def ts_argmin(s, d):
    return _ts(s, lambda x: x.rolling(d).apply(
        lambda a: float(np.argmin(a)) + 1.0, raw=True))
def decay_linear(s, d):
    w = np.arange(1, d + 1, dtype=float); w /= w.sum()
    return _ts(s, lambda x: x.rolling(d).apply(lambda a: float(np.dot(a, w)), raw=True))
def delay(s, d):
    return _ser(s).groupby(_code, sort=False).shift(d).to_numpy(float)
def delta(s, d):
    return _arr(s) - delay(s, d)

def _roll_corr_cov(x, y, d, kind):
    xs = pd.Series(_arr(x), index=px.index)
    ys = pd.Series(_arr(y), index=px.index)
    if kind == "corr":
        out = xs.groupby(_code, sort=False).transform(
            lambda s: s.rolling(d).corr(ys.reindex(s.index)))
    else:
        out = xs.groupby(_code, sort=False).transform(
            lambda s: s.rolling(d).cov(ys.reindex(s.index)))
    return out.to_numpy(float)

def corr(x, y, d):
    return _roll_corr_cov(x, y, d, "corr")
def cov(x, y, d):
    return _roll_corr_cov(x, y, d, "cov")

def _cs_rank(v):
    tmp = pd.DataFrame({"d": _td, "v": _arr(v)})
    return tmp.groupby("d")["v"].rank(pct=True).to_numpy(float)

def _cs_scale(v, a=1.0):
    tmp = pd.DataFrame({"d": _td, "v": _arr(v)})
    denom = tmp.groupby("d")["v"].transform(lambda x: x.abs().sum()).to_numpy(float)
    return (a * _arr(v) / np.where(denom == 0, 1.0, denom)).astype(float)

def rank(s):
    return _cs_rank(_arr(s))
def scale(s, a=1.0):
    return _cs_scale(_arr(s), a)

def _sign(s):
    return np.sign(_arr(s))
def _where(cond, a, b):
    return np.where(np.asarray(cond), _arr(a), _arr(b))

print(f"[4] 算子库就绪, 耗时{time.time()-t0:.0f}s")

# ---------- 5. Alpha101 因子 (全部输出 numpy array) ----------
F = {}
_dc1 = delta(C, 1)

# alpha_001: rank(ts_argmax(signedpower(where(returns<0, stddev(returns,20), close), 2), 5)) - 0.5
_std20 = ts_std(R, 20)
_sp1 = _sign(_where(R < 0, _std20, C)) * np.abs(_where(R < 0, _std20, C)) ** 2.0
F["alpha_001"] = rank(ts_argmax(_sp1, 5)) - 0.5

# alpha_002: -1 * correlation(rank(delta(log(volume),2)), rank((close-open)/open), 6)
_dv = np.log(np.where(V > 0, V, np.nan))
F["alpha_002"] = -corr(rank(delta(_dv, 2)), rank((C - O) / O), 6)

# alpha_003: -1 * correlation(rank(open), rank(volume), 10)
F["alpha_003"] = -corr(rank(O), rank(V), 10)

# alpha_004: -1 * ts_rank(rank(low), 9)
F["alpha_004"] = -ts_rank(rank(L), 9)

# alpha_005: rank((open - sum(vwap,10)/10)) * (-1 * abs(rank(close-vwap)))
F["alpha_005"] = rank(O - ts_mean(VW, 10)) * (-1 * np.abs(rank(C - VW)))

# alpha_006: -1 * correlation(open, volume, 10)
F["alpha_006"] = -corr(O, V, 10)

# alpha_007: where(adv20<volume, -1*ts_rank(abs(delta(close,7)),60)*sign(delta(close,7)), -1)
F["alpha_007"] = _where(ADV[20] < V, -ts_rank(np.abs(delta(C, 7)), 60) * _sign(delta(C, 7)), -1.0)

# alpha_008: -1 * rank((sum(open,5)*sum(returns,5)) - delay((sum(open,5)*sum(returns,5)),10))
_s5 = ts_sum(O, 5) * ts_sum(R, 5)
F["alpha_008"] = -rank(_s5 - delay(_s5, 10))

# alpha_009: where(0<ts_min(delta(close,1),5), delta(close,1), where(ts_max(delta(close,1),5)<0, delta(close,1), -1*delta(close,1)))
F["alpha_009"] = _where(ts_min(_dc1, 5) > 0, _dc1, _where(ts_max(_dc1, 5) < 0, _dc1, -_dc1))

# alpha_010: rank(alpha_009 with window 4)
F["alpha_010"] = rank(_where(ts_min(_dc1, 4) > 0, _dc1, _where(ts_max(_dc1, 4) < 0, _dc1, -_dc1)))

# alpha_011: (rank(ts_max(vwap-close,3)) + rank(ts_min(vwap-close,3))) * rank(delta(volume,3))
_vc = VW - C
F["alpha_011"] = (rank(ts_max(_vc, 3)) + rank(ts_min(_vc, 3))) * rank(delta(V, 3))

# alpha_012: sign(delta(volume,1)) * (-1 * delta(close,1))
F["alpha_012"] = _sign(delta(V, 1)) * (-_dc1)

# alpha_013: -1 * rank(covariance(rank(close), rank(volume), 5))
F["alpha_013"] = -rank(cov(rank(C), rank(V), 5))

# alpha_014: (-1 * rank(delta(returns,3))) * correlation(open, volume, 10)
F["alpha_014"] = (-rank(delta(R, 3))) * corr(O, V, 10)

# alpha_015: -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)
F["alpha_015"] = -ts_sum(rank(corr(rank(H), rank(V), 3)), 3)

# alpha_016: -1 * rank(covariance(rank(high), rank(volume), 5))
F["alpha_016"] = -rank(cov(rank(H), rank(V), 5))

# alpha_017: -rank(ts_rank(close,10)) * rank(delta(delta(close,1),1)) * rank(ts_rank(volume/adv20,5))
F["alpha_017"] = (-rank(ts_rank(C, 10)) * rank(delta(_dc1, 1)) * rank(ts_rank(V / ADV[20], 5)))

# alpha_018: -1 * rank(stddev(abs(close-open),5) + (close-open) + correlation(close,open,10))
F["alpha_018"] = -rank(ts_std(np.abs(C - O), 5) + (C - O) + corr(C, O, 10))

# alpha_019: -1*sign((close-delay(close,7))+delta(close,7)) * (1+rank(1+sum(returns,250)))
F["alpha_019"] = (-_sign((C - delay(C, 7)) + delta(C, 7)) * (1 + rank(1 + ts_sum(R, 250))))

# alpha_020: -rank(open-delay(high,1)) * rank(open-delay(close,1)) * rank(open-delay(low,1))
F["alpha_020"] = (-rank(O - delay(H, 1)) * rank(O - delay(C, 1)) * rank(O - delay(L, 1)))

# alpha_021: 均线突破方向
_s8 = ts_mean(C, 8); _s2 = ts_mean(C, 2); _std8 = ts_std(C, 8)
F["alpha_021"] = _where((_s8 + _std8) < _s2, -1.0, _where(_s2 < (_s8 - _std8), 1.0, _where((V / ADV[20]) >= 1.0, 1.0, -1.0)))

# alpha_022: -1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))
F["alpha_022"] = -delta(corr(H, V, 5), 5) * rank(ts_std(C, 20))

# alpha_023: where(sum(high,20)/20 < high, -1*delta(high,2), 0)
F["alpha_023"] = _where(ts_mean(H, 20) < H, -delta(H, 2), 0.0)

# alpha_024: 100日均线突破回撤
_d100 = delta(ts_mean(C, 100), 100) / delay(C, 100)
F["alpha_024"] = _where(_d100 <= 0.05, -(C - ts_min(C, 100)), -delta(C, 3))

# alpha_025: rank(-1*returns*adv20*vwap*(high-close))
F["alpha_025"] = rank((-R) * ADV[20] * VW * (H - C))

# alpha_026: -1 * ts_max(correlation(ts_rank(volume,5), ts_rank(high,5), 5), 3)
F["alpha_026"] = -ts_max(corr(ts_rank(V, 5), ts_rank(H, 5), 5), 3)

# alpha_027: where(0.5 < rank(sum(corr(rank(volume),rank(vwap),6),2)/2), -1, 1)
F["alpha_027"] = _where(rank(ts_mean(corr(rank(V), rank(VW), 6), 2)) > 0.5, -1.0, 1.0)

# alpha_028: scale(correlation(adv20, low, 5) + (high+low)/2 - close)
F["alpha_028"] = scale(corr(ADV[20], L, 5) + (H + L) / 2 - C)

# alpha_030: (1-rank(sign(...)+sign(...)+sign(...))) * sum(volume,5)/sum(volume,20)
_sgn = (_sign(C - delay(C, 1)) + _sign(delay(C, 1) - delay(C, 2)) + _sign(delay(C, 2) - delay(C, 3)))
F["alpha_030"] = (1 - rank(_sgn)) * (ts_sum(V, 5) / ts_sum(V, 20))

# alpha_031: rank(rank(rank(decay_linear(-1*rank(rank(delta(close,10))),10)))) + rank(-1*delta(close,3)) + sign(scale(corr(adv20,low,12)))
F["alpha_031"] = (rank(decay_linear(-rank(delta(C, 10)), 10)) + rank(-delta(C, 3))
                  + _sign(scale(corr(ADV[20], L, 12))))

# alpha_032: scale(sum(close,7)/7 - close) + 20*scale(correlation(vwap, delay(close,5), 230))
F["alpha_032"] = (scale(ts_mean(C, 7) - C) + 20 * scale(corr(VW, delay(C, 5), 230)))

# alpha_033: rank(-1 * (1 - open/close))
F["alpha_033"] = rank(-(1 - O / C))

# alpha_034: rank(1 - rank(stddev(returns,2)/stddev(returns,5)) + 1 - rank(delta(close,1)))
F["alpha_034"] = rank(1 - rank(ts_std(R, 2) / ts_std(R, 5)) + 1 - rank(_dc1))

# alpha_035: ts_rank(volume,32) * (1-ts_rank(close+high-low,16)) * (1-ts_rank(returns,32))
F["alpha_035"] = ts_rank(V, 32) * (1 - ts_rank(C + H - L, 16)) * (1 - ts_rank(R, 32))

# alpha_036: 综合加权
F["alpha_036"] = (2.21 * rank(corr(C - O, delay(V, 1), 15)) + 0.7 * rank(O - C)
                  + 0.73 * rank(ts_rank(delay(-R, 6), 5)) + rank(np.abs(corr(VW, ADV[20], 6)))
                  + 0.6 * rank((ts_mean(C, 200) - O) * (C - O)))

# alpha_037: rank(correlation(delay(open-close,1), close, 200)) + rank(open-close)
F["alpha_037"] = rank(corr(delay(O - C, 1), C, 200)) + rank(O - C)

# alpha_038: -1 * rank(ts_rank(close,10)) * rank(close/open)
F["alpha_038"] = -rank(ts_rank(C, 10)) * rank(C / O)

# alpha_039: -1*rank(delta(close,7)*(1-rank(decay_linear(volume/adv20,9)))) * (1+rank(sum(returns,250)))
F["alpha_039"] = (-rank(delta(C, 7) * (1 - rank(decay_linear(V / ADV[20], 9))))
                  * (1 + rank(ts_sum(R, 250))))

# alpha_040: -1 * rank(stddev(high,10)) * correlation(high, volume, 10)
F["alpha_040"] = -rank(ts_std(H, 10)) * corr(H, V, 10)

# alpha_041: (high*low)^0.5 - vwap
F["alpha_041"] = np.sqrt(H * L) - VW

# alpha_042: rank(vwap-close) / rank(vwap+close)
F["alpha_042"] = rank(VW - C) / (rank(VW + C) + 1e-9)

# alpha_043: ts_rank(volume/adv20,20) * ts_rank(-1*delta(close,7),8)
F["alpha_043"] = ts_rank(V / ADV[20], 20) * ts_rank(-delta(C, 7), 8)

# alpha_044: -1 * correlation(high, rank(volume), 5)
F["alpha_044"] = -corr(H, rank(V), 5)

# alpha_045: -1*(rank(sum(delay(close,5),20)/20) * correlation(close,volume,2) * rank(corr(sum(close,5),sum(close,20),2)))
F["alpha_045"] = (-rank(ts_mean(delay(C, 5), 20)) * corr(C, V, 2)
                  * rank(corr(ts_sum(C, 5), ts_sum(C, 20), 2)))

# alpha_046: 加速度方向
_acc = (delay(C, 20) - delay(C, 10)) / 10 - (delay(C, 10) - C) / 10
F["alpha_046"] = _where(_acc > 0.25, -1.0, _where(_acc < 0, 1.0, -1.0 * (C - delay(C, 1))))

# alpha_047: ((rank(1/close)*volume)/adv20) * ((high*rank(high-close))/mean(high,5)) - rank(vwap-delay(vwap,5))
F["alpha_047"] = ((rank(1 / C) * V / ADV[20]) * (H * rank(H - C) / ts_mean(H, 5))
                  - rank(VW - delay(VW, 5)))

# alpha_050: -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)
F["alpha_050"] = -ts_max(rank(corr(rank(V), rank(VW), 5)), 5)

# alpha_052: ((-ts_min(low,5)+delay(ts_min(low,5),5)) * rank((sum(returns,240)-sum(returns,20))/220)) * ts_rank(volume,5)
F["alpha_052"] = ((-ts_min(L, 5) + delay(ts_min(L, 5), 5)) * rank((ts_sum(R, 240) - ts_sum(R, 20)) / 220) * ts_rank(V, 5))

# alpha_053: -1 * delta((close-low)-(high-close)/(close-low), 9)
F["alpha_053"] = -delta((C - L) - (H - C) / (C - L + 1e-9), 9)

# alpha_054: -1 * (low-close)*(open^5) / ((low-high)*(close^5))
F["alpha_054"] = -(L - C) * (O ** 5) / ((L - H) * (C ** 5) + 1e-9)

# alpha_055: -1 * correlation(rank((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12))), rank(volume), 6)
F["alpha_055"] = -corr(rank((C - ts_min(L, 12)) / (ts_max(H, 12) - ts_min(L, 12) + 1e-9)), rank(V), 6)

# alpha_057: -1 * (close - vwap) / decay_linear(rank(ts_argmax(close,30)), 2)
F["alpha_057"] = -(C - VW) / (decay_linear(rank(ts_argmax(C, 30)), 2) + 1e-9)

# alpha_060: -1 * (2*scale(rank(((close-low)-(high-close))/(high-low)*volume)) - scale(rank(ts_argmax(close,10))))
_x60 = (C - L) - (H - C) / (H - L + 1e-9) * V
F["alpha_060"] = -(2 * scale(rank(_x60)) - scale(rank(ts_argmax(C, 10))))

# alpha_061: rank(vwap - ts_min(vwap,16)) < rank(correlation(vwap, adv180, 18))
F["alpha_061"] = (rank(VW - ts_min(VW, 16)) < rank(corr(VW, ADV[180], 18))).astype(float)

# alpha_062: (rank(corr(vwap, sum(adv20,22), 10)) < rank((rank(open)+rank(open)) < (rank((high+low)/2)+rank(high)))) * -1
F["alpha_062"] = (rank(corr(VW, ts_sum(ADV[20], 22), 10))
                  < rank((rank(O) + rank(O)) < (rank((H + L) / 2) + rank(H)))).astype(float) * -1

# alpha_064: (rank(corr(sum(open*0.1784+low*0.8216, 13), sum(adv120,13), 17)) < rank(delta((high+low)/2*0.1784+vwap*0.8216, 4))) * -1
F["alpha_064"] = (rank(corr(ts_sum(O * 0.178404 + L * (1 - 0.178404), 13), ts_sum(ADV[120], 13), 17))
                  < rank(delta((H + L) / 2 * 0.178404 + VW * (1 - 0.178404), 4))).astype(float) * -1

# alpha_065: (rank(corr(open*0.0082+vwap*0.9918, sum(adv60,9), 6)) < rank(open - ts_min(open,14))) * -1
F["alpha_065"] = (rank(corr(O * 0.00817205 + VW * (1 - 0.00817205), ts_sum(ADV[60], 9), 6))
                  < rank(O - ts_min(O, 14))).astype(float) * -1

# alpha_066: (rank(decay_linear(delta(vwap,4),7)) + ts_rank(decay_linear((low-vwap)/(open-(high+low)/2), 11), 7)) * -1
F["alpha_066"] = (rank(decay_linear(delta(VW, 4), 7))
                  + ts_rank(decay_linear((L - VW) / (O - (H + L) / 2 + 1e-9), 11), 7)) * -1

# alpha_068: (ts_rank(corr(rank(high), rank(adv15), 9), 14) < rank(delta(close*0.5184+low*0.4816, 1))) * -1
F["alpha_068"] = (ts_rank(corr(rank(H), rank(ADV[15]), 9), 14)
                  < rank(delta(C * 0.518371 + L * (1 - 0.518371), 1))).astype(float) * -1

# alpha_071: max(ts_rank(decay_linear(corr(ts_rank(close,3), ts_rank(adv180,12), 18), 4), 16), ts_rank(decay_linear(rank((low+open)-(vwap+vwap))^2, 16), 4))
F["alpha_071"] = np.maximum(
    ts_rank(decay_linear(corr(ts_rank(C, 3), ts_rank(ADV[180], 12), 18), 4), 16),
    ts_rank(decay_linear(rank((L + O) - 2 * VW) ** 2, 16), 4))

# alpha_072: rank(decay_linear(corr((high+low)/2, adv40, 9), 10)) / rank(decay_linear(corr(ts_rank(vwap,4), ts_rank(volume,19), 7), 3))
F["alpha_072"] = (rank(decay_linear(corr((H + L) / 2, ADV[40], 9), 10))
                  / (rank(decay_linear(corr(ts_rank(VW, 4), ts_rank(V, 19), 7), 3)) + 1e-9))

# alpha_073: max(rank(decay_linear(delta(vwap,5),3)), ts_rank(decay_linear(-delta(open*0.147+low*0.853,2)/(open*0.147+low*0.853), 3), 17)) * -1
_hl73 = O * 0.147155 + L * (1 - 0.147155)
F["alpha_073"] = np.maximum(
    rank(decay_linear(delta(VW, 5), 3)),
    ts_rank(decay_linear(-delta(_hl73, 2) / (_hl73 + 1e-9), 3), 17)) * -1

# alpha_074: (rank(corr(close, sum(adv30,37), 15)) < rank(corr(rank(high*0.026+vwap*0.974), rank(volume), 11))) * -1
F["alpha_074"] = (rank(corr(C, ts_sum(ADV[30], 37), 15))
                  < rank(corr(rank(H * 0.0261661 + VW * (1 - 0.0261661)), rank(V), 11))).astype(float) * -1

# alpha_075: rank(corr(vwap, volume, 4)) < rank(corr(rank(low), rank(adv50), 12))
F["alpha_075"] = (rank(corr(VW, V, 4)) < rank(corr(rank(L), rank(ADV[50]), 12))).astype(float)

# alpha_077: min(rank(decay_linear((high+low)/2+high-(vwap+high), 20)), rank(decay_linear(corr((high+low)/2, adv40, 3), 6)))
F["alpha_077"] = np.minimum(
    rank(decay_linear((H + L) / 2 - VW, 20)),
    rank(decay_linear(corr((H + L) / 2, ADV[40], 3), 6)))

# alpha_078: rank(corr(sum(low*0.352+vwap*0.648, 20), sum(adv40,20), 7)) ^ rank(corr(rank(vwap), rank(volume), 6))
F["alpha_078"] = (rank(corr(ts_sum(L * 0.352233 + VW * (1 - 0.352233), 20), ts_sum(ADV[40], 20), 7))
                  ** rank(corr(rank(VW), rank(V), 6)))

# alpha_081: (rank(log(product(rank(rank(corr(vwap, sum(adv10,50), 8))^4), 15))) < rank(corr(rank(vwap), rank(volume), 5))) * -1
_x81 = rank(corr(VW, ts_sum(ADV[10], 50), 8)) ** 4
F["alpha_081"] = (rank(ts_sum(np.log(np.clip(_x81, 1e-12, None)), 15))
                  < rank(corr(rank(VW), rank(V), 5))).astype(float) * -1

# alpha_083: (rank(delay((high-low)/(sum(close,5)/5), 2)) * rank(rank(volume))) / ((high-low)/(sum(close,5)/5)/(vwap-close))
_hl83 = (H - L) / ts_mean(C, 5)
F["alpha_083"] = (rank(delay(_hl83, 2)) * rank(rank(V))) / (_hl83 / (VW - C + 1e-9))

# alpha_084: signedpower(ts_rank(vwap - ts_max(vwap,15), 21), delta(close,5))
_x84 = ts_rank(VW - ts_max(VW, 15), 21)
F["alpha_084"] = np.sign(_x84) * np.abs(_x84) ** delta(C, 5)

# alpha_085: rank(corr(high*0.877+close*0.123, adv30, 10)) ^ rank(corr(ts_rank((high+low)/2,4), ts_rank(volume,10), 7))
F["alpha_085"] = (rank(corr(H * 0.876703 + C * (1 - 0.876703), ADV[30], 10))
                  ** rank(corr(ts_rank((H + L) / 2, 4), ts_rank(V, 10), 7)))

# alpha_086: (ts_rank(corr(close, sum(adv20,15), 6), 20) < rank(open+close-vwap-open)) * -1
F["alpha_086"] = (ts_rank(corr(C, ts_sum(ADV[20], 15), 6), 20) < rank(C - VW)).astype(float) * -1

# alpha_088: min(rank(decay_linear(rank(open)+rank(low)-rank(high)-rank(close), 8)), ts_rank(decay_linear(corr(ts_rank(close,8), ts_rank(adv60,21), 8), 7), 3))
F["alpha_088"] = np.minimum(
    rank(decay_linear(rank(O) + rank(L) - rank(H) - rank(C), 8)),
    ts_rank(decay_linear(corr(ts_rank(C, 8), ts_rank(ADV[60], 21), 8), 7), 3))

# alpha_092: min(ts_rank(decay_linear((high+low)/2+close < low+open, 15), 19), ts_rank(decay_linear(corr(rank(low), rank(adv30), 8), 7), 7))
F["alpha_092"] = np.minimum(
    ts_rank(decay_linear(((H + L) / 2 + C < L + O).astype(float), 15), 19),
    ts_rank(decay_linear(corr(rank(L), rank(ADV[30]), 8), 7), 7))

# alpha_094: (rank(vwap - ts_min(vwap,12)) ^ ts_rank(corr(ts_rank(vwap,20), ts_rank(adv60,4), 18), 3)) * -1
F["alpha_094"] = (rank(VW - ts_min(VW, 12))
                  ** ts_rank(corr(ts_rank(VW, 20), ts_rank(ADV[60], 4), 18), 3)) * -1

# alpha_095: rank(open - ts_min(open,12)) < ts_rank(rank(corr(sum((high+low)/2,19), sum(adv40,19), 13))^5, 12)
F["alpha_095"] = (rank(O - ts_min(O, 12))
                  < ts_rank(rank(corr(ts_sum((H + L) / 2, 19), ts_sum(ADV[40], 19), 13)) ** 5, 12)).astype(float)

# alpha_096: max(ts_rank(decay_linear(corr(rank(vwap), rank(volume), 4), 4), 8), ts_rank(decay_linear(ts_argmax(corr(ts_rank(close,7), ts_rank(adv60,4), 4), 13), 14), 13)) * -1
F["alpha_096"] = np.maximum(
    ts_rank(decay_linear(corr(rank(VW), rank(V), 4), 4), 8),
    ts_rank(decay_linear(ts_argmax(corr(ts_rank(C, 7), ts_rank(ADV[60], 4), 4), 13), 14), 13)) * -1

# alpha_098: rank(decay_linear(corr(vwap, sum(adv5,26), 5), 7)) - rank(decay_linear(ts_rank(ts_argmin(corr(rank(open), rank(adv15), 21), 9), 7), 8))
F["alpha_098"] = (rank(decay_linear(corr(VW, ts_sum(ADV[5], 26), 5), 7))
                  - rank(decay_linear(ts_rank(ts_argmin(corr(rank(O), rank(ADV[15]), 21), 9), 7), 8)))

# alpha_099: (rank(corr(sum((high+low)/2,20), sum(adv60,20), 9)) < rank(corr(low, volume, 6))) * -1
F["alpha_099"] = (rank(corr(ts_sum((H + L) / 2, 20), ts_sum(ADV[60], 20), 9))
                  < rank(corr(L, V, 6))).astype(float) * -1

# alpha_101: (close-open)/((high-low)+0.001)
F["alpha_101"] = (C - O) / (H - L + 0.001)

print(f"[5] Alpha101 因子计算完成: {len(F)} 个, 耗时{time.time()-t0:.0f}s")

# ---------- 6. 前视标签 (fwd_5/20/60) ----------
px["fwd_5"] = (g["close"].shift(-5) / C - 1)
px["fwd_20"] = (g["close"].shift(-20) / C - 1)
px["fwd_60"] = (g["close"].shift(-60) / C - 1)

# ---------- 7. 月末快照 + 成分对齐 (无前视) ----------
cal = sorted(px["trade_date"].unique())
cal_s = pd.Series(cal)
month_last = cal_s.groupby(cal_s // 100).max().tolist()
month_last = [d for d in month_last if 20191201 <= d <= 20260630]
print(f"[6] 月末快照: {len(month_last)} 个月 ({month_last[0]}~{month_last[-1]})")

iw_by_date = {d: set(g2["con_code"]) for d, g2 in iw.groupby("iw_date")}

def latest_members(rebal_d):
    for d in reversed(iw_dates):
        if d <= rebal_d:
            return iw_by_date[d]
    return set()

factor_cols = list(F.keys())
snap_cols = ["trade_date", "ts_code", "fwd_5", "fwd_20", "fwd_60"] + factor_cols
snap = px[["trade_date", "ts_code", "fwd_5", "fwd_20", "fwd_60"]].copy()
for name in factor_cols:
    snap[name] = F[name]
snap = snap[snap["trade_date"].isin(month_last)].copy()

keep = []
for d in month_last:
    members = latest_members(d)
    sub = snap[snap["trade_date"] == d]
    keep.append(sub[sub["ts_code"].isin(members)])
snap = pd.concat(keep, ignore_index=True)

# ---------- 8. 行业映射 ----------
im = pd.read_parquet(os.path.join(ROOT, "research", "studies", "study_008_enhancements",
                                  "data", "industry_map.parquet"))
ind_map = dict(zip(im["ts_code"], im["industry"]))
snap["industry"] = snap["ts_code"].map(ind_map).fillna("其他")

# ---------- 9. 输出 ----------
out_cols = ["trade_date", "ts_code", "industry", "fwd_5", "fwd_20", "fwd_60"] + factor_cols
snap = snap[out_cols].copy()
snap = snap.dropna(subset=["fwd_20"])
snap.to_parquet(OUT, index=False)
print(f"\n[7] 因子面板保存: {OUT}")
print(f"    行数: {len(snap):,}, 月份: {snap['trade_date'].nunique()}, "
      f"股票: {snap['ts_code'].nunique()}, 因子: {len(factor_cols)}, 耗时{time.time()-t0:.0f}s")
