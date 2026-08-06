# exp_factor_multiplicity — 21 因子多重检验控制（审查 P0-3）

## 运行

```bash
C:\Users\liuqi\anaconda3\python.exe research/experiments/exp_factor_multiplicity/run.py
```

输出 `multiplicity_report.json` + 控制台表格。**分析型实验**（无 expected 对比）。

## 内容与结果（2026-08-06）

1. **BH-FDR / Bonferroni**：21 因子 NW t → p → 校正。
   - BH-FDR 显著 **9/21**：turnover_vol_20、csad_std_21、volume_surge_vol、loud_vol、csad_ratio_20_120、moderate_loud_vol、lg_net_5d、illiq_money_20(反向)、net_support_vol(反向)
   - Bonferroni 显著 **6/21**：turnover_vol_20、csad_std_21、volume_surge_vol、loud_vol、csad_ratio_20_120、illiq_money_20(反向)
2. **NW lag 敏感性**（lag=0/4/19 HAC t 值）：turnover_vol_20 t = 6.66→7.17→8.51，对 lag 稳健。
3. **IC 自相关诊断（Ljung-Box）**：
   - turnover_vol_20：LB p=0.72/0.95/0.67 → 无显著自相关，**lag=4 充分**；
   - csad_std_21 / volume_surge_vol / loud_vol / net_mf_5d：LB p<0.05 → IC 序列强自相关，**lag=4 校正不足，t 值被高估**。
4. **DSR（Bailey-López de Prado 2014, N=21 trials）**：turnover_vol_20 IC-SR≈0.75 → **DSR=1.000（通过）**。
   ⚠️ DSR 严格用于收益序列，此处以月频 IC 序列近似，仅作量级参考。

## 结论

- **多重检验不构成对 turnover_vol_20 IC 信号的威胁**（校正后仍 p<1e-11，lag 稳健、无自相关）；
- 但注意：N=21 仅覆盖"因子 IC 检验"这一层选择自由度；因子组合/风控参数/打分方式等其他研究者自由度未纳入；
- csad_std_21 等强 IC 自相关因子的 t 值需更保守 lag（≥19）重新评估，当前 NW t=10.88 不可直接引用。
