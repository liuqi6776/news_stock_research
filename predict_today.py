import pandas as pd
import numpy as np

PRED_FILE = 'research/study_004_1d_release/predictions/predictions_1d_open_wf_monthly.parquet'
df = pd.read_parquet(PRED_FILE)
df['trade_date'] = df['trade_date'].astype(str)

# 新闻中提到的所有股票代码（A股格式）
news_map = {
    '002491.SZ': '通鼎互联-光通信3板',
    '603738.SH': '泰晶科技-光通信3板',
    '002281.SZ': '光迅科技-光通信2板',
    '600184.SH': '光电股份-光通信1板',
    '603920.SH': '红板科技-PCB3板',
    '002543.SZ': '山东玻纤-PCB3板',
    '600531.SH': '豫光金铅-白银',
    '002716.SZ': '湖南白银-白银',
    '600312.SH': '白银有色-白银',
    '000612.SZ': '盛达资源-白银',
    '000878.SZ': '北方铜业-铜',
    '600362.SH': '江西铜业-铜',
    '002733.SZ': '雄韬股份-氢能源',
    '600860.SH': '京城股份-氢能源',
    '688339.SH': '亿华通-氢能源',
    '601991.SH': '大唐发电-算电协同4板',
    '600584.SH': '长电科技-封测',
    '300302.SZ': '同有科技-存储芯片',
    '600599.SH': '钧达股份-太空光伏2板',
    '603667.SH': '五洲新春-特斯拉机器人',
    '002236.SZ': '大华股份-特斯拉机器人',
    '601127.SH': '小康股份-特斯拉机器人',
    '002049.SZ': '紫光国微-国产算力',
    '300162.SZ': '中化岩土-卫星',
    '688599.SH': '天合光能-异质结',
    '300118.SZ': '东方日升-异质结',
    '600104.SH': '上汽集团-特朗普访华',
    '601138.SH': '工业富联-苹果',
    '002491.SZ': '通鼎互联-光通信',
    '000908.SZ': '红板科技-PCB',
    '600355.SH': '精伦电子-PCB',
    '002092.SZ': '中化岩土-算力',
    '603629.SH': '利通电子-算力',
    '002733.SZ': '雄韬股份-氢能源',
    '688256.SH': '寒武纪-国产算力',
    '002512.SZ': '达安基因-汉坦病毒2板',
    '600640.SH': '号百控股-卫星',
    '002230.SZ': '科大讯飞-AI',
    '002415.SZ': '海康威视-苹果',
    '603501.SH': '韦尔股份-半导体',
}

# 取最近5个交易日的数据，看模型对这些新闻股票的近期预测趋势
recent_dates = sorted(df['trade_date'].unique())[-5:]
print(f'最近5个交易日: {recent_dates}')
print()

# 新闻股票在最近预测中的表现
print('=' * 80)
print('新闻热点股票 - 模型近期预测概率')
print('=' * 80)

results = []
for code, desc in sorted(news_map.items(), key=lambda x: x[0]):
    stock_data = df[df['ts_code'] == code]
    if len(stock_data) == 0:
        results.append({'ts_code': code, 'desc': desc, 'latest_prob': None, 'trend': '无数据'})
        continue
    
    recent = stock_data[stock_data['trade_date'].isin(recent_dates)].sort_values('trade_date')
    if len(recent) == 0:
        results.append({'ts_code': code, 'desc': desc, 'latest_prob': None, 'trend': '无近期数据'})
        continue
    
    latest_prob = recent['prob'].values[-1]
    avg_prob = recent['prob'].mean()
    trend = '↑' if len(recent) > 1 and recent['prob'].values[-1] > recent['prob'].values[0] else '↓'
    
    results.append({
        'ts_code': code, 
        'desc': desc, 
        'latest_prob': latest_prob,
        'avg_prob': avg_prob,
        'trend': trend,
        'n_recent': len(recent)
    })

# 按prob排序
results_valid = [r for r in results if r['latest_prob'] is not None]
results_invalid = [r for r in results if r['latest_prob'] is None]
results_valid.sort(key=lambda x: x['latest_prob'], reverse=True)

print(f'\n有预测数据的股票 ({len(results_valid)}只):')
print(f'{"代码":12s} {"描述":20s} {"最新prob":>8s} {"5日均prob":>10s} {"趋势":>4s}')
print('-' * 60)
for r in results_valid:
    marker = ' ★' if r['latest_prob'] >= 0.50 else ''
    print(f'{r["ts_code"]:12s} {r["desc"]:20s} {r["latest_prob"]:8.4f} {r["avg_prob"]:10.4f} {r["trend"]:>4s}{marker}')

print(f'\n无预测数据的股票 ({len(results_invalid)}只):')
for r in results_invalid:
    print(f'  {r["ts_code"]:12s} {r["desc"]:20s} {r["trend"]}')

# 综合推荐：prob>=0.50的新闻股票
print('\n' + '=' * 80)
print('综合推荐：模型prob >= 0.50 的新闻热点股票')
print('=' * 80)
recommended = [r for r in results_valid if r['latest_prob'] >= 0.50]
if recommended:
    for r in recommended:
        print(f'  {r["ts_code"]:12s} {r["desc"]:20s} prob={r["latest_prob"]:.4f} {r["trend"]}')
else:
    print('  无满足条件的股票')

# 如果没有满足条件的，显示prob最高的5只
if not recommended:
    print('\n无新闻股票满足prob>=0.50，显示prob最高的5只:')
    for r in results_valid[:5]:
        print(f'  {r["ts_code"]:12s} {r["desc"]:20s} prob={r["latest_prob"]:.4f} {r["trend"]}')

# 最新日期模型选出的top10（不限新闻股票）
print('\n' + '=' * 80)
print(f'模型最新日期({recent_dates[-1]}) Top 10 高概率股票')
print('=' * 80)
latest_sub = df[df['trade_date'] == recent_dates[-1]].nlargest(10, 'prob')
for _, r in latest_sub.iterrows():
    code = r['ts_code']
    desc = news_map.get(code, '')
    extra = f' ({desc})' if desc else ''
    print(f'  {code:12s} prob={r["prob"]:.4f}{extra}')
