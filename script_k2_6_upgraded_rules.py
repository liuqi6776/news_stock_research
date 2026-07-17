#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案B：基于K2.6分析经验的升级版规则模型
相比旧版，新增以下关键改进：
1. 辟谣/澄清类新闻 = 利空（而非中性）
2. 高管减持 > 大股东减持（更负面）
3. ST股复牌 = 利空
4. 蹭热点澄清 = 强利空
5. 业绩超预期幅度分级
6. 区分"产品涨价"和"原材料涨价"
7. 引入事件强度乘数
8. 引入"情感极性反转"检测
"""

import re
import pandas as pd

class K2_6_NewsAnalyzer:
    """基于K2.6分析经验的金融新闻情感分析器"""
    
    def __init__(self):
        # ========== 事件类型词典（扩展版）==========
        self.event_types = {
            '业绩超预期': {
                'keywords': ['业绩超预期', '净利润大增', '营收翻倍', '盈利翻倍', '业绩爆发', '业绩大幅改善', '扭亏为盈', '利润创新高', '营收创新高'],
                'multiplier': 1.5,
                'base_score': 2
            },
            '业绩小幅增长': {
                'keywords': ['营收增长', '净利润增长', '盈利增长', '业绩改善', '营收增加'],
                'multiplier': 1.0,
                'base_score': 1
            },
            '业绩暴雷': {
                'keywords': ['业绩暴雷', '净利润下滑', '亏损', '业绩不及预期', '营收下降', '利润下降', '业绩预亏', '业绩变脸', '净利润同比降'],
                'multiplier': 2.0,
                'base_score': -2
            },
            '并购重组': {
                'keywords': ['并购重组', '收购', '合并', '借壳', '资产注入', '股权收购', '战略重组'],
                'multiplier': 2.0,
                'base_score': 2
            },
            '技术突破': {
                'keywords': ['技术突破', '获得专利', '新产品发布', '技术领先', '研发成功', '量产', '商业化'],
                'multiplier': 1.5,
                'base_score': 2
            },
            '政策利好': {
                'keywords': ['政策支持', '补贴', '税收优惠', '政策利好', '国家战略', '纳入规划', '重点扶持', '放开限制'],
                'multiplier': 1.5,
                'base_score': 1.5
            },
            '政策限制': {
                'keywords': ['政策收紧', '限制', '监管加强', '规范', '整顿', '叫停', '禁止'],
                'multiplier': 1.5,
                'base_score': -1.5
            },
            '高管减持': {
                'keywords': ['高管减持', '董监高减持', '董事长减持', '总经理减持', '副总经理减持'],
                'multiplier': 1.8,
                'base_score': -2
            },
            '大股东减持': {
                'keywords': ['大股东减持', '控股股东减持', '实控人减持', '减持计划', '拟减持'],
                'multiplier': 1.5,
                'base_score': -1.5
            },
            '增持/回购': {
                'keywords': ['大股东增持', '回购', '员工持股', '股权激励', '增持计划'],
                'multiplier': 1.0,
                'base_score': 1.5
            },
            '违规/处罚': {
                'keywords': ['违规', '处罚', '立案调查', '行政处罚', '监管', '证监会', '警示函', '通报批评', '罚款'],
                'multiplier': 1.8,
                'base_score': -2
            },
            '产品/安全': {
                'keywords': ['召回', '停产', '事故', '安全隐患', '质量问题', '产品缺陷', '下架'],
                'multiplier': 1.5,
                'base_score': -1.5
            },
            '诉讼/仲裁': {
                'keywords': ['诉讼', '仲裁', '赔偿', '败诉', '知识产权纠纷', '侵权'],
                'multiplier': 1.2,
                'base_score': -1
            },
            '退市/破产': {
                'keywords': ['退市', 'ST', '*ST', '破产', '债务违约', '资金链断裂', '重整'],
                'multiplier': 2.5,
                'base_score': -3
            },
            '订单/合同': {
                'keywords': ['签订大单', '订单饱满', '中标', '合同', '框架协议', '战略合作', '供不应求'],
                'multiplier': 1.2,
                'base_score': 1.5
            },
            '产品价格下跌': {
                'keywords': ['产品降价', '价格战', '价格下跌', '售价下调'],
                'multiplier': 1.2,
                'base_score': -1
            },
            '原材料涨价': {
                'keywords': ['原材料涨价', '成本上升', '原材料价格上涨', '大宗商品涨价'],
                'multiplier': 1.2,
                'base_score': -1
            },
            '产品涨价': {
                'keywords': ['产品涨价', '提价', '售价上调', '价格上涨'],
                'multiplier': 1.2,
                'base_score': 1.5
            },
        }
        
        # ========== K2.6关键规则：特殊场景处理 ==========
        self.special_rules = {
            '辟谣': {
                'patterns': ['谣言', '不实消息', '辟谣', '澄清', '假的', '网传', '不属实'],
                'score_adjust': -1.0,  # 辟谣=利空（市场已price in谣言）
                'reason': '辟谣类新闻通常意味着市场已有负面预期，辟谣只是确认没有额外利好'
            },
            '澄清蹭热点': {
                'patterns': ['未与', '无业务', '无合作关系', '未签署', '未涉及', '澄清说明'],
                'score_adjust': -2.0,  # 澄清蹭热点=强利空（证伪）
                'reason': '澄清蹭热点属于证伪，短期炒作资金会撤离'
            },
            'ST复牌': {
                'patterns': ['ST', '*ST', '复牌', '核查完成', '停牌', '核查'],
                'score_adjust': -1.5,
                'reason': 'ST股停牌核查后复牌通常补跌'
            },
            '高管变动负面': {
                'patterns': ['董事长辞职', '总经理辞职', '高管辞职', '实控人变更', '失联'],
                'score_adjust': -1.5,
                'reason': '高管离职+负面关键词=利空'
            },
            '业绩超预期幅度': {
                'patterns': ['增长超', '大增', '翻倍', '增长.*%'],
                'bonus_threshold': {
                    '50': 0.5,   # 增长50%以上额外+0.5
                    '100': 1.0,  # 增长100%以上额外+1.0
                    '200': 1.5,  # 增长200%以上额外+1.5
                }
            },
        }
        
        # 否定词列表
        self.negation_words = ['不', '未', '无', '非', '否', '没有', '未能', '不及', '不是', '并未', '尚无', '别', '勿', '莫', '难以', '难']
        
        # 转折词（降低极端分数）
        self.turn_words = ['但', '但是', '然而', '不过', '却', '反而', '尽管', '虽然']
    
    def analyze(self, title, content):
        """分析单条新闻"""
        text = f"{title} {content}"
        
        score = 0
        event_type = None
        max_confidence = 0
        reasons = []
        
        # 1. 事件类型检测
        for etype, config in self.event_types.items():
            count = 0
            for kw in config['keywords']:
                if kw in text:
                    count += 1
            
            if count > 0:
                event_score = config['base_score'] * config['multiplier'] * count
                score += event_score
                
                confidence = count * config['multiplier']
                if confidence > max_confidence:
                    max_confidence = confidence
                    event_type = etype
                
                reasons.append(f"{etype}: {event_score:+.1f}")
        
        # 2. 特殊规则处理（K2.6核心改进）
        for rule_name, rule_config in self.special_rules.items():
            if rule_name == '业绩超预期幅度':
                continue  # 已在事件类型中处理
            
            matched = False
            for pattern in rule_config['patterns']:
                if pattern in text:
                    matched = True
                    break
            
            if matched:
                score += rule_config['score_adjust']
                reasons.append(f"{rule_name}: {rule_config['score_adjust']:+.1f}")
        
        # 3. 否定词处理
        for neg in self.negation_words:
            for pos_word in ['涨', '突破', '利好', '增长', '上升', '上涨']:
                if neg + pos_word in text:
                    score -= 1.0
                    reasons.append(f"否定词反转({neg}{pos_word}): -1.0")
            for neg_word in ['跌', '下滑', '利空', '下降', '下跌']:
                if neg + neg_word in text:
                    score += 1.0
                    reasons.append(f"否定词反转({neg}{neg_word}): +1.0")
        
        # 4. 转折词处理
        for turn in self.turn_words:
            if turn in text:
                score *= 0.7
                reasons.append(f"转折词({turn}): ×0.7")
                break
        
        # 5. 业绩超预期幅度检测
        if event_type == '业绩超预期':
            # 尝试提取百分比
            import re
            pct_matches = re.findall(r'增长(\d+)%', text)
            if pct_matches:
                pct = int(pct_matches[0])
                if pct >= 200:
                    score += 1.5
                    reasons.append(f"业绩超预期200%+: +1.5")
                elif pct >= 100:
                    score += 1.0
                    reasons.append(f"业绩超预期100%+: +1.0")
                elif pct >= 50:
                    score += 0.5
                    reasons.append(f"业绩超预期50%+: +0.5")
        
        # 6. 分数限制
        score = max(-3, min(3, score))
        
        return {
            'score': round(score, 2),
            'event_type': event_type or '其他',
            'reason': '; '.join(reasons) if reasons else '无明显事件',
            'confidence': min(max_confidence, 3.0)
        }
    
    def analyze_batch(self, df):
        """批量分析DataFrame"""
        results = []
        for _, row in df.iterrows():
            result = self.analyze(str(row.get('title', '')), str(row.get('content', '')))
            results.append(result)
        return pd.DataFrame(results)


# ========== 使用示例 ==========
if __name__ == "__main__":
    import duckdb
    
    # 读取新闻
    con = duckdb.connect()
    df = con.execute("SELECT datetime, title, content FROM read_parquet('D:/iquant_data/data_v2/news_raw_data/*.parquet')").df()
    con.close()
    
    # 分析
    analyzer = K2_6_NewsAnalyzer()
    results = analyzer.analyze_batch(df)
    
    # 合并结果
    df['sentiment_k2_6'] = results['score']
    df['event_type'] = results['event_type']
    df['reason'] = results['reason']
    
    # 保存
    df.to_csv('news_k2_6_analysis.csv', index=False, encoding='utf-8-sig')
    print(f"Analyzed {len(df)} news articles")
    print(f"Sentiment distribution:")
    print(df['sentiment_k2_6'].value_counts().sort_index())
