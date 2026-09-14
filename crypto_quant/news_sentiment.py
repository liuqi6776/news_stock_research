# -*- coding: utf-8 -*-
"""
Ethereum Real-Time News & Sentiment Analyzer
以太坊实时新闻头条与情绪解析器 (Google News RSS + 金融词典极性分析)
"""
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# 加密货币专属情感词典
BULLISH_KEYWORDS = {
    'surge': 2.0, 'rally': 2.0, 'bull': 1.5, 'bullish': 2.0, 'breakout': 2.0,
    'high': 1.0, 'gain': 1.5, 'gains': 1.5, 'inflow': 2.0, 'inflows': 2.0,
    'upgrade': 1.5, 'approval': 2.5, 'etf': 1.5, 'adoption': 1.5, 'growth': 1.5,
    'accumulate': 1.5, 'whale buy': 2.5, 'record': 1.5, 'soar': 2.0, 'jump': 1.5,
    'expansion': 1.2, 'staking': 1.0, 'outperform': 1.5, 'partnership': 1.5
}

BEARISH_KEYWORDS = {
    'crash': -2.5, 'dump': -2.0, 'bear': -1.5, 'bearish': -2.0, 'plunge': -2.5,
    'drop': -1.5, 'loss': -1.5, 'losses': -1.5, 'outflow': -2.0, 'outflows': -2.0,
    'hack': -3.0, 'exploit': -3.0, 'scam': -3.0, 'lawsuit': -2.0, 'ban': -2.5,
    'reject': -2.0, 'rejection': -2.0, 'sec': -1.0, 'investigation': -2.0,
    'liquidation': -2.0, 'selloff': -2.5, 'slump': -2.0, 'decline': -1.5
}


def fetch_latest_eth_news(limit=50):
    """从 Google News RSS 抓取最新以太坊相关新闻头条"""
    url = 'https://news.google.com/rss/search?q=Ethereum&hl=en-US&gl=US&ceid=US:en'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    
    articles = []
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_data = resp.read()
            root = ET.fromstring(xml_data)
            items = root.findall('.//item')
            for item in items[:limit]:
                title = item.find('title').text if item.find('title') is not None else ''
                pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ''
                link = item.find('link').text if item.find('link') is not None else ''
                
                # 简单清洗
                title_clean = title.split(' - ')[0] if ' - ' in title else title
                articles.append({
                    'title': title_clean,
                    'pub_date': pub_date,
                    'link': link
                })
    except Exception as e:
        print(f"Error fetching news RSS: {e}")
        return []

    return articles


def score_news_sentiment(title: str) -> float:
    """计算单条新闻标题的情绪得分 (-1.0 到 +1.0)"""
    title_lower = title.lower()
    score = 0.0
    matches = 0

    for word, weight in BULLISH_KEYWORDS.items():
        if word in title_lower:
            score += weight
            matches += 1

    for word, weight in BEARISH_KEYWORDS.items():
        if word in title_lower:
            score += weight
            matches += 1

    if matches == 0:
        return 0.0
    
    # 双曲正切归一化至 [-1.0, +1.0]
    normalized_score = float(np.tanh(score / 3.0))
    return normalized_score


def get_latest_eth_news_sentiment():
    """获取最新新闻列表与综合情绪打分"""
    articles = fetch_latest_eth_news(limit=50)
    if not articles:
        return {'articles': [], 'mean_sentiment': 0.0, 'sentiment_label': 'Neutral'}

    scores = []
    for a in articles:
        s = score_news_sentiment(a['title'])
        a['sentiment'] = s
        scores.append(s)

    mean_s = float(np.mean(scores))
    if mean_s > 0.15:
        label = 'Bullish / 偏多'
    elif mean_s < -0.15:
        label = 'Bearish / 偏空'
    else:
        label = 'Neutral / 中性'

    return {
        'articles': articles,
        'mean_sentiment': mean_s,
        'sentiment_label': label,
        'total_news': len(articles)
    }


if __name__ == '__main__':
    res = get_latest_eth_news_sentiment()
    print(f"Total News: {res['total_news']} | Mean Sentiment: {res['mean_sentiment']:+.4f} ({res['sentiment_label']})")
    print("\nSample Headlines with Scores:")
    for a in res['articles'][:5]:
        print(f"[{a['sentiment']:+.2f}] {a['title']}")
