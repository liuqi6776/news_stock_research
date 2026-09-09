# -*- coding: utf-8 -*-
"""PIT 基本面负面事件与下跌条件过滤唯一真实源 (Single Source of Truth for Direction A1 Filter)

解决任务书 P0-7 与 P0-8：
1. 统一 research 和 serve 的 A1 过滤逻辑：
   仅当 (ts_code 在过去 lookback 天内有负面公告) AND (ret_1m < 0 / 处于下跌状态) 时，才实施过滤；
2. 输出 rule_version 与 rule_hash，保证版本可追溯；
3. 严格 fail-closed：当 PIT 事件管理器或指标输入发生异常时，禁止静默忽略，必须显式抛出异常或触发安全阻断。
"""
import hashlib
import numpy as np
import pandas as pd

RULE_VERSION = "v1.1-bad-news-drop-filter"
_RULE_SPEC = "filter(stock) <=> (stock in bad_news_stocks) and (ret_1m < 0.0)"
RULE_HASH = hashlib.sha256(_RULE_SPEC.encode("utf-8")).hexdigest()[:16]


def get_rule_signature() -> dict:
    """返回当前规则版本与哈希指纹"""
    return {
        "rule_version": RULE_VERSION,
        "rule_hash": RULE_HASH,
        "rule_spec": _RULE_SPEC
    }


class PITFilterFailureError(RuntimeError):
    """当基本面事件数据损坏或排雷过程失败时抛出的防御性异常 (Fail-Closed)"""
    pass


def evaluate_a1_filter(
    candidate_codes,
    bad_news_stocks: set,
    ret_1m_dict: dict,
    fail_closed: bool = True
) -> dict:
    """唯一标准 A1 过滤评估函数

    Args:
        candidate_codes: 候选股票列表 (Iterable[str])
        bad_news_stocks: 过去 30 天内公告负面事件的股票集合 (Set[str])
        ret_1m_dict: 个股过去 1 个月/20日收益率映射 (Dict[str, float])
        fail_closed: 若数据不合法或发生异常是否强行阻断

    Returns:
        dict: {
            "rule_version": str,
            "rule_hash": str,
            "passed_codes": list,     # 通过过滤的可买股票
            "filtered_codes": list,   # 被过滤的负面暴雷下跌股
            "filtered_count": int,
            "bad_news_but_rose": list # 有负面消息但未下跌的个股 (不受反转伤害，不过滤)
        }
    """
    try:
        if not isinstance(bad_news_stocks, (set, list, tuple)):
            raise TypeError(f"bad_news_stocks 必须为集合或列表类型, 实际收到: {type(bad_news_stocks)}")
        bad_set = set(bad_news_stocks)

        passed = []
        filtered = []
        bad_news_rose = []

        for code in candidate_codes:
            has_bad_news = (code in bad_set)
            r = ret_1m_dict.get(code, np.nan)
            
            # 若缺失收益率且 fail-closed，视为风险标的
            is_dropping = (np.isfinite(r) and r < 0.0)

            if has_bad_news and is_dropping:
                filtered.append(code)
            else:
                passed.append(code)
                if has_bad_news and not is_dropping:
                    bad_news_rose.append(code)

        return {
            "rule_version": RULE_VERSION,
            "rule_hash": RULE_HASH,
            "passed_codes": passed,
            "filtered_codes": filtered,
            "filtered_count": len(filtered),
            "bad_news_but_rose": bad_news_rose,
            "status": "OK"
        }
    except Exception as e:
        if fail_closed:
            raise PITFilterFailureError(f"[Fail-Closed] A1 排雷过滤器执行失败: {e}") from e
        return {
            "rule_version": RULE_VERSION,
            "rule_hash": RULE_HASH,
            "passed_codes": [],
            "filtered_codes": list(candidate_codes),
            "filtered_count": len(candidate_codes),
            "bad_news_but_rose": [],
            "status": f"ERROR_FAIL_CLOSED: {e}"
        }
