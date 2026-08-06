# -*- coding: utf-8 -*-
"""docs_sync 测试: txt 解析 / 指标容差对比 / markdown 表格行同步。"""
import numpy as np
import pytest

from repro_core import docs_sync


def test_compare_metrics_ok():
    a = {"cagr": 0.1383, "note": "x"}
    e = {"cagr": 0.1383, "note": "any-string"}
    assert docs_sync.compare_metrics(a, e, rtol=0.01) == []


def test_compare_metrics_tolerance():
    a = {"cagr": 0.14}
    e = {"cagr": 0.1383}
    # |0.14-0.1383| = 0.0017 > 0.01*0.1383 = 0.001383 -> 超差
    diffs = docs_sync.compare_metrics(a, e, rtol=0.01)
    assert len(diffs) == 1
    assert docs_sync.compare_metrics({"cagr": 0.1386}, {"cagr": 0.1383}, rtol=0.01) == []


def test_compare_metrics_missing_and_nan():
    assert len(docs_sync.compare_metrics({}, {"x": 1.0})) == 1
    assert len(docs_sync.compare_metrics({"x": float("nan")}, {"x": 1.0})) == 1


def test_parse_risk_control_txt():
    txt = (
        "BASE+VAL         12.00%    1.10   15.00%   55.0%   -1.00%   0.80   100%\n"
        "+MA20三档098     13.83%    0.81   18.36%   55.0%   59.91%   0.75    78%\n"
    )
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write(txt)
        path = fh.name
    try:
        rows = docs_sync.parse_risk_control_txt(path, ["BASE+VAL", "+MA20三档098"])
        r = rows["+MA20三档098"]
        assert r["cagr"] == pytest.approx(0.1383)
        assert r["sharpe"] == pytest.approx(0.81)
        assert r["mdd"] == pytest.approx(0.1836)
        assert r["excess_v_etf"] == pytest.approx(0.5991)
        assert r["calmar"] == pytest.approx(0.75)
        assert r["avg_weight"] == pytest.approx(0.78)
    finally:
        os.remove(path)


def test_render_pct():
    assert docs_sync.render_pct(0.1383) == "13.83%"
    assert docs_sync.render_pct(float("nan")) == "n/a"


def test_sync_doc_table_replace_existing_row(tmp_path):
    md = tmp_path / "doc.md"
    md.write_text(
        "| 策略 | 年化 | Sharpe |\n"
        "|---|---|---|\n"
        "| +MA20三档098 | 16.04% | 0.90 |\n"
        "| BASE+VAL | 15.12% | 0.88 |\n", encoding="utf-8")
    replaced = docs_sync.sync_doc_table(str(md), "t", {"cagr": 0.1383, "sharpe": 0.81},
                                        "+MA20三档098", "snap=data_20260806-v1")
    assert replaced is True
    content = md.read_text(encoding="utf-8")
    assert "| +MA20三档098 | 13.83% | 0.81 |" in content
    assert "<!-- snap=data_20260806-v1 -->" in content


def test_sync_doc_table_append_when_missing(tmp_path):
    md = tmp_path / "doc.md"
    md.write_text("| 策略 | 年化 |\n|---|---|\n", encoding="utf-8")
    replaced = docs_sync.sync_doc_table(str(md), "t", {"cagr": 0.1}, "NEW", "")
    assert replaced is True
    assert "| NEW | 10.00% |" in md.read_text(encoding="utf-8")


def test_sync_doc_table_missing_file(tmp_path):
    assert docs_sync.sync_doc_table(str(tmp_path / "no.md"), "t", {}, "X", "") is False
