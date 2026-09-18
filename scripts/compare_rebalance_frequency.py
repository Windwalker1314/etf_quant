"""Frozen monthly versus Friday-weekly research; never changes the live account/config."""
from __future__ import annotations

import copy
import hashlib
import html
import shutil
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go

from steadyquant.adaptive import adaptive_weights
from steadyquant.adaptive_audit import paired_sharpe_interval
from steadyquant.backtest import simulate
from steadyquant.commission_study import metrics, save_result
from steadyquant.config import ROOT, load_active_config, write_json
from steadyquant.data import TZ, Cache
from steadyquant.metrics import yearly
from steadyquant.reports import figure_html, shell


def main():
    cfg = load_active_config()
    assert cfg['model'] == 'adaptive' and cfg['risk_method'] == 'equal_risk'
    assert cfg['rebalance_frequency'] == 'monthly_first_session'
    protected = [ROOT / 'configs/active.yaml', ROOT / 'data/portfolio.json']
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    cache = Cache()
    data = cache.load(cfg)
    ends = {d.date.max() for d in data.values()}
    assert len(ends) == 1, 'All assets must have the same latest close'
    end = str(next(iter(ends)).date())
    out = ROOT / 'outputs/frequency' / datetime.now(TZ).strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True)
    variants = {}
    for name, frequency in [('monthly', 'monthly_first_session'), ('weekly', 'weekly')]:
        variant = copy.deepcopy(cfg)
        variant.update(initial_cash=200000, rebalance_frequency=frequency)
        variants[name] = variant
    assert {k for k in variants['monthly'] if variants['monthly'][k] != variants['weekly'][k]} == {'rebalance_frequency'}
    write_json(out / 'protocol.json', dict(
        created_at=datetime.now(TZ).isoformat(), end=end, initial_cash=200000,
        main_start='2019-01-01', additional_fresh_starts=['2015-01-05', '2023-01-01'],
        variants=variants, cost_multipliers=[1, 2],
        weekly='Friday close -> next observed session open. Friday holidays skip that week, no makeup.',
        comparison='Only rebalance frequency differs. Shared causal targets, fixed 3pp drift band and asset order. No strategy selection or activation.',
        limitations='Retrospective selected universe; not an untouched holdout. Funds unavailable before inception. Adjustment factors approximate reinvestment, not real dividend payment dates. No historical QDII premium gate.',
        protected_sha256=before, data_sha256=cache.snapshot_digest,
    ))
    (out / 'source').mkdir()
    shutil.copyfile(__file__, out / 'source/compare_rebalance_frequency.py')
    for module in ['adaptive', 'strategy', 'backtest', 'execution', 'metrics', 'commission_study']:
        shutil.copyfile(ROOT / f'src/steadyquant/{module}.py', out / f'source/{module}.py')
    (out / 'data').mkdir()
    for symbol, frame in data.items():
        frame.to_parquet(out / f'data/{symbol}.parquet', index=False)
    print('Computing shared targets and checking truncated-history causality', flush=True)
    targets, _ = adaptive_weights(data, cfg)
    targets.to_parquet(out / 'targets.parquet')
    truncated = {s: d[d.date <= pd.Timestamp('2022-12-30')] for s, d in data.items()}
    prefix, _ = adaptive_weights(truncated, cfg)
    pd.testing.assert_frame_equal(prefix, targets.loc[prefix.index])
    results, rows, annual, checks = {}, [], [], {}
    for start in ['2015-01-05', '2019-01-01', '2023-01-01']:
        for mul in ([1, 2] if start != '2015-01-05' else [1]):
            for name, variant in variants.items():
                key = f'{name}_{start[:4]}_cost{mul}'
                print(f'Run and independently replay {key}', flush=True)
                result = simulate(data, targets, variant, start=start, end=end, cost_multiplier=mul)
                checks[key] = save_result(out / key, result, data, variant, mul)
                record = dict(model=name, fresh_start=start, cost_multiplier=mul,
                              **metrics(result, variant, cost_multiplier=mul))
                record.update(final_nav=float(result.equity.equity.iloc[-1]),
                              trade_days=int(result.trades.date.nunique()))
                rows.append(record)
                results[key] = result
                if start == '2019-01-01' and mul == 1:
                    annual.extend(dict(model=name, **r) for r in yearly(result.equity.equity, .02).to_dict('records'))
    table = pd.DataFrame(rows)
    table.to_csv(out / 'comparison.csv', index=False)
    pd.DataFrame(annual).to_csv(out / 'yearly.csv', index=False)
    intervals = {}
    for start in ['2019', '2023']:
        intervals[start] = paired_sharpe_interval(results[f'weekly_{start}_cost1'].equity.equity,
                                                  results[f'monthly_{start}_cost1'].equity.equity)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == before[str(p.relative_to(ROOT))] for p in protected)
    assert cache.digest(cfg) == cache.snapshot_digest
    summary = dict(end=end, prefix_check_passed=True, replay_checks=checks, paired_sharpe_difference=intervals,
                   active_config_and_account_unchanged=True, comparisons=rows)
    write_json(out / 'summary.json', summary)
    labels = {'monthly':'月频', 'weekly':'周频（周五）'}
    body = f'<h1>月频 vs 周频调仓</h1><p class="sub">20 万本金 · 截至 {end} · 等风险配置 · 两组只改变调仓频率</p>'
    body += '<div class="note">佣金万 1.5，每笔最低 5 元；单边滑点 5bp。保留 3 个百分点偏离阈值，触发调仓日不等于必须成交。周五休市则跳过该周，下一交易日执行。账户和实盘策略未修改。</div>'
    for start in ['2019-01-01', '2023-01-01', '2015-01-05']:
        body += f'<h2>{start[:4]} 年起重新投入 20 万</h2>'
        display = table[(table.fresh_start == start) & (table.cost_multiplier == 1)].copy()
        display['model'] = display.model.map(labels)
        for field in ['cagr', 'max_drawdown']:
            display[field] = display[field].map(lambda x: f'{x:.2%}')
        for field in ['sharpe', 'annual_turnover']:
            display[field] = display[field].map(lambda x: f'{x:.2f}')
        for field in ['commission_cny', 'slippage_cny', 'final_nav']:
            display[field] = display[field].map(lambda x: f'{x:,.0f}')
        display = display[['model','cagr','sharpe','max_drawdown','final_nav','trades','trade_days','annual_turnover','commission_cny','slippage_cny']]
        display.columns = ['调仓频率','年化收益','夏普','最大回撤','期末资产（元）','成交笔数','成交天数','年换手倍数（双边合计）','累计佣金（元）','累计滑点（元）']
        body += '<div class="box" style="overflow-x:auto">' + display.to_html(index=False, escape=True, border=0) + '</div>'
    fig = go.Figure()
    drawdown = go.Figure()
    for name in variants:
        nav = results[f'{name}_2019_cost1'].equity.equity
        fig.add_trace(go.Scatter(x=nav.index, y=nav / 200000, name=labels[name]))
        drawdown.add_trace(go.Scatter(x=nav.index, y=nav / nav.cummax() - 1, name=labels[name]))
    drawdown.update_yaxes(tickformat='.0%')
    body += '<h2>2019 年起净值</h2>' + figure_html(fig) + '<h2>回撤</h2>' + figure_html(drawdown)
    body += '<h2>费用翻倍压力测试</h2><p class="sub">佣金、最低费用与滑点同时乘 2；重新模拟成交。</p>'
    stress = table[table.cost_multiplier == 2][['model','fresh_start','cagr','sharpe','max_drawdown']].copy()
    stress['model'] = stress.model.map(labels)
    for col in ['cagr','max_drawdown']:
        stress[col] = stress[col].map(lambda v:f'{v:.2%}')
    stress['sharpe'] = stress.sharpe.map(lambda v:f'{v:.2f}')
    stress.columns = ['调仓频率','重新建仓日','年化收益','夏普','最大回撤']
    body += '<div class="box">' + stress.to_html(index=False, border=0) + '</div>'
    body += '<h2>逐年收益</h2>'
    years = pd.DataFrame(annual).pivot(index='year', columns='model', values='total_return').rename(columns=labels)
    body += '<div class="box">' + years.map(lambda v:f'{v:.2%}').to_html(border=0) + '</div>'
    body += '<h2>验证与限制</h2><p class="sub">10 组模拟均通过独立现金、份额、手续费和净值重放核对；截断数据重算信号一致。所有历史窗口均为回顾性检验，不能视为未见过的样本外证据。当前资产池存在选择偏差；分红按复权因子近似再投资，未建模历史 QDII 溢价。</p>'
    body += '<p class="sub">周频减月频夏普差值的历史区块重采样 95% 区间：' + html.escape(str(intervals)) + '</p>'
    (out / 'report.html').write_text(shell('调仓频率对照研究', body, plotly=True), encoding='utf-8')
    write_json(ROOT / 'outputs/latest_frequency.json', {'path':str(out)})
    print(table[['model','fresh_start','cost_multiplier','cagr','sharpe','max_drawdown','trades','commission_cny','slippage_cny']].to_string(index=False), flush=True)
    print(f'REPORT={out}', flush=True)


if __name__ == '__main__':
    main()
