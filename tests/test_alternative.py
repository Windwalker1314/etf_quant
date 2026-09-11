import numpy as np
import pandas as pd

from steadyquant.alternative_factors import (
    analyst_revision,
    factor_scores,
    financial_asof,
    industry_asof,
    prepare_analysts,
    prepare_financial,
    ttm,
)
from steadyquant.alternative_panel import disclosure_surprises


def financial_row(**overrides):
    row = dict(
        ts_code="600000.SH",
        ann_date="20230320",
        f_ann_date="20230320",
        end_date="20221231",
        report_type="1",
        comp_type="1",
        update_flag="0",
        revenue=100,
        oper_cost=60,
        operate_profit=30,
        n_income=20,
        n_income_attr_p=18,
        basic_eps=1.8,
    )
    return {**row, **overrides}


def test_financial_revision_does_not_backfill_and_comparative_waits():
    raw = pd.DataFrame(
        [
            financial_row(),
            financial_row(update_flag="1", n_income=999),
            financial_row(report_type="4", ann_date="20240320", f_ann_date="20240320", n_income=25),
        ]
    )
    clean = prepare_financial(raw, "income")
    assert len(clean) == 2
    assert financial_asof(clean, pd.Timestamp("2023-03-20 18:30")).empty
    assert financial_asof(clean, pd.Timestamp("2023-03-21")).iloc[0].n_income == 20
    assert financial_asof(clean, pd.Timestamp("2024-03-21")).iloc[0].n_income == 25


def test_same_time_conflicting_initial_values_are_unavailable():
    raw = pd.DataFrame([financial_row(), financial_row(n_income=1000)])
    assert prepare_financial(raw, "income").empty


def test_ttm_requires_both_prior_components():
    d = pd.DataFrame(
        {"revenue": [50, 200, 80]}, index=pd.to_datetime(["2022-06-30", "2022-12-31", "2023-06-30"])
    )
    assert ttm(d, pd.Timestamp("2023-06-30"), "revenue") == 230
    assert np.isnan(ttm(d.drop(pd.Timestamp("2022-06-30")), pd.Timestamp("2023-06-30"), "revenue"))


def test_analyst_update_delay_fiscal_pairing_and_one_vote_per_broker():
    rows = []
    for org in ["a", "b", "c"]:
        for day, value in [("20230501", 1000), ("20230801", 1100), ("20230802", 1200)]:
            rows.append(
                dict(
                    ts_code="600000.SH",
                    report_date=day,
                    org_name=org,
                    quarter="2023Q4",
                    np=value,
                    create_time=str(pd.Timestamp(day) + pd.Timedelta(hours=21)),
                )
            )
    rows += [
        dict(rows[0], quarter="2024Q4", np=999999),
        dict(rows[0], org_name="delayed", create_time="2024-01-01"),
        dict(rows[0], org_name="missing", create_time=None),
    ]
    clean = prepare_analysts(pd.DataFrame(rows))
    assert set(clean.org_name) == {"a", "b", "c"}
    result = analyst_revision(clean, pd.Timestamp("2023-09-01"), pd.Timestamp("2023-06-01")).iloc[0]
    assert result.paired_brokers == 3
    assert np.isclose(result.np_revision_63, 0.2)
    assert result.revision_breadth_63 == 1


def test_missing_new_analyst_number_does_not_keep_old_prediction():
    rows = [
        dict(
            ts_code="600000.SH",
            org_name="a",
            quarter="2023Q4",
            report_date="20230501",
            create_time="2023-05-01 21:00",
            np=1000,
        ),
        dict(
            ts_code="600000.SH",
            org_name="a",
            quarter="2023Q4",
            report_date="20230801",
            create_time="2023-08-01 21:00",
            np=None,
        ),
    ]
    result = analyst_revision(
        prepare_analysts(pd.DataFrame(rows)), pd.Timestamp("2023-09-01"), pd.Timestamp("2023-06-01")
    )
    assert result.empty


def test_unknown_industry_and_ambiguous_overlap_are_missing():
    d = pd.DataFrame(
        [
            dict(ts_code="A", in_date="20200101", out_date="20240101", l1_code="one"),
            dict(ts_code="A", in_date="20230101", out_date=None, l1_code="two"),
            dict(ts_code="B", in_date="20250101", out_date=None, l1_code="one"),
        ]
    )
    assert industry_asof(d, pd.Timestamp("2023-06-01")).empty
    assert industry_asof(d, pd.Timestamp("2022-06-01")).loc["A"] == "one"


def test_group_missingness_does_not_create_balanced_score():
    d = factor_scores(pd.DataFrame({"np_revision_63": [1.0], "revision_breadth_63": [1.0]}))
    assert d.revision.iloc[0] == 1
    assert np.isnan(d.balanced.iloc[0])
    assert np.isnan(d.cash_quality.iloc[0])


def test_actual_within_prior_guidance_has_zero_incremental_surprise():
    forecasts = pd.DataFrame(
        [
            dict(
                ts_code="600000.SH",
                ann_date="20230110",
                first_ann_date="20230110",
                end_date="20221231",
                update_flag="0",
                net_profit_min=10,
                net_profit_max=20,
            )
        ]
    )
    income = prepare_financial(pd.DataFrame([financial_row(n_income_attr_p=150000)]), "income")
    balance = pd.DataFrame(
        [
            dict(
                ts_code="600000.SH",
                available_at=pd.Timestamp("2022-10-30 23:59"),
                end_date=pd.Timestamp("2022-09-30"),
                total_assets=10000000,
            )
        ]
    )
    raw = dict(forecast=forecasts, express=pd.DataFrame(columns=["ts_code", "end_date", "ann_date"]))
    events = disclosure_surprises(raw, income, balance)
    assert events.iloc[-1].incremental_surprise == 0
    assert events.iloc[0].reason == "no_prior_comparable_company_disclosure"


def test_intervening_express_prevents_recounting_known_information():
    forecasts = pd.DataFrame(
        [
            dict(
                ts_code="600000.SH",
                ann_date="20230110",
                first_ann_date="20230110",
                end_date="20221231",
                update_flag="0",
                net_profit_min=10,
                net_profit_max=20,
            )
        ]
    )
    income = prepare_financial(pd.DataFrame([financial_row(n_income_attr_p=250000)]), "income")
    balance = pd.DataFrame(
        [
            dict(
                ts_code="600000.SH",
                available_at=pd.Timestamp("2022-10-30 23:59"),
                end_date=pd.Timestamp("2022-09-30"),
                total_assets=10000000,
            )
        ]
    )
    raw = dict(
        forecast=forecasts,
        express=pd.DataFrame([dict(ts_code="600000.SH", end_date="20221231", ann_date="20230210")]),
    )
    e = disclosure_surprises(raw, income, balance).iloc[-1]
    assert np.isnan(e.incremental_surprise)
    assert e.reason == "intervening_express_profit_scope_not_certified"
