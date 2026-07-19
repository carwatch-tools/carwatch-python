import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import carwatch as cw
from carwatch.exceptions import SchemaError


def saliva_data(include_s0=False):
    samples = ["S0", "S1", "S2", "S3", "S4"] if include_s0 else ["S1", "S2", "S3", "S4"]
    times = [-10, 0, 10, 20, 30] if include_s0 else [0, 10, 20, 30]
    index = pd.MultiIndex.from_product(
        [["VP01", "VP02"], ["D1"], samples],
        names=["participant", "day", "sample"],
    )
    values = [5, 1, 3, 2, 4, 10, 12, 8, 6, 4] if include_s0 else [1, 3, 2, 4, 10, 12, 8, 6]
    return pd.DataFrame(
        {
            "cortisol": values,
            "amylase": np.asarray(values) * 10,
            "time": times * 2,
        },
        index=index,
    )


def test_auc_matches_pruessner_examples():
    index = pd.MultiIndex.from_product(
        [["P1", "P2"], ["S1", "S2", "S3", "S4", "S5"]],
        names=["participant", "sample"],
    )
    data = pd.DataFrame(
        {
            "cortisol": [3.5, 7, 14, 7, 10] * 2,
            "time": [1, 2, 3, 4, 5, 0, 10, 15, 30, 45],
        },
        index=index,
    )

    result = cw.saliva.auc(data)

    assert result.loc["P1", "cortisol_auc_g"] == pytest.approx(34.75)
    assert result.loc["P1", "cortisol_auc_i"] == pytest.approx(20.75)
    assert result.loc["P2", "cortisol_auc_g"] == pytest.approx(390)
    assert result.loc["P2", "cortisol_auc_i"] == pytest.approx(232.5)


def test_auc_supports_individual_times_and_multiday_index():
    result = cw.saliva.auc(saliva_data())

    assert result.index.names == ["participant", "day"]
    assert result.loc[("VP01", "D1"), "cortisol_auc_g"] == pytest.approx(75)
    assert result.loc[("VP02", "D1"), "cortisol_auc_i"] == pytest.approx(-20)


def test_auc_uses_explicit_times_and_computes_post_auc():
    result = cw.saliva.auc(
        saliva_data(),
        sample_times=[-10, 0, 10, 20],
        compute_auc_post=True,
    )

    assert "cortisol_auc_i_post" in result
    assert result.loc[("VP01", "D1"), "cortisol_auc_i_post"] == pytest.approx(-5)


def test_auc_returns_nan_for_incomplete_curve():
    data = saliva_data()
    data.loc[("VP01", "D1", "S2"), "cortisol"] = np.nan

    result = cw.saliva.auc(data)

    assert pd.isna(result.loc[("VP01", "D1"), "cortisol_auc_g"])
    assert not pd.isna(result.loc[("VP02", "D1"), "cortisol_auc_g"])


def test_auc_rejects_non_increasing_time():
    data = saliva_data()
    data.loc[("VP01", "D1"), "time"] = [0, 10, 10, 30]

    with pytest.raises(ValueError, match="strictly increasing"):
        cw.saliva.auc(data)


def test_slope_by_label_and_index():
    by_label = cw.saliva.slope(saliva_data(), sample_labels=("S1", "S4"))
    by_index = cw.saliva.slope(saliva_data(), sample_idx=(0, -1))

    assert_frame_equal(by_label, by_index)
    assert by_label.loc[("VP01", "D1"), "cortisol_slopeS1S4"] == pytest.approx(0.1)


def test_basic_response_features():
    data = saliva_data()

    assert cw.saliva.initial_value(data)["cortisol_ini_val"].tolist() == [1, 10]
    assert cw.saliva.max_value(data)["cortisol_max_val"].tolist() == [4, 12]
    assert cw.saliva.max_increase(data)["cortisol_max_inc"].tolist() == [3, 2]
    assert cw.saliva.max_increase(data, percent=True)["cortisol_max_inc_percent"].tolist() == [300, 20]


def test_remove_s0_changes_feature_baseline():
    data = saliva_data(include_s0=True)

    assert cw.saliva.initial_value(data)["cortisol_ini_val"].tolist() == [5, 10]
    assert cw.saliva.initial_value(data, remove_s0=True)["cortisol_ini_val"].tolist() == [1, 12]


def test_compute_features_returns_common_feature_set():
    result = cw.saliva.compute_features(saliva_data())

    assert result.columns.tolist() == [
        "cortisol_auc_g",
        "cortisol_auc_i",
        "cortisol_ini_val",
        "cortisol_max_val",
        "cortisol_max_inc",
        "cortisol_slopeS1S4",
    ]


def test_metrics_support_multiple_analytes():
    result = cw.saliva.auc(saliva_data(), saliva_type=["cortisol", "amylase"])

    assert set(result) == {"cortisol", "amylase"}
    assert result["amylase"].iloc[0, 0] == pytest.approx(result["cortisol"].iloc[0, 0] * 10)


def test_standard_features_and_mean_se():
    standard = cw.saliva.standard_features(saliva_data())
    summary = cw.saliva.mean_se(saliva_data())

    assert standard.loc[("VP01", "D1"), "cortisol_argmax"] == 3
    assert standard.loc[("VP01", "D1"), "cortisol_mean"] == pytest.approx(2.5)
    assert summary.index.names == ["sample", "time"]
    assert summary.loc[("S1", 0), "mean"] == pytest.approx(5.5)


def test_saliva_feature_wide_to_long():
    features = cw.saliva.compute_features(saliva_data())
    result = cw.saliva.utils.saliva_feature_wide_to_long(features, "cortisol")

    assert result.index.names == ["participant", "day", "saliva_feature"]
    assert result.columns.tolist() == ["cortisol"]


def test_datetime_sample_times_to_minutes():
    times = pd.DataFrame(
        [["06:00:00", "06:15:00", "06:45:00"]],
        index=pd.Index(["VP01"], name="participant"),
        columns=pd.Index(["S1", "S2", "S3"], name="sample"),
    )

    result = cw.saliva.utils.sample_times_datetime_to_minute(times)

    assert result.loc["VP01"].tolist() == [0, 15, 45]


def test_metrics_validate_schema_and_parameters():
    with pytest.raises(SchemaError):
        cw.saliva.auc(pd.DataFrame({"cortisol": [1]}))
    with pytest.raises(SchemaError):
        cw.saliva.auc(saliva_data(), saliva_type="missing")
    with pytest.raises(IndexError):
        cw.saliva.slope(saliva_data())
    with pytest.raises(IndexError):
        cw.saliva.slope(saliva_data(), sample_labels=("missing", "S2"))
