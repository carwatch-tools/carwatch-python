import pandas as pd
import pytest

import carwatch as cw
from carwatch.exceptions import SchemaError


def _write_csv(tmp_path, content):
    path = tmp_path / "saliva.csv"
    path.write_text(content)
    return path


def test_load_long_saliva_preserves_ids_and_barcode(tmp_path):
    content = """Participant,Day,Tube,Barcode,Cortisol,Comment
02,1,B1,0010101,2.5,valid
02,1,B2,0010102,3.5,valid
"""
    result = cw.io.load_saliva(
        _write_csv(tmp_path, content),
        participant_col="Participant",
        day_col="Day",
        sample_col="Tube",
        barcode_col="Barcode",
        value_cols="Cortisol",
    )

    assert result.index.names == ["participant", "day", "sample"]
    assert result.index.get_level_values("participant").unique().tolist() == ["02"]
    assert result["barcode"].tolist() == ["0010101", "0010102"]
    assert result["Cortisol"].tolist() == [2.5, 3.5]
    assert result["Comment"].tolist() == ["valid", "valid"]


def test_load_long_saliva_infers_numeric_values(tmp_path):
    content = """participant,sample,cortisol,note
VP01,S1,1.2,ok
VP01,S2,2.4,ok
"""
    result = cw.io.load_saliva(_write_csv(tmp_path, content))

    assert result["cortisol"].tolist() == [1.2, 2.4]
    assert result["note"].tolist() == ["ok", "ok"]


def test_load_long_saliva_applies_identifier_maps(tmp_path):
    content = """participant,day,sample,cortisol
2,A,Cort_1,1.2
"""
    result = cw.io.load_saliva(
        _write_csv(tmp_path, content),
        participant_map={"2": "02"},
        day_map={"A": "D1"},
        sample_map={"Cort_1": "B1"},
    )

    assert result.index.tolist() == [("02", "D1", "B1")]


def test_load_wide_saliva_with_explicit_mapping(tmp_path):
    content = """Participant,Day,Cort_1,Cort_2
02,D1,2.5,3.5
"""
    result = cw.io.load_saliva(
        _write_csv(tmp_path, content),
        format="wide",
        participant_col="Participant",
        day_col="Day",
        sample_columns={"Cort_1": "B1", "Cort_2": "B2"},
        value_name="cortisol",
    )

    assert result.index.tolist() == [("02", "D1", "B1"), ("02", "D1", "B2")]
    assert result["cortisol"].tolist() == [2.5, 3.5]


def test_load_wide_saliva_can_be_inferred(tmp_path):
    content = """participant,B1,B2
VP01,2.5,3.5
"""
    result = cw.io.load_saliva(_write_csv(tmp_path, content), value_name="cortisol")

    assert result.index.tolist() == [("VP01", "B1"), ("VP01", "B2")]


def test_load_saliva_supports_multiple_measurements(tmp_path):
    content = """participant,sample,cortisol,amylase
VP01,S1,1.2,10
VP01,S2,2.4,12
"""
    result = cw.io.load_saliva(
        _write_csv(tmp_path, content), value_cols=["cortisol", "amylase"]
    )

    assert result[["cortisol", "amylase"]].to_dict(orient="list") == {
        "cortisol": [1.2, 2.4],
        "amylase": [10, 12],
    }


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"format": "wide"}, "value_name"),
        ({"format": "invalid"}, "format"),
        ({"participant_col": "missing"}, "missing required"),
    ],
)
def test_load_saliva_rejects_invalid_configuration(tmp_path, kwargs, match):
    content = "participant,sample,cortisol\nVP01,S1,1.2\n"

    with pytest.raises((SchemaError, ValueError), match=match):
        cw.io.load_saliva(_write_csv(tmp_path, content), **kwargs)


def test_load_saliva_rejects_non_numeric_measurements(tmp_path):
    content = "participant,sample,cortisol\nVP01,S1,invalid\n"

    with pytest.raises(SchemaError, match="non-numeric"):
        cw.io.load_saliva(_write_csv(tmp_path, content), value_cols="cortisol")


def test_load_saliva_rejects_duplicate_tubes(tmp_path):
    content = """participant,day,sample,cortisol
VP01,D1,S1,1.2
VP01,D1,S1,2.4
"""

    with pytest.raises(SchemaError, match="duplicate"):
        cw.io.load_saliva(_write_csv(tmp_path, content))


def test_load_saliva_rejects_missing_sample_identifier(tmp_path):
    content = "participant,sample,cortisol\nVP01,,1.2\n"

    with pytest.raises(SchemaError, match="missing values"):
        cw.io.load_saliva(_write_csv(tmp_path, content))


def test_load_saliva_returns_numeric_nan_for_empty_measurement(tmp_path):
    content = """participant,sample,cortisol
VP01,S1,1.2
VP01,S2,
"""
    result = cw.io.load_saliva(_write_csv(tmp_path, content), value_cols="cortisol")

    assert pd.isna(result.loc[("VP01", "S2"), "cortisol"])
