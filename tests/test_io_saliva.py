import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

import carwatch as cw
from carwatch.exceptions import SchemaError


def _write_csv(tmp_path, content):
    path = tmp_path / "saliva.csv"
    path.write_text(content)
    return path


def test_load_saliva(tmp_path):
    content = """participant,sample,cortisol
VP_01,S0,
VP_01,S1,1.8659
VP_02,S0,3.17605
"""

    result = cw.io.load_saliva(_write_csv(tmp_path, content))

    assert result.index.names == ["participant", "sample"]
    assert result.index.tolist() == [("VP_01", "S0"), ("VP_01", "S1"), ("VP_02", "S0")]
    assert result.columns.tolist() == ["cortisol"]
    assert pd.isna(result.loc[("VP_01", "S0"), "cortisol"])
    assert result.loc[("VP_01", "S1"), "cortisol"] == 1.8659
    assert is_numeric_dtype(result["cortisol"])


def test_load_saliva_supports_other_saliva_types(tmp_path):
    content = """participant,sample,amylase
VP_01,S0,12.5
"""

    result = cw.io.load_saliva(_write_csv(tmp_path, content), saliva_type="amylase")

    assert result.columns.tolist() == ["amylase"]
    assert result.loc[("VP_01", "S0"), "amylase"] == 12.5


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("participant,sample\nVP_01,S0\n", "missing columns"),
        (
            "participant,sample,cortisol,note\nVP_01,S0,1.0,valid\n",
            "unexpected columns",
        ),
        ("subject,sample,cortisol\nVP_01,S0,1.0\n", "missing columns"),
    ],
)
def test_load_saliva_rejects_invalid_columns(tmp_path, content, match):
    with pytest.raises(SchemaError, match=match):
        cw.io.load_saliva(_write_csv(tmp_path, content))


@pytest.mark.parametrize(
    "content",
    [
        "participant,sample,cortisol\n,S0,1.0\n",
        "participant,sample,cortisol\nVP_01,,1.0\n",
        "participant,sample,cortisol\nVP_01,   ,1.0\n",
    ],
)
def test_load_saliva_rejects_missing_identifiers(tmp_path, content):
    with pytest.raises(SchemaError, match="missing values"):
        cw.io.load_saliva(_write_csv(tmp_path, content))


def test_load_saliva_strips_identifiers(tmp_path):
    content = "participant,sample,cortisol\n VP_01 , S0 ,1.0\n"

    result = cw.io.load_saliva(_write_csv(tmp_path, content))

    assert result.index.tolist() == [("VP_01", "S0")]


def test_load_saliva_rejects_non_numeric_measurements(tmp_path):
    content = "participant,sample,cortisol\nVP_01,S0,invalid\n"

    with pytest.raises(SchemaError, match="non-numeric"):
        cw.io.load_saliva(_write_csv(tmp_path, content))


def test_load_saliva_rejects_duplicate_samples(tmp_path):
    content = """participant,sample,cortisol
VP_01,S0,1.0
VP_01,S0,2.0
"""

    with pytest.raises(SchemaError, match="duplicate participant/sample"):
        cw.io.load_saliva(_write_csv(tmp_path, content))


def test_load_saliva_rejects_empty_file(tmp_path):
    with pytest.raises(SchemaError, match="does not contain"):
        cw.io.load_saliva(_write_csv(tmp_path, "participant,sample,cortisol\n"))


def test_load_saliva_rejects_non_csv_file(tmp_path):
    path = tmp_path / "saliva.xlsx"
    path.write_text("not an Excel file")

    with pytest.raises(ValueError, match="CSV"):
        cw.io.load_saliva(path)


@pytest.mark.parametrize("saliva_type", ["", "   ", None])
def test_load_saliva_rejects_invalid_saliva_type(tmp_path, saliva_type):
    content = "participant,sample,cortisol\nVP_01,S0,1.0\n"

    with pytest.raises(ValueError, match="saliva_type"):
        cw.io.load_saliva(_write_csv(tmp_path, content), saliva_type=saliva_type)
