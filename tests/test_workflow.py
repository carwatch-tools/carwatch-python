from pandas.testing import assert_frame_equal

import carwatch as cw


STUDY_RESULTS = """Study Name,Participant ID,date_D1,awakening_time_D1_app,awakening_type_D1,sample_mismatches_D1,sampling_time_D1_B1,sample_barcode_D1_B1,sample_scanned_D1_B1,sampling_time_D1_B2,sample_barcode_D1_B2,sample_scanned_D1_B2,sampling_time_D1_B3,sample_barcode_D1_B3,sample_scanned_D1_B3,sampling_time_D1_B4,sample_barcode_D1_B4,sample_scanned_D1_B4
logs,02,2025-05-15,06:13:30,self-report,B2->B3;B3->B2,06:13:55,0010101,B1,06:43:51,0010103,B3,06:58:52,0010102,B2,07:13:47,0010104,B4
"""

LONG_SALIVA = """Participant,Day,Tube,Barcode,cortisol
02,1,B1,0010101,1.0
02,1,B2,0010102,2.0
02,1,B3,0010103,3.0
02,1,B4,0010104,4.0
"""

WIDE_SALIVA = """Participant,Day,Cort_1,Cort_2,Cort_3,Cort_4
02,1,1.0,2.0,3.0,4.0
"""


def test_complete_workflow_corrects_swaps_for_long_and_wide_saliva(tmp_path):
    study_path = tmp_path / "study_results.csv"
    long_path = tmp_path / "saliva_long.csv"
    wide_path = tmp_path / "saliva_wide.csv"
    study_path.write_text(STUDY_RESULTS)
    long_path.write_text(LONG_SALIVA)
    wide_path.write_text(WIDE_SALIVA)

    study_results = cw.io.load_study_results(study_path)
    long_saliva = cw.io.load_saliva(
        long_path,
        participant_col="Participant",
        day_col="Day",
        sample_col="Tube",
        barcode_col="Barcode",
        value_cols="cortisol",
        day_map={"1": "D1"},
    )
    wide_saliva = cw.io.load_saliva(
        wide_path,
        format="wide",
        participant_col="Participant",
        day_col="Day",
        sample_columns={
            "Cort_1": "B1",
            "Cort_2": "B2",
            "Cort_3": "B3",
            "Cort_4": "B4",
        },
        value_name="cortisol",
        day_map={"1": "D1"},
    )

    merged_long = cw.merge_saliva(study_results, long_saliva, allow_unmatched=False)
    merged_wide = cw.merge_saliva(study_results, wide_saliva, allow_unmatched=False)

    assert merged_long["cortisol"].tolist() == [1.0, 3.0, 2.0, 4.0]
    assert merged_wide["cortisol"].tolist() == [1.0, 3.0, 2.0, 4.0]
    assert merged_long["mismatch_corrected"].tolist() == [False, True, True, False]
    assert merged_wide["mismatch_corrected"].tolist() == [False, True, True, False]
    assert merged_long["match_method"].tolist() == ["barcode"] * 4
    assert merged_wide["match_method"].tolist() == ["scanned_sample"] * 4

    long_features = cw.saliva.compute_features(merged_long)
    wide_features = cw.saliva.compute_features(merged_wide)
    assert_frame_equal(long_features, wide_features)
    assert long_features.index.tolist() == [("logs", "02", "D1")]
