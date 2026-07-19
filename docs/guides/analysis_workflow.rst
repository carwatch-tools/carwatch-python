Analysis workflow
=================

Data model
----------

The package uses a long-format sample table as its central representation. A
Study Manager result is indexed by ``study``, ``participant``, ``day``, and the
expected ``sample``. The main identifiers have distinct meanings:

``sample``
    Intended position in the sampling protocol.
``sample_scanned``
    Physical tube scanned when that position was recorded.
``barcode``
    Barcode recorded by CARWatch for the physical tube.
``saliva_sample``
    Physical tube from the laboratory file after merging.

Load Study Manager results
--------------------------

.. code-block:: python

   import carwatch as cw

   study_results = cw.io.load_study_results(
       "study_results.csv",
       tz="Europe/Berlin",
   )

The loader combines each study date with its local awakening and sampling
times. ``time`` contains minutes relative to the app-reported awakening.

Load laboratory measurements
----------------------------

Long files require participant and physical sample columns. Study and day can
be omitted only when they are unambiguous in the Study Manager results.

.. code-block:: python

   saliva = cw.io.load_saliva(
       "saliva.csv",
       participant_col="Participant",
       day_col="Day",
       sample_col="Tube",
       barcode_col="Barcode",
       value_cols=["cortisol", "amylase"],
       day_map={"1": "D1"},
   )

For wide files, map every measurement column to its physical tube label.

.. code-block:: python

   saliva = cw.io.load_saliva(
       "saliva_wide.csv",
       format="wide",
       participant_col="Participant",
       day_col="Day",
       sample_columns={"Cort_1": "B1", "Cort_2": "B2"},
       value_name="cortisol",
       day_map={"1": "D1"},
   )

Correct swaps during the merge
------------------------------

.. code-block:: python

   merged = cw.merge_saliva(study_results, saliva)

Automatic matching uses a unique barcode when available. Otherwise it matches
the laboratory tube against ``sample_scanned``. Consider this recorded case:

.. list-table::
   :header-rows: 1

   * - Expected sample
     - Scanned tube
     - Laboratory value assigned
   * - B1
     - B1
     - Tube B1
   * - B2
     - B3
     - Tube B3
   * - B3
     - B2
     - Tube B2
   * - B4
     - B4
     - Tube B4

The result remains indexed by expected sample, so each corrected value is paired
with its actual collection time. Ambiguous barcodes, duplicate tubes, and
non-bijective swaps raise an error. Set ``allow_unmatched=False`` when a complete
one-to-one match is required.

Compute response features
-------------------------

.. code-block:: python

   features = cw.saliva.compute_features(
       merged,
       saliva_type="cortisol",
       slope_pairs=[("B1", "B2"), ("B1", "B4")],
   )

The function returns one row per study, participant, and day. It uses the
participant-specific ``time`` column by default. Missing values produce missing
AUCs instead of silently integrating an incomplete curve. Sampling times must
be strictly increasing.

Audit the result
----------------

Persist the merge provenance with any derived features. In particular, inspect
``merge_status``, ``match_method``, ``sample_mismatch``, and
``mismatch_corrected`` before statistical analysis.
