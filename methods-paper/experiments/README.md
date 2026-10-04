# Experiments

Diagnostic reanalysis of the original pipeline. Written to run in Google Colab with Drive
mounted; see the configuration block in cell 0 for paths.

## Files

| File | What it is |
| --- | --- |
| `01_diagnostics_waterfall.ipynb` | The whole analysis. Part A re-scores the original saved models; Part B rebuilds the pipeline with provenance retained; Section 6 builds the corrected dataset. |
| `run_as_run/` | Output of the run using the WESAD copy the original pipeline actually had (11 of 15 subjects). |
| `run_complete/` | Output of the run using all 15 WESAD subjects, extracted from `WESAD.zip`. |

Each run directory holds `results.json`, `section6.json`, and two derived CSVs
(`obf_corrected_person_day.csv`, `obf_paired_person_level.csv`).

## Why there are two runs

Four WESAD participant folders (S6–S9) had been extracted without their `.pkl` files, so the
original pipeline ran on 11 of the 15 available subjects without this being noticed. The missing
files were recoverable from the dataset archive.

Both runs are kept because they answer different questions. `run_as_run` reproduces what the
original pipeline did, and is the source for any claim *about that pipeline*. `run_complete` uses
the full dataset, and is the source for any claim *about WESAD itself*.

| Claim | Run to cite |
| --- | --- |
| Part A, B4 replication, B6 waterfall, B7 calibration | `run_as_run` |
| B1 condition × state, B5 leave-one-subject-out, 6.6 chest/wrist control | `run_complete` |
| B2, B3, B8, B9, Section 6 (OBF only — no WESAD dependency) | identical in both |

The fused dataset is capped at 141 rows by the smallest OBF cluster, so the headline 0.826 figure
comes from `run_as_run`; the `complete` pass draws a different WESAD pool and gives 0.794. Neither
is more correct — they describe different things.

## Running it

Set `WESAD_SOURCE` in cell 0 to `'as_run'` or `'complete'`, which selects both the WESAD path and
the output directory. For `'complete'`, extract `WESAD.zip` to `/content/WESAD_extracted` first;
`/content` is wiped whenever the runtime restarts.

BERT embeddings and WESAD window features are cached in a shared `cache/` directory, keyed by
source. The first run takes roughly an hour; later runs are much faster.

## Notes

- Row order in the OBF concatenation comes from an unsorted `os.listdir`, deliberately, because it
  replicates the original pipeline's ordering.
- `perm_within_cells` iterates `np.unique` rather than a Python set, so the permutation is
  deterministic given the seed. An earlier version iterated a set of strings, whose order varies
  between processes, and produced a slightly different figure each session.
- Two deviations from the original run are unavoidable: torch was never seeded in the original,
  and scikit-learn defaults have changed since. Headline figures still reproduce to three
  decimals.

## Reproducing from the JSON

Every number in the paper traces to `results.json` or `section6.json`. The figures can be redrawn
from those files without re-running anything.