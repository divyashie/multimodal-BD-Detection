# multimodal-BD-Detection

> **Status:** Re-scoped. This was an attempt to build a multimodal bipolar-disorder state
> classifier. It reported 82.6% macro-F1 and passed four standard validation checks. It does
> not measure bipolar state, and the checks could not have revealed that. The repository now
> documents how that happened. The original code is preserved in [`legacy/`](./legacy).

## What this project is

A worked failure case in multimodal mental-health machine learning, with diagnostics.

A classifier was built by fusing four public corpora — two physiological, two text — none of
which share participants. It reached 82.6% macro-F1 and passed label shuffling, a fusion ladder,
noise injection, and calibration. Re-running the pipeline with provenance retained shows three
independent mechanisms, each sufficient on its own to produce that result:

1. **The pairing was vacuous.** Rows from different corpora were paired at random within each
   class. A fused row shares nothing across modalities except its label. Permuting one modality
   *within* each class leaves performance unchanged to three decimals (0.828 → 0.828), so no
   cross-modal information was ever present. The conventional shuffle test permutes across all
   classes and therefore cannot detect this.

2. **Source-file identity was laundered into the labels.** The OBF-Psychiatric download holds a
   derived feature table covering three of five diagnostic groups, plus per-group metadata
   tables. These were concatenated vertically rather than joined on the participant key, and the
   resulting gaps filled with column means. No participant in the dataset has both halves of the
   8-feature vector used — it was not merely unmeasured but unmeasurable. The clustering that
   produced the labels separated rows by which file they came from; the "Mania" class is 46 of 47
   rows from two clinical-only files. Missingness pattern alone predicts the label at 0.542
   macro-F1 against a 0.333 floor.

3. **The physiological label mapping did not survive inspection.** WESAD condition codes were
   mapped to bipolar states by hand. The resulting largest class is 39,507 windows of undefined
   inter-condition time; the "manic" class pools stress and meditation.

Two further mechanisms were hypothesised and **ruled out**: duplicate rows from resampling with
replacement (0.0% of test rows), and fully-imputed rows driving a class (all 78 fell in a
different cluster). Both are reported as negative results.

## What the data actually supports

Rebuilt correctly — person-day features computed from the raw activity series for all 162
participants, joined on the participant key, with complete-day and wear-time filters stated —
OBF-Psychiatric supports a modest, honest benchmark:

| Task | Macro-F1 | Chance |
| --- | --- | --- |
| 5-class diagnosis, subject-grouped | 0.314 ± 0.008 | 0.200 |
| 5-class diagnosis, random person-day split | 0.400 ± 0.004 | 0.200 |
| 3-class (control / depression / schizophrenia), subject-grouped | 0.523 ± 0.016 | 0.333 |

The gap between the grouped and random splits on identical data is itself an illustration of why
subject-grouped evaluation matters.

## Output

Two papers.

- **Paper 1** (in preparation): the failure case, the three mechanisms, and a seven-item
  reporting checklist. Target: medRxiv preprint, then JMIR Mental Health. Outline and
  experiments in [`methods-paper/`](./methods-paper).
- **Paper 2** (planned): the checklist implemented as a validated diagnostic package, tested
  against simulations with known ground truth.

## Repository structure

```
.
├── README.md                  This file.
├── RETROSPECTIVE.md           Postmortem of the original project.
├── methods-paper/             Current work.
│   ├── outline.md             The paper's structure and argument.
│   └── experiments/           Diagnostic reanalysis (Colab), and its JSON output.
└── legacy/                    The original pipeline, preserved and not maintained.
    ├── notebooks/             The fusion and control notebooks as they ran.
    ├── scripts/ training/     Original pipeline code.
    ├── configs/ data/ models/
    ├── VALIDATION_AND_RESULTS.md   Superseded results document, kept for the record.
    └── ORIGINAL_README.md
```

## Reproducing the analysis

The diagnostics run in Google Colab against the source corpora in Drive. See
[`methods-paper/experiments/README.md`](./methods-paper/experiments/README.md). Every number in
the paper traces to `results.json` or `section6.json`.

The corpora are not redistributed here. WESAD and OBF-Psychiatric are available from their
authors under their own terms; see `RETROSPECTIVE.md` for the full source list.

## Datasets

- **WESAD** — Schmidt et al., ICMI 2018. Wearable stress and affect detection, 15 subjects.
- **OBF-Psychiatric** — Garcia-Ceja et al., *Scientific Data* 2025. Motor activity for 162
  people across ADHD, clinical, depression, schizophrenia and control groups.
- Three public text corpora (Reddit and Twitter), labelled by corpus authors or by the original
  pipeline itself. Label provenance for each is given in the paper.

## Citation

See [`CITATION.cff`](./CITATION.cff).