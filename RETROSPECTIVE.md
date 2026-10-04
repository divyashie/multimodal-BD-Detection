# Retrospective

This document records what the original `multimodal-BD-Detection` pipeline did, what it
reported, and what a provenance-preserving reanalysis found. It replaces an earlier version of
this file that described the project inaccurately — it named two corpora where four were used,
omitted OBF-Psychiatric entirely, and described the fusion step as a learned alignment when the
code performs random pairing within label.

It is written as a factual record rather than an apology. The errors below are specific,
reproducible, and quantified; that is what makes them useful.

## 1. What the project set out to do

Build a classifier for bipolar-disorder states (euthymia / depression / mania) by fusing
physiological signals with mental-health text, using public corpora, and validate it with the
internal checks conventional in multimodal mental-health ML.

## 2. What was actually built

**Four corpora**, none sharing participants with any other:

| Corpus | Modality | Content | Who assigned the labels |
| --- | --- | --- | --- |
| WESAD | Physiological | Wearable signals, 15 subjects, lab stress protocol | Corpus authors (condition codes), remapped by hand to BD states |
| OBF-Psychiatric | Physiological | Motor activity, 162 people (ADHD, clinical, depression, schizophrenia, control) | This pipeline, by Gaussian mixture over autoencoder latents |
| Kaggle bipolar Reddit | Text | Reddit posts | This pipeline, by *k*-means over BERT embeddings, named by VADER sentiment |
| Multi-class depression Twitter | Text | Tweets | Corpus authors, remapped by hand |

Three of the four label sets were therefore assigned by the pipeline itself or by a hand mapping,
not by clinicians.

**Fusion.** For each class, equal numbers of text and physiological rows were drawn without
replacement and concatenated in draw order. A fused row is one text post and one unrelated
physiological recording that happen to share a label. Nothing else relates them. Class sizes were
equalised to the smallest available, yielding 141 rows, 47 per class.

**Model and evaluation.** Random forest, 200 trees, balanced class weights, stratified 80/20
split at seed 42. Isotonic calibration fitted on the training set of a prefit forest; expected
calibration error computed on the 29 test rows.

## 3. What was reported

Macro-F1 of 0.756 text-only, 0.823 physiological-only, 0.826 fused. Four validation checks
passed: label shuffling cost 5.3%, the fusion ladder was incremental, noise injection degraded
performance to the text-only baseline, and calibrated ECE was 0.048. These numbers are
reproducible — the reanalysis recovers all three headline figures to three decimals.

## 4. What the reanalysis found

Three mechanisms, each on its own sufficient to produce the reported result.

### 4.1 The pairing carried no cross-modal information

The conventional shuffle test permutes one modality across all rows, which destroys that
modality's label-marginal information. Permuting *within* each class instead leaves the joint
distribution unchanged if rows were paired by label alone.

| Condition | Repeated 5-fold CV |
| --- | --- |
| Aligned | 0.828 ± 0.012 |
| Within-class shuffle | 0.828 ± 0.016 |
| Across-class shuffle | 0.752 ± 0.013 |

Identical to three decimals, and identical at every stage of a correction waterfall. The reported
5.3% drop measured only that physiological features carry label information on their own. No
alignment claim was supportable at any point.

### 4.2 Source-file identity was laundered into the labels

The OBF-Psychiatric download contains a derived feature table (`features.csv`), per-group
metadata tables, and raw per-participant activity series. The pipeline concatenated the CSVs
vertically instead of joining them on the participant key, then filled the resulting gaps with
column means.

The 8-feature vector it used could not exist for anyone. `features.csv` covers exactly the 77
people in the control, depression and schizophrenia groups — none of which carry clinical rating
scales. The 85 people who do have `madrs`, `hads_d`, `asrs` and `mdq_pos` (the ADHD and clinical
groups) have no rows in that table at all. Every row of the fused vector was therefore half
measurement and half column mean, and which half was constant identified the source file.

The mixture model clustered that structure. The cluster that became "Mania" — and which capped
the fused dataset at 141 rows — is 46 of 47 rows from the two clinical-only files. Within the
fused data, that class has 2 unique actigraphy vectors across 47 rows; the "Euthymia" class has 1
unique clinical vector across 47. Missingness pattern alone predicts the cluster label at 0.542
macro-F1 against a 0.333 floor. Agreement with the real diagnoses already present in the dataset
is ARI 0.059.

Separately, OBF contains no bipolar patients in its diagnosis field: the groups are control,
depression and schizophrenia, with ADHD and a mixed clinical sample. An explicit `bipolar` flag
exists for the 85 ADHD and clinical participants, and was never used.

### 4.3 The physiological label mapping did not survive inspection

WESAD condition codes were mapped to bipolar states by hand:

| Assigned state | Actual WESAD condition (windows) |
| --- | --- |
| Euthymia | transient/undefined 39,507; amusement 5,575 |
| Depression | baseline 17,611; code 6 790 |
| Mania | meditation 11,806; stress 9,966 |

The largest class is mostly undefined inter-condition time. The "manic" class pools the two
conditions furthest apart in arousal. Evaluated on WESAD alone with the real condition labels,
leave-one-subject-out gives 0.532 against a 0.333 floor, while a random-window split gives 0.951
— the split, not the signal, produced most of the apparent performance.

Note also that the analysis ran on 11 of WESAD's 15 subjects. Four participant folders (S6–S9)
had been extracted without their `.pkl` files, leaving only the raw Empatica archive and
questionnaire. The loader reported each skip to stdout and continued, and the reduced sample
size was never checked against the dataset documentation. Every WESAD figure in the original
work therefore rests on 73% of the available data. The missing files were recoverable from the
dataset archive, so the analysis below was run twice: once on the 11-subject copy the pipeline
actually used, and once on all 15. Figures describing WESAD itself are taken from the complete
run; figures describing the original pipeline are taken from the as-run copy. Both are published
in `methods-paper/experiments/`.

### 4.4 Label circularity

The Kaggle bipolar labels were *k*-means clusters of the same BERT embeddings later used as model
input, and are recovered exactly (1.000) by nearest centroid — deterministic, not merely
correlated. The OBF labels were derived from features that were then fed back in as inputs.

### 4.5 Source confounding in the text side

A probe predicting which corpus a post came from, using only the embeddings, reaches 0.936
macro-F1. Reddit and Twitter are trivially separable, and the two corpora have different label
distributions.

### 4.6 The calibration result was a reporting artefact

ECE on 29 test rows is 0.048 at 5 bins, 0.104 at 10, and 0.178 at 15; the 5-bin 95% bootstrap CI
is [0.027, 0.217]. The calibrator was fitted on the training set of a prefit model. Fitted by
internal cross-validation and evaluated held-out over 20 splits, ECE is 0.131 [0.078, 0.185].
The headline figure was a bin-count choice on a test set far too small to support it.

## 5. What was hypothesised and ruled out

Both are recorded because negative results are part of the record.

**Duplicate-row leakage.** Class balancing used `sklearn.utils.resample`, whose default samples
with replacement; 29.3% of the balanced WESAD pool is duplicated rows. Predicted consequence:
identical rows on both sides of the split. Observed: 0.0% of test rows have an exact duplicate in
training. The mechanism is real but did not reach the fused data.

**Imputed rows driving a class.** 78 OBF rows had all eight features missing and were filled
entirely with column means. Predicted consequence: a constant vector repeated within one class.
Observed: all 78 fell in a single cluster, and the suspected class contains 47 distinct vectors.
Rejected. Pursuing this hypothesis is what surfaced the complementary-halves mechanism in 4.2,
which a whole-vector uniqueness check cannot see.

## 6. What the data actually supports

Rebuilt from the raw activity series for all 162 participants, joined on the participant key, and
filtered to complete days (1,440 minutes) with nonzero activity and no duplicate timestamps:
1,785 person-days from 2,206, with 335 incomplete days, 87 non-wear days and 6 days with
duplicate timestamps removed. The feature recipe reproduces the published `features.csv` exactly
where the two overlap.

| Task | Macro-F1 | Chance |
| --- | --- | --- |
| 5-class diagnosis, subject-grouped | 0.314 ± 0.008 | 0.200 |
| 5-class diagnosis, random person-day split | 0.400 ± 0.004 | 0.200 |
| 3-class (control / depression / schizophrenia), subject-grouped | 0.523 ± 0.016 | 0.333 |

Modest and above chance. The 0.086 gap between grouped and random splits on identical data shows
how much of a headline number subject-level evaluation can remove.

Two genuine same-person pairings were then tested to see whether the within-class permutation
test fires when pairing is real: actigraphy with clinical scales for 71 people (gap +0.006), and
co-registered WESAD chest and wrist sensors (0.446 ± 0.013 aligned against 0.459 ± 0.004 shuffled,
gap +0.013). Both null. The test is diagnostic of a claim rather than of a mechanism — a null
result means no alignment claim is supportable, which is what it showed for the original pipeline.

One caveat found while running that second control: under subject-grouped cross-validation the
permutation must be confined to (subject, class) cells. Permuting across all rows hands a test
row its second modality from a training subject, reintroducing the leakage the grouping removed;
the shuffled score then *rises* to 0.611. A shuffle that improves performance is a leakage
signature, not evidence of alignment.

## 7. What went wrong procedurally

- A relational join was performed as a concatenation, and mean imputation concealed it.
- Labels were derived from features that were then used as inputs, in two separate places.
- A hand-written label mapping was never checked against the source codebook.
- Validation thresholds and the ECE bin count were chosen after seeing results.
- Four of fifteen subjects were missing from an incomplete extraction; the loader reported the
  skips and carried on, and the sample size was never reconciled with the dataset documentation.
- A merge was validated on its own join key, which always reports zero unmatched rows.

None of these is exotic. All four conventional checks pass in their presence, because every one
of them takes the assembled feature matrix as given and never examines its assembly.

## 8. Status

The original pipeline is preserved in [`legacy/`](./legacy) and is not maintained. The current
work is a methods paper analysing this failure and proposing diagnostics that detect it; see
[`methods-paper/`](./methods-paper). All figures in this document trace to
`methods-paper/experiments/results.json` and `section6.json`.