# Retrospective: what the original multimodal-BD-Detection project got wrong

This document is a postmortem of the original version of this repository, which attempted to build a multimodal classifier for bipolar-disorder states by fusing physiological signals from the WESAD dataset with self-reported mental-health text from public Reddit and Twitter corpora. The project ran from its first commit through late 2025 and was re-scoped because of the methodological problems described below.

The failure is more useful as documentation than as deleted commits. The current methods paper builds directly on what is described here.

## What the project tried to do

- Use text from public Reddit and Twitter corpora that had been labeled — by corpus authors, not clinicians — with mental-health categories including variants of bipolar, depression, and euthymia.
- Use physiological signals (BVP, EDA, accelerometer, temperature) from WESAD, in which eleven healthy adults were recorded during lab-induced stress, amusement, and baseline paradigms.
- Align the modalities by clustering physiological segments into latent affective states with a Gaussian mixture model, then assigning text-derived clinical labels to physio clusters via a learned mapping.
- Train a fusion classifier and validate it with internal checks: label shuffling, noise injection, train/test splits, expected calibration error (ECE).

Reported results passed those checks. Accuracy dropped about 5.3% under label shuffling; ECE was ≈ 0.048; train/test splits produced headline numbers that looked publishable.

## What was actually wrong

The numbers were unremarkable because the *checks* were unremarkable, not because the *system* worked. Four overlapping problems:

### 1. Population mismatch

WESAD recorded eleven healthy adults in a lab. None had bipolar disorder, depression, or any clinical mood diagnosis. The physiological signals reflect acute stress and amusement in healthy people. The text corpora sampled self-identified Reddit and Twitter users whose posts mentioned mood-disorder vocabulary. The two populations have no demographic, clinical, or temporal overlap. Calling the latent GMM clusters "bipolar states" was a labeling decision dressed as a measurement.

### 2. Label-space mismatch

WESAD labels are affective states (stress / amusement / neutral / meditation). The text labels are clinical-sounding categories drawn from user-generated content. These are not different views of the same construct; they are different constructs that share some vocabulary. The GMM alignment forced a mapping, but it was made by the optimizer, not by clinical theory.

### 3. No within-subject linkage

In a properly designed multimodal mental-health study (CALYPSO, TIMEBASE, mindLAMP, StudentLife, BiAffect, SNAPSHOT), every multimodal row is one person providing all modalities. Here, no row corresponded to one person who had produced both a text post and a physiological recording. The "subjects" of the fused dataset were synthetic.

### 4. Construct invalidity dressed as construct validity

Shuffling, noise injection, and ECE measure whether the model has learned a function of the input that is robust to specific perturbations. They do not measure whether the function corresponds to the construct in the paper's abstract. A model that has learned "Reddit posts about manic episodes use different word frequencies than depressive episodes" will pass all of these checks and report nothing about bipolar disorder.

## What survives

1. **The pipeline.** Useful as a concrete, runnable example of how this kind of fusion is typically built. The methods paper requires that worked example.
2. **The validation battery.** Shuffling, noise injection, calibration, and split-based evaluation are standard and necessary. The methods paper does not argue against them; it argues they are insufficient.
3. **The honest record.** A repository that quietly disappears communicates less than one that explains what happened.

## What the project will not become

It will not become a "fixed" multimodal BD detector by acquiring better data. Construct-validity problems are not patched by more data; they are patched by data with shared subjects and shared clinical labels, of the kind held by CALYPSO, TIMEBASE, BiAffect, mindLAMP, and SNAPSHOT. Access to those datasets requires institutional affiliations, ethics approvals, and a different scientific framing than this repository can support as a solo, public artifact.

The contribution from here forward is methodological.

## Acknowledgments

The original framing was developed in consultation with a faculty advisor during 2024–2025. The decision to re-scope the project as a methods paper, and all responsibility for the arguments in this retrospective, are the author's alone.

## Reading next

- [`methods-paper/outline.md`](./methods-paper/outline.md) — the current direction.
- [`legacy/`](./legacy) — the original codebase.
