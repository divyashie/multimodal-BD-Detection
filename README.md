# multimodal-BD-Detection

> **Status:** Active. The repository has been re-scoped from its original goal (a multimodal bipolar-disorder detector) into a methods-focused project. The original codebase is preserved in [`legacy/`](./legacy) for reproducibility and transparency. The current work is described below.

## What this project is now

A study of how **standard internal validation checks fail to detect construct mismatch** when multimodal mental-health detection systems are built by fusing public corpora that do not share subjects, clinical labels, or even the underlying construct being measured.

Concretely, this repository:

1. Documents a concrete failure case — an attempt to fuse a healthy-subject physiological corpus (WESAD-style stress responses) with self-reported text from a clinically unrelated population in order to "detect bipolar disorder."
2. Shows that the resulting system passes the validation conventions widely used in the multimodal mental-health ML literature: label shuffling drops accuracy only modestly (~5.3%), calibration error is low (ECE ≈ 0.048), and standard train/test splits report headline numbers that look publishable.
3. Argues that these conventions are necessary but not sufficient. They do not detect that the system's labels were never grounded in clinical reality — and that a system can satisfy all of them while answering a different question than the one its abstract claims to answer.
4. Proposes a small set of additional diagnostic checks (out-of-population validation, construct-alignment audit, label-provenance disclosure) that would have caught the failure earlier.

The output is a single methods paper, drafted in [`methods-paper/`](./methods-paper).

## Why this matters

Multimodal mental-health ML is increasingly built by fusing public corpora. The convenience is real; the danger is that label spaces, populations, and constructs are rarely aligned across sources. Papers can clear internal validation while producing systems with no real clinical signal. As the field moves toward larger models trained on more heterogeneous data, the gap between "passes validation" and "captures the intended construct" widens.

This repository is one worked example of that gap and a proposal for diagnostics that close it.

## Repository structure

```
.
├── README.md                       This file.
├── RETROSPECTIVE.md                Honest postmortem of the original project.
├── legacy/                         Original code, configs, and notebooks, preserved.
├── methods-paper/                  Current work: the cautionary methods paper.
└── ...
```

## How to read this repository

- For **the scientific argument**, read [`methods-paper/outline.md`](./methods-paper/outline.md).
- For **the postmortem**, read [`RETROSPECTIVE.md`](./RETROSPECTIVE.md).
- For **the original code as it existed before re-scoping**, see [`legacy/`](./legacy).

## Status

| Milestone                                       | Status      |
|-------------------------------------------------|-------------|
| Original BD-detection project                   | Archived in `legacy/` |
| Retrospective written                           | Done        |
| Methods-paper outline                           | Done        |
| Re-run of original system with full diagnostics | In progress |
| Methods-paper draft v1                          | In progress |
| Methods-paper submission                        | Targeted: JMIR Mental Health / ML4H |

## Contact

Open an issue, or contact the author through the address on the GitHub profile.
