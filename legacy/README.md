# Legacy

This directory contains the original `multimodal-BD-Detection` codebase as it existed before the project was re-scoped into a methods paper.

It is preserved verbatim for three reasons:

1. **Transparency.** Deleting failed work from version control communicates less than keeping it.
2. **Reproducibility of the worked example.** The current methods paper uses this codebase as a runnable case study; the methods-paper experiments load from this directory.
3. **Honest record.** This is what the project looked like before its scientific framing was corrected.

## Layout

```
legacy/
├── configs/             Original training and pipeline configurations.
├── data/                Data-loading and preprocessing code.
├── models/              Model architectures and trained checkpoints.
├── notebooks/           Exploratory and analysis notebooks.
└── ORIGINAL_README.md   The original repo README, preserved.
```

## Not in here

- No external dataset files are committed. WESAD and the Reddit/Twitter corpora retain their original licenses and should be downloaded from their original sources following the relevant config files.

## How to read this today

Treat it as a frozen snapshot, not as code intended for continued development. For the post-mortem of what went wrong methodologically, see [`../RETROSPECTIVE.md`](../RETROSPECTIVE.md). For the current direction, see [`../methods-paper/`](../methods-paper).
