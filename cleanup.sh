#!/usr/bin/env bash
# Repository cleanup for multimodal-BD-Detection.
# Run from the repo root on a clean working tree. Review each section before running;
# this is written to be read and executed in pieces, not blindly.
set -euo pipefail

echo "== 1. Remove committed junk =="
# .DS_Store and __pycache__ should never have been tracked.
git rm -r --cached --quiet --ignore-unmatch \
  .DS_Store research_BD_states \
  legacy/configs/__pycache__ legacy/data/__pycache__ legacy/models/__pycache__ \
  scripts/__pycache__ training_backup/__pycache__ 2>/dev/null || true
find . -name .DS_Store -not -path './.git/*' -delete
find . -name __pycache__ -type d -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
rm -rf research_BD_states   # contained only a .DS_Store; the real dirs are gitignored

echo "== 2. Drop the duplicate training_backup/ =="
# Near-identical to training/; both are original-pipeline code. History retains it.
git rm -r --quiet --ignore-unmatch training_backup

echo "== 3. Move all original-pipeline code into legacy/ =="
# scripts/ and training/ import configs.config, data.dataset, models.* — modules that
# already live in legacy/. Keeping them at the root implies they run; they don't.
mkdir -p legacy/scripts
git mv training legacy/training
for f in scripts/*.py; do git mv "$f" legacy/scripts/; done
rmdir scripts 2>/dev/null || true

echo "== 4. Retire the superseded results document =="
git mv VALIDATION_AND_RESULTS.md legacy/VALIDATION_AND_RESULTS.md
cat > /tmp/val_header.md <<'EOF'
> **Superseded.** This document reports the original pipeline's validation results as
> written in 2025. Its conclusions do not hold: the 82.6% macro-F1 it reports is
> reproducible but does not measure bipolar state. See `RETROSPECTIVE.md` for what went
> wrong and `methods-paper/` for the analysis. This file is preserved unaltered below
> because the methods paper examines these specific claims.

EOF
cat /tmp/val_header.md legacy/VALIDATION_AND_RESULTS.md > /tmp/val.md
mv /tmp/val.md legacy/VALIDATION_AND_RESULTS.md

echo "== 5. Write .gitignore =="
cat > .gitignore <<'EOF'
# OS
.DS_Store

# Python
__pycache__/
*.py[cod]
.venv/
venv/

# Embedded backup repositories
research_BD_states/

# Data and large artifacts — never commit the corpora
data/
*.pkl
*.npz
*.npy

# Paper build artifacts
methods-paper/figures/*.pdf
methods-paper/figures/*.png
EOF

echo "== 6. Scaffold the methods-paper experiments =="
mkdir -p methods-paper/experiments methods-paper/figures
touch methods-paper/figures/.gitkeep
cat > methods-paper/experiments/README.md <<'EOF'
# Experiments

Diagnostic reanalysis of the original pipeline. Written to run in Google Colab with
Drive mounted; see the configuration block at the top of each file for paths.

| File | What it does |
| --- | --- |
| `diagnostics_01.py` | Parts A and B: reproduces the reported numbers, then measures the three mechanisms. Writes `results.json`. |
| `section6_corrected_dataset.py` | Builds the corrected 162-person OBF dataset and runs the positive control. Writes `section6.json`. |
| `results.json` | Output of `diagnostics_01.py`. All Results-section numbers trace to this file. |
| `section6.json` | Output of `section6_corrected_dataset.py`. |

Both are percent-format Python. Convert with `jupytext --to notebook <file>` to open as
a Colab notebook.
EOF

echo "== 7. Update requirements =="
cat > requirements.txt <<'EOF'
# Diagnostic reanalysis (methods-paper/experiments)
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.10
matplotlib>=3.7
transformers>=4.30
torch>=2.0
vaderSentiment>=3.3
jupytext>=1.15

# Original pipeline (legacy/) additionally used:
seaborn
EOF

echo
echo "Done. Still to do by hand:"
echo "  - CITATION.cff still has Lastname/Firstname/Affiliation/doi: TODO"
echo "  - README.md and RETROSPECTIVE.md describe 2 corpora; the pipeline used 5 (no OBF mentioned)"
echo "  - copy diagnostics_01.py and section6_corrected_dataset.py into methods-paper/experiments/"
echo "  - copy results.json and section6.json from Drive into methods-paper/experiments/"
echo
git status --short