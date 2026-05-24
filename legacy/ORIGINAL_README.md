# Multimodal Mental Health Classification

This repository contains a modularized PyTorch pipeline for multimodal classification (text, audio, video).

## Structure
- `configs/` → Configurations
- `data/` → Dataset, preprocessing, splitting utilities
- `models/` → Model architectures & losses
- `training/` → Trainer & evaluator
- `scripts/` → Main entry point

## Usage
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
