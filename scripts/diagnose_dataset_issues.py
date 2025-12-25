#!/usr/bin/env python3
"""
Dataset Quality Diagnostic Tool

Analyzes preprocessing issues in ablation datasets:
1. Audio/video feature quality (zero vs non-zero)
2. Label distribution by source
3. Feature embedding space analysis
4. Class imbalance metrics

Usage:
    python scripts/diagnose_dataset_issues.py --data_dir all_modalities
"""

import os
import sys
import pickle
import numpy as np
import torch
import argparse
import logging
from pathlib import Path
from collections import Counter
import json
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDiagnostics:
    """Comprehensive dataset quality diagnostics"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.datasets = {}
        self.results = {}

    def load_datasets(self):
        """Load all available datasets"""
        logger.info("Loading datasets...")

        dataset_files = list(self.data_dir.glob("*.pkl"))

        # Filter out metadata
        dataset_files = [f for f in dataset_files if 'metadata' not in f.name.lower()]

        for filepath in dataset_files:
            dataset_name = filepath.stem
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

                if len(data) > 0:
                    self.datasets[dataset_name] = data
                    logger.info(f"  ✓ {dataset_name}: {len(data):,} samples")
                else:
                    logger.warning(f"  ✗ {dataset_name}: empty")

            except Exception as e:
                logger.error(f"  ✗ {dataset_name}: {e}")

        logger.info(f"Loaded {len(self.datasets)} non-empty datasets")

    def analyze_feature_quality(self):
        """Test 1: Check audio/video feature quality"""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Feature Quality Analysis")
        logger.info("="*80)

        results = {}

        for dataset_name, samples in self.datasets.items():
            logger.info(f"\nAnalyzing {dataset_name}...")

            # Compute feature statistics
            text_norms = []
            audio_norms = []
            video_norms = []
            physio_norms = []

            text_zeros = 0
            audio_zeros = 0
            video_zeros = 0
            physio_zeros = 0

            for sample in samples:
                # Text
                text_feat = sample['text']
                if isinstance(text_feat, torch.Tensor):
                    text_feat = text_feat.numpy()
                text_norm = np.linalg.norm(text_feat)
                text_norms.append(text_norm)
                if text_norm == 0:
                    text_zeros += 1

                # Audio
                audio_feat = sample['audio']
                if isinstance(audio_feat, torch.Tensor):
                    audio_feat = audio_feat.numpy()
                audio_norm = np.linalg.norm(audio_feat)
                audio_norms.append(audio_norm)
                if audio_norm == 0:
                    audio_zeros += 1

                # Video
                video_feat = sample['video']
                if isinstance(video_feat, torch.Tensor):
                    video_feat = video_feat.numpy()
                video_norm = np.linalg.norm(video_feat)
                video_norms.append(video_norm)
                if video_norm == 0:
                    video_zeros += 1

                # Physio
                physio_feat = sample['physio']
                if isinstance(physio_feat, torch.Tensor):
                    physio_feat = physio_feat.numpy()
                physio_norm = np.linalg.norm(physio_feat)
                physio_norms.append(physio_norm)
                if physio_norm == 0:
                    physio_zeros += 1

            n = len(samples)

            # Report
            logger.info(f"  Text features:")
            logger.info(f"    Norm: mean={np.mean(text_norms):.4f}, std={np.std(text_norms):.4f}")
            logger.info(f"    All-zero: {text_zeros}/{n} ({100*text_zeros/n:.1f}%)")

            logger.info(f"  Audio features:")
            logger.info(f"    Norm: mean={np.mean(audio_norms):.4f}, std={np.std(audio_norms):.4f}")
            logger.info(f"    All-zero: {audio_zeros}/{n} ({100*audio_zeros/n:.1f}%)")

            logger.info(f"  Video features:")
            logger.info(f"    Norm: mean={np.mean(video_norms):.4f}, std={np.std(video_norms):.4f}")
            logger.info(f"    All-zero: {video_zeros}/{n} ({100*video_zeros/n:.1f}%)")

            logger.info(f"  Physio features:")
            logger.info(f"    Norm: mean={np.mean(physio_norms):.4f}, std={np.std(physio_norms):.4f}")
            logger.info(f"    All-zero: {physio_zeros}/{n} ({100*physio_zeros/n:.1f}%)")

            # Store results
            results[dataset_name] = {
                'text': {
                    'mean_norm': float(np.mean(text_norms)),
                    'std_norm': float(np.std(text_norms)),
                    'pct_zero': float(100 * text_zeros / n)
                },
                'audio': {
                    'mean_norm': float(np.mean(audio_norms)),
                    'std_norm': float(np.std(audio_norms)),
                    'pct_zero': float(100 * audio_zeros / n)
                },
                'video': {
                    'mean_norm': float(np.mean(video_norms)),
                    'std_norm': float(np.std(video_norms)),
                    'pct_zero': float(100 * video_zeros / n)
                },
                'physio': {
                    'mean_norm': float(np.mean(physio_norms)),
                    'std_norm': float(np.std(physio_norms)),
                    'pct_zero': float(100 * physio_zeros / n)
                }
            }

            # Warnings
            if audio_zeros / n > 0.5:
                logger.warning(f"    ⚠️  >50% audio features are all-zero!")
            if video_zeros / n > 0.5:
                logger.warning(f"    ⚠️  >50% video features are all-zero!")

        self.results['feature_quality'] = results

    def analyze_label_distribution(self):
        """Test 2: Check label distribution by source"""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Label Distribution by Source")
        logger.info("="*80)

        results = {}

        label_names = {0: 'Depression', 1: 'Euthymia', 2: 'Mania'}

        for dataset_name, samples in self.datasets.items():
            logger.info(f"\n{dataset_name}:")

            # Overall distribution
            all_labels = [s['label'] for s in samples]
            label_dist = Counter(all_labels)

            logger.info(f"  Overall: {len(samples)} samples")
            for label_id, count in sorted(label_dist.items()):
                pct = 100 * count / len(samples)
                logger.info(f"    {label_names[label_id]}: {count} ({pct:.1f}%)")

            # By source (if available)
            if 'source' in samples[0]:
                by_source = {}
                for sample in samples:
                    source = sample['source']
                    if source not in by_source:
                        by_source[source] = []
                    by_source[source].append(sample['label'])

                logger.info(f"  By source:")
                source_results = {}
                for source, labels in by_source.items():
                    label_dist_source = Counter(labels)
                    logger.info(f"    {source}: {len(labels)} samples")

                    source_dist = {}
                    for label_id, count in sorted(label_dist_source.items()):
                        pct = 100 * count / len(labels)
                        logger.info(f"      {label_names[label_id]}: {count} ({pct:.1f}%)")
                        source_dist[label_names[label_id]] = {
                            'count': count,
                            'percentage': float(pct)
                        }
                    source_results[source] = source_dist

                results[dataset_name] = {
                    'total_samples': len(samples),
                    'overall_distribution': {label_names[k]: v for k, v in label_dist.items()},
                    'by_source': source_results
                }
            else:
                results[dataset_name] = {
                    'total_samples': len(samples),
                    'overall_distribution': {label_names[k]: v for k, v in label_dist.items()}
                }

            # Check for severe imbalance
            min_count = min(label_dist.values())
            max_count = max(label_dist.values())
            imbalance_ratio = max_count / min_count

            if imbalance_ratio > 5:
                logger.warning(f"    ⚠️  Severe class imbalance! Ratio: {imbalance_ratio:.1f}:1")
                logger.warning(f"       Smallest class has only {min_count} samples")

        self.results['label_distribution'] = results

    def analyze_class_imbalance(self):
        """Test 3: Detailed class imbalance metrics"""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: Class Imbalance Analysis")
        logger.info("="*80)

        label_names = {0: 'Depression', 1: 'Euthymia', 2: 'Mania'}
        results = {}

        for dataset_name, samples in self.datasets.items():
            logger.info(f"\n{dataset_name}:")

            # Get labels
            labels = np.array([s['label'] for s in samples])

            # Compute imbalance metrics
            unique, counts = np.unique(labels, return_counts=True)

            # Imbalance ratio
            max_count = np.max(counts)
            min_count = np.min(counts)
            imbalance_ratio = max_count / min_count

            # Entropy (higher = more balanced)
            probs = counts / len(labels)
            entropy = -np.sum(probs * np.log2(probs))
            max_entropy = np.log2(len(unique))
            normalized_entropy = entropy / max_entropy

            logger.info(f"  Class counts:")
            for label_id, count in zip(unique, counts):
                logger.info(f"    {label_names[label_id]}: {count}")

            logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
            logger.info(f"  Entropy: {entropy:.3f} / {max_entropy:.3f} ({normalized_entropy:.1%} of max)")

            # Estimate train/val/test split
            train_size = int(0.7 * len(labels))
            val_size = int(0.15 * len(labels))
            test_size = len(labels) - train_size - val_size

            logger.info(f"  Estimated split (70/15/15):")
            logger.info(f"    Train: {train_size} samples")
            logger.info(f"    Val: {val_size} samples")
            logger.info(f"    Test: {test_size} samples")

            # Warning for small minority class
            minority_train = int(0.7 * min_count)
            if minority_train < 50:
                logger.warning(f"    ⚠️  Minority class ({label_names[np.argmin(counts)]}) has only ~{minority_train} training samples!")

            results[dataset_name] = {
                'imbalance_ratio': float(imbalance_ratio),
                'normalized_entropy': float(normalized_entropy),
                'minority_class': label_names[int(np.argmin(counts))],
                'minority_count': int(min_count),
                'majority_class': label_names[int(np.argmax(counts))],
                'majority_count': int(max_count)
            }

        self.results['class_imbalance'] = results

    def analyze_feature_correlation(self):
        """Test 4: Check if features are informative"""
        logger.info("\n" + "="*80)
        logger.info("TEST 4: Feature-Label Correlation Analysis")
        logger.info("="*80)

        results = {}

        for dataset_name, samples in self.datasets.items():
            logger.info(f"\n{dataset_name}:")

            # Extract features and labels
            labels = np.array([s['label'] for s in samples])

            # Text features
            text_feats = []
            for s in samples:
                feat = s['text']
                if isinstance(feat, torch.Tensor):
                    feat = feat.numpy()
                text_feats.append(feat)
            text_feats = np.array(text_feats)

            # Audio features
            audio_feats = []
            for s in samples:
                feat = s['audio']
                if isinstance(feat, torch.Tensor):
                    feat = feat.numpy()
                audio_feats.append(feat)
            audio_feats = np.array(audio_feats)

            # Video features
            video_feats = []
            for s in samples:
                feat = s['video']
                if isinstance(feat, torch.Tensor):
                    feat = feat.numpy()
                video_feats.append(feat)
            video_feats = np.array(video_feats)

            # Compute variance (low variance = uninformative)
            text_var = np.mean(np.var(text_feats, axis=0))
            audio_var = np.mean(np.var(audio_feats, axis=0))
            video_var = np.mean(np.var(video_feats, axis=0))

            logger.info(f"  Feature variance (mean across dimensions):")
            logger.info(f"    Text: {text_var:.6f}")
            logger.info(f"    Audio: {audio_var:.6f}")
            logger.info(f"    Video: {video_var:.6f}")

            # Check for zero variance
            text_zero_var = np.sum(np.var(text_feats, axis=0) == 0)
            audio_zero_var = np.sum(np.var(audio_feats, axis=0) == 0)
            video_zero_var = np.sum(np.var(video_feats, axis=0) == 0)

            logger.info(f"  Dimensions with zero variance:")
            logger.info(f"    Text: {text_zero_var}/{text_feats.shape[1]}")
            logger.info(f"    Audio: {audio_zero_var}/{audio_feats.shape[1]}")
            logger.info(f"    Video: {video_zero_var}/{video_feats.shape[1]}")

            # Warnings
            if audio_var < 0.001:
                logger.warning(f"    ⚠️  Audio features have very low variance - likely uninformative!")
            if video_var < 0.001:
                logger.warning(f"    ⚠️  Video features have very low variance - likely uninformative!")

            results[dataset_name] = {
                'text_variance': float(text_var),
                'audio_variance': float(audio_var),
                'video_variance': float(video_var),
                'text_zero_var_dims': int(text_zero_var),
                'audio_zero_var_dims': int(audio_zero_var),
                'video_zero_var_dims': int(video_zero_var)
            }

        self.results['feature_correlation'] = results

    def generate_report(self, output_path: str):
        """Generate comprehensive diagnostic report"""
        logger.info("\n" + "="*80)
        logger.info("GENERATING DIAGNOSTIC REPORT")
        logger.info("="*80)

        # Save results as JSON
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Saved detailed results to: {output_path}")

        # Generate summary
        logger.info("\n" + "="*80)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("="*80)

        for dataset_name in self.datasets.keys():
            logger.info(f"\n{dataset_name}:")

            # Feature quality
            if 'feature_quality' in self.results:
                fq = self.results['feature_quality'][dataset_name]
                issues = []
                if fq['audio']['pct_zero'] > 50:
                    issues.append(f"❌ Audio features {fq['audio']['pct_zero']:.0f}% zero")
                if fq['video']['pct_zero'] > 50:
                    issues.append(f"❌ Video features {fq['video']['pct_zero']:.0f}% zero")

                if issues:
                    logger.warning("  Feature Quality Issues:")
                    for issue in issues:
                        logger.warning(f"    {issue}")
                else:
                    logger.info("  ✓ Feature quality: OK")

            # Class imbalance
            if 'class_imbalance' in self.results:
                ci = self.results['class_imbalance'][dataset_name]
                if ci['imbalance_ratio'] > 5:
                    logger.warning(f"  ⚠️  Class imbalance: {ci['imbalance_ratio']:.1f}:1")
                    logger.warning(f"     Minority class: {ci['minority_class']} ({ci['minority_count']} samples)")
                else:
                    logger.info(f"  ✓ Class balance: {ci['imbalance_ratio']:.1f}:1 (acceptable)")

        logger.info("\n" + "="*80)

    def run_all_diagnostics(self):
        """Run all diagnostic tests"""
        self.load_datasets()

        if len(self.datasets) == 0:
            logger.error("No datasets found!")
            return

        self.analyze_feature_quality()
        self.analyze_label_distribution()
        self.analyze_class_imbalance()
        self.analyze_feature_correlation()

        # Generate report
        output_path = self.data_dir / 'diagnostic_report.json'
        self.generate_report(str(output_path))


def main():
    parser = argparse.ArgumentParser(description='Diagnose dataset quality issues')
    parser.add_argument('--data_dir', type=str, default='all_modalities',
                       help='Directory containing ablation datasets')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("DATASET DIAGNOSTICS TOOL")
    logger.info("="*80)
    logger.info(f"Data directory: {args.data_dir}\n")

    diagnostics = DatasetDiagnostics(args.data_dir)
    diagnostics.run_all_diagnostics()

    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTICS COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
