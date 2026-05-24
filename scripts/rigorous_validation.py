#!/usr/bin/env python3
"""
Rigorous Multimodal Fusion Validation Framework

Implements 5 critical reliability checks:
1. ID-based shuffling control (CRITICAL)
2. Pairwise fusion ladder
3. Noise-injection control
4. Confidence calibration
5. Cross-dataset generalization

Usage:
    python rigorous_validation.py --models_dir /path/to/models --data_dir /path/to/data
"""

import numpy as np
import pandas as pd
import pickle
import argparse
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class MultimodalValidation:
    """Rigorous validation framework for multimodal fusion models"""

    def __init__(self, models_dir: str, results_dir: str = None):
        """
        Args:
            models_dir: Directory containing saved .pkl models
            results_dir: Directory to save validation results
        """
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir) if results_dir else self.models_dir / 'validation_results'
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Load models
        self.models = self._load_models()

        # Results storage
        self.results = {
            'shuffling_control': {},
            'pairwise_fusion': {},
            'noise_control': {},
            'calibration': {},
            'cross_dataset': {}
        }

    def _load_models(self) -> Dict:
        """Load all saved models"""
        models = {}

        model_files = {
            'text': 'text_model.pkl',
            'physio': 'physio_model.pkl',
            'fusion': 'fusion_model.pkl',
            'weighted_fusion': 'weighted_fusion_model.pkl'
        }

        for name, filename in model_files.items():
            path = self.models_dir / filename
            if path.exists():
                with open(path, 'rb') as f:
                    models[name] = pickle.load(f)
                print(f"✓ Loaded {name} model")
            else:
                print(f"⚠ {name} model not found at {path}")

        return models

    # =========================================================================
    # A. ID-BASED SHUFFLING CONTROL (CRITICAL)
    # =========================================================================

    def id_shuffling_control(
        self,
        text_aligned: np.ndarray,
        physio_aligned: np.ndarray,
        labels_aligned: np.ndarray,
        user_ids: np.ndarray = None,
        n_shuffles: int = 5
    ) -> Dict:
        """
        CRITICAL TEST: Shuffle one modality across users/samples

        Expected: Performance should collapse (proves semantic alignment matters)

        Args:
            text_aligned: Text features [n_samples, 768]
            physio_aligned: Physio features [n_samples, 34]
            labels_aligned: Labels [n_samples]
            user_ids: User/sample IDs [n_samples] (if None, uses indices)
            n_shuffles: Number of random shuffles

        Returns:
            Dict with shuffling results
        """
        print("\n" + "="*80)
        print("A. ID-BASED SHUFFLING CONTROL (CRITICAL)")
        print("="*80)

        if user_ids is None:
            user_ids = np.arange(len(labels_aligned))

        # Original (aligned) performance
        fusion_features = np.hstack([text_aligned, physio_aligned])

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            fusion_features, labels_aligned, test_size=0.2,
            stratify=labels_aligned, random_state=42
        )

        original_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        original_model.fit(X_train, y_train)
        original_f1 = f1_score(y_test, original_model.predict(X_test), average='macro')

        print(f"\n✓ Original (aligned) F1: {original_f1:.4f}")

        # Shuffled experiments
        shuffled_f1_scores = []

        for i in range(n_shuffles):
            # Shuffle physio across different users/samples
            shuffle_idx = np.random.permutation(len(physio_aligned))
            physio_shuffled = physio_aligned[shuffle_idx]

            # Create misaligned fusion
            fusion_shuffled = np.hstack([text_aligned, physio_shuffled])

            X_train_shuf, X_test_shuf, y_train_shuf, y_test_shuf = train_test_split(
                fusion_shuffled, labels_aligned, test_size=0.2,
                stratify=labels_aligned, random_state=42+i
            )

            shuffled_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42+i)
            shuffled_model.fit(X_train_shuf, y_train_shuf)
            shuffled_f1 = f1_score(y_test_shuf, shuffled_model.predict(X_test_shuf), average='macro')
            shuffled_f1_scores.append(shuffled_f1)

            print(f"  Shuffle {i+1}: F1 = {shuffled_f1:.4f}")

        mean_shuffled = np.mean(shuffled_f1_scores)
        std_shuffled = np.std(shuffled_f1_scores)

        print(f"\n✓ Shuffled (misaligned) F1: {mean_shuffled:.4f} ± {std_shuffled:.4f}")
        print(f"✓ Performance drop: {(original_f1 - mean_shuffled)*100:.2f}%")

        # Verdict
        if mean_shuffled < original_f1 - 0.05:  # At least 5% drop
            print("\n✅ PASS: Performance collapsed with shuffling")
            print("   → Model relies on semantic alignment, not spurious correlations")
        else:
            print("\n⚠️  WARNING: Performance did NOT collapse")
            print("   → Model may be exploiting dataset bias or class priors")

        results = {
            'original_f1': original_f1,
            'shuffled_f1_mean': mean_shuffled,
            'shuffled_f1_std': std_shuffled,
            'shuffled_f1_scores': shuffled_f1_scores,
            'performance_drop': original_f1 - mean_shuffled,
            'pass': mean_shuffled < original_f1 - 0.05
        }

        self.results['shuffling_control'] = results
        return results

    # =========================================================================
    # B. PAIRWISE FUSION LADDER
    # =========================================================================

    def pairwise_fusion_ladder(
        self,
        text_aligned: np.ndarray,
        physio_aligned: np.ndarray,
        labels_aligned: np.ndarray
    ) -> Dict:
        """
        Test all pairwise combinations to show incremental gains

        Configurations:
        - Text only
        - Physio only
        - Text + Physio (fusion)

        Expected: Incremental gains, not magic jumps
        """
        print("\n" + "="*80)
        print("B. PAIRWISE FUSION LADDER")
        print("="*80)

        from sklearn.model_selection import train_test_split

        results = {}

        # 1. Text only
        print("\n1. Text-Only Baseline")
        X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
            text_aligned, labels_aligned, test_size=0.2,
            stratify=labels_aligned, random_state=42
        )
        text_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        text_model.fit(X_train_t, y_train_t)
        text_f1 = f1_score(y_test_t, text_model.predict(X_test_t), average='macro')
        print(f"   F1: {text_f1:.4f}")
        results['text_only'] = {'f1': text_f1, 'dims': text_aligned.shape[1]}

        # 2. Physio only
        print("\n2. Physio-Only Baseline")
        X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
            physio_aligned, labels_aligned, test_size=0.2,
            stratify=labels_aligned, random_state=42
        )
        physio_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        physio_model.fit(X_train_p, y_train_p)
        physio_f1 = f1_score(y_test_p, physio_model.predict(X_test_p), average='macro')
        print(f"   F1: {physio_f1:.4f}")
        results['physio_only'] = {'f1': physio_f1, 'dims': physio_aligned.shape[1]}

        # 3. Text + Physio fusion
        print("\n3. Text + Physio Fusion")
        fusion_features = np.hstack([text_aligned, physio_aligned])
        X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
            fusion_features, labels_aligned, test_size=0.2,
            stratify=labels_aligned, random_state=42
        )
        fusion_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        fusion_model.fit(X_train_f, y_train_f)
        fusion_f1 = f1_score(y_test_f, fusion_model.predict(X_test_f), average='macro')
        print(f"   F1: {fusion_f1:.4f}")
        results['fusion'] = {'f1': fusion_f1, 'dims': fusion_features.shape[1]}

        # Summary
        print("\n" + "-"*80)
        print("PAIRWISE FUSION LADDER SUMMARY")
        print("-"*80)
        print(f"Text-Only:       {text_f1:.4f} ({text_aligned.shape[1]}D)")
        print(f"Physio-Only:     {physio_f1:.4f} ({physio_aligned.shape[1]}D)")
        print(f"Text+Physio:     {fusion_f1:.4f} ({fusion_features.shape[1]}D)")
        print(f"\nFusion gain over Text:   +{(fusion_f1-text_f1)*100:.2f}%")
        print(f"Fusion gain over Physio: +{(fusion_f1-physio_f1)*100:.2f}%")

        # Check for incremental gains
        if fusion_f1 > max(text_f1, physio_f1):
            print("\n✅ PASS: Fusion shows incremental gain")
        else:
            print("\n⚠️  WARNING: Fusion does not improve over best unimodal")

        self.results['pairwise_fusion'] = results
        return results

    # =========================================================================
    # C. NOISE-INJECTION CONTROL
    # =========================================================================

    def noise_injection_control(
        self,
        text_aligned: np.ndarray,
        physio_aligned: np.ndarray,
        labels_aligned: np.ndarray
    ) -> Dict:
        """
        Replace one modality with noise to prove information content matters

        Noise types:
        1. Gaussian noise
        2. Random embeddings
        3. Constant vectors

        Expected: Performance drops to unimodal or worse
        """
        print("\n" + "="*80)
        print("C. NOISE-INJECTION CONTROL")
        print("="*80)

        from sklearn.model_selection import train_test_split

        results = {}

        # Original fusion
        fusion_features = np.hstack([text_aligned, physio_aligned])
        X_train, X_test, y_train, y_test = train_test_split(
            fusion_features, labels_aligned, test_size=0.2,
            stratify=labels_aligned, random_state=42
        )
        original_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        original_model.fit(X_train, y_train)
        original_f1 = f1_score(y_test, original_model.predict(X_test), average='macro')

        print(f"\n✓ Original fusion F1: {original_f1:.4f}")

        # 1. Replace physio with Gaussian noise
        print("\n1. Physio → Gaussian Noise")
        noise_gaussian = np.random.randn(*physio_aligned.shape)
        fusion_noise_gaussian = np.hstack([text_aligned, noise_gaussian])

        X_train_ng, X_test_ng, y_train_ng, y_test_ng = train_test_split(
            fusion_noise_gaussian, labels_aligned, test_size=0.2,
            stratify=labels_aligned, random_state=42
        )
        model_ng = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        model_ng.fit(X_train_ng, y_train_ng)
        f1_ng = f1_score(y_test_ng, model_ng.predict(X_test_ng), average='macro')
        print(f"   F1: {f1_ng:.4f} (drop: {(original_f1-f1_ng)*100:.2f}%)")
        results['gaussian_noise'] = {'f1': f1_ng, 'drop': original_f1 - f1_ng}

        # 2. Replace physio with random embeddings (uniform)
        print("\n2. Physio → Random Embeddings")
        noise_random = np.random.uniform(-1, 1, physio_aligned.shape)
        fusion_noise_random = np.hstack([text_aligned, noise_random])

        X_train_nr, X_test_nr, y_train_nr, y_test_nr = train_test_split(
            fusion_noise_random, labels_aligned, test_size=0.2,
            stratify=labels_aligned, random_state=42
        )
        model_nr = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        model_nr.fit(X_train_nr, y_train_nr)
        f1_nr = f1_score(y_test_nr, model_nr.predict(X_test_nr), average='macro')
        print(f"   F1: {f1_nr:.4f} (drop: {(original_f1-f1_nr)*100:.2f}%)")
        results['random_embeddings'] = {'f1': f1_nr, 'drop': original_f1 - f1_nr}

        # 3. Replace physio with constant vectors
        print("\n3. Physio → Constant Vectors")
        noise_constant = np.zeros_like(physio_aligned)
        fusion_noise_constant = np.hstack([text_aligned, noise_constant])

        X_train_nc, X_test_nc, y_train_nc, y_test_nc = train_test_split(
            fusion_noise_constant, labels_aligned, test_size=0.2,
            stratify=labels_aligned, random_state=42
        )
        model_nc = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        model_nc.fit(X_train_nc, y_train_nc)
        f1_nc = f1_score(y_test_nc, model_nc.predict(X_test_nc), average='macro')
        print(f"   F1: {f1_nc:.4f} (drop: {(original_f1-f1_nc)*100:.2f}%)")
        results['constant_vectors'] = {'f1': f1_nc, 'drop': original_f1 - f1_nc}

        # Verdict
        mean_noise_drop = np.mean([results[k]['drop'] for k in results])
        print(f"\n✓ Average performance drop with noise: {mean_noise_drop*100:.2f}%")

        if mean_noise_drop > 0.05:  # At least 5% drop
            print("\n✅ PASS: Performance degrades with noise")
            print("   → Architecture doesn't just like more dimensions")
            print("   → Information content matters")
        else:
            print("\n⚠️  WARNING: Performance resilient to noise")
            print("   → Model may not be using the second modality effectively")

        results['original_f1'] = original_f1
        results['pass'] = mean_noise_drop > 0.05

        self.results['noise_control'] = results
        return results

    # =========================================================================
    # D. CONFIDENCE CALIBRATION
    # =========================================================================

    def confidence_calibration(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "Model"
    ) -> Dict:
        """
        Analyze prediction confidence calibration

        Metrics:
        - Reliability diagram
        - Expected Calibration Error (ECE)
        - Confidence distribution
        - High-confidence accuracy
        """
        print("\n" + "="*80)
        print(f"D. CONFIDENCE CALIBRATION - {model_name}")
        print("="*80)

        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        y_conf = np.max(y_proba, axis=1)  # Max probability per sample

        # Overall metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"\nOverall Performance:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")

        # Confidence analysis
        print(f"\nConfidence Analysis:")
        print(f"  Mean confidence: {y_conf.mean():.4f}")
        print(f"  Std confidence:  {y_conf.std():.4f}")
        print(f"  Min confidence:  {y_conf.min():.4f}")
        print(f"  Max confidence:  {y_conf.max():.4f}")

        # High-confidence predictions
        for threshold in [0.7, 0.8, 0.9, 0.95]:
            high_conf_mask = y_conf >= threshold
            if high_conf_mask.sum() > 0:
                high_conf_acc = accuracy_score(
                    y_test[high_conf_mask],
                    y_pred[high_conf_mask]
                )
                pct = high_conf_mask.mean() * 100
                print(f"  Conf ≥ {threshold}: {pct:5.1f}% of samples, Acc = {high_conf_acc:.4f}")

        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        bin_data = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_conf > bin_lower) & (y_conf <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = (y_pred[in_bin] == y_test[in_bin]).mean()
                avg_confidence_in_bin = y_conf[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                bin_data.append({
                    'bin': f'[{bin_lower:.1f}, {bin_upper:.1f}]',
                    'confidence': avg_confidence_in_bin,
                    'accuracy': accuracy_in_bin,
                    'count': in_bin.sum(),
                    'gap': abs(avg_confidence_in_bin - accuracy_in_bin)
                })

        print(f"\n✓ Expected Calibration Error (ECE): {ece:.4f}")

        if ece < 0.1:
            print("  ✅ Well calibrated (ECE < 0.1)")
        elif ece < 0.2:
            print("  ⚠️  Moderately calibrated (0.1 ≤ ECE < 0.2)")
        else:
            print("  ⚠️  Poorly calibrated (ECE ≥ 0.2)")

        results = {
            'accuracy': acc,
            'f1': f1,
            'mean_confidence': float(y_conf.mean()),
            'ece': ece,
            'bin_data': bin_data,
            'predictions': y_pred,
            'probabilities': y_proba,
            'confidences': y_conf
        }

        self.results['calibration'][model_name] = results
        return results

    # =========================================================================
    # E. CROSS-DATASET GENERALIZATION (Lightweight)
    # =========================================================================

    def cross_dataset_generalization(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
        dataset_a_name: str = "Dataset A",
        dataset_b_name: str = "Dataset B"
    ) -> Dict:
        """
        Train on one dataset, test on another (same modality, same labels)

        Expected: Performance may drop, but drop shows honesty
        """
        print("\n" + "="*80)
        print("E. CROSS-DATASET GENERALIZATION")
        print("="*80)

        X_train, y_train = train_data
        X_test, y_test = test_data

        print(f"\nTrain on: {dataset_a_name} ({len(y_train)} samples)")
        print(f"Test on:  {dataset_b_name} ({len(y_test)} samples)")

        # Train model
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)

        # Within-dataset performance (sanity check)
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        model_val = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        model_val.fit(X_train_split, y_train_split)
        within_f1 = f1_score(y_val_split, model_val.predict(X_val_split), average='macro')

        # Cross-dataset performance
        y_pred = model.predict(X_test)
        cross_f1 = f1_score(y_test, y_pred, average='macro')

        print(f"\n✓ Within-dataset F1:      {within_f1:.4f}")
        print(f"✓ Cross-dataset F1:       {cross_f1:.4f}")
        print(f"✓ Generalization gap:     {(within_f1 - cross_f1)*100:.2f}%")

        if cross_f1 > 0.5:  # Better than random (for 3 classes)
            print("\n✅ Model generalizes to new dataset")
        else:
            print("\n⚠️  Poor generalization (F1 ≤ 0.5)")

        results = {
            'within_dataset_f1': within_f1,
            'cross_dataset_f1': cross_f1,
            'generalization_gap': within_f1 - cross_f1,
            'dataset_a': dataset_a_name,
            'dataset_b': dataset_b_name
        }

        self.results['cross_dataset'] = results
        return results

    # =========================================================================
    # VISUALIZATION & REPORTING
    # =========================================================================

    def generate_report(self, save_path: str = None):
        """Generate comprehensive validation report"""

        if save_path is None:
            save_path = self.results_dir / 'validation_report.txt'

        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RIGOROUS MULTIMODAL FUSION VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")

            # A. Shuffling Control
            if 'shuffling_control' in self.results and self.results['shuffling_control']:
                f.write("A. ID-BASED SHUFFLING CONTROL\n")
                f.write("-"*80 + "\n")
                r = self.results['shuffling_control']
                f.write(f"Original (aligned) F1:     {r['original_f1']:.4f}\n")
                f.write(f"Shuffled (misaligned) F1:  {r['shuffled_f1_mean']:.4f} ± {r['shuffled_f1_std']:.4f}\n")
                f.write(f"Performance drop:          {r['performance_drop']*100:.2f}%\n")
                f.write(f"Status: {'✅ PASS' if r['pass'] else '⚠️ FAIL'}\n\n")

            # B. Pairwise Fusion
            if 'pairwise_fusion' in self.results and self.results['pairwise_fusion']:
                f.write("B. PAIRWISE FUSION LADDER\n")
                f.write("-"*80 + "\n")
                r = self.results['pairwise_fusion']
                for config, data in r.items():
                    f.write(f"{config:20s}: F1 = {data['f1']:.4f} ({data['dims']}D)\n")
                f.write("\n")

            # C. Noise Control
            if 'noise_control' in self.results and self.results['noise_control']:
                f.write("C. NOISE-INJECTION CONTROL\n")
                f.write("-"*80 + "\n")
                r = self.results['noise_control']
                f.write(f"Original F1:        {r['original_f1']:.4f}\n")
                for noise_type in ['gaussian_noise', 'random_embeddings', 'constant_vectors']:
                    if noise_type in r:
                        f.write(f"{noise_type:20s}: F1 = {r[noise_type]['f1']:.4f} "
                               f"(drop: {r[noise_type]['drop']*100:.2f}%)\n")
                f.write(f"Status: {'✅ PASS' if r['pass'] else '⚠️ FAIL'}\n\n")

            # D. Calibration
            if 'calibration' in self.results:
                f.write("D. CONFIDENCE CALIBRATION\n")
                f.write("-"*80 + "\n")
                for model_name, r in self.results['calibration'].items():
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  Accuracy:         {r['accuracy']:.4f}\n")
                    f.write(f"  F1 Score:         {r['f1']:.4f}\n")
                    f.write(f"  Mean Confidence:  {r['mean_confidence']:.4f}\n")
                    f.write(f"  ECE:              {r['ece']:.4f}\n")

            # E. Cross-dataset
            if 'cross_dataset' in self.results and self.results['cross_dataset']:
                f.write("\nE. CROSS-DATASET GENERALIZATION\n")
                f.write("-"*80 + "\n")
                r = self.results['cross_dataset']
                f.write(f"Within-dataset F1:    {r['within_dataset_f1']:.4f}\n")
                f.write(f"Cross-dataset F1:     {r['cross_dataset_f1']:.4f}\n")
                f.write(f"Generalization gap:   {r['generalization_gap']*100:.2f}%\n")

        print(f"\n✓ Report saved to {save_path}")
        return save_path


def main():
    parser = argparse.ArgumentParser(description='Rigorous multimodal fusion validation')
    parser.add_argument('--models_dir', type=str, required=True,
                       help='Directory containing saved models')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to aligned data (NPZ file)')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Directory to save results')

    args = parser.parse_args()

    # Initialize validator
    validator = MultimodalValidation(args.models_dir, args.results_dir)

    # Load data
    print("Loading data...")
    data = np.load(args.data_path)
    text_aligned = data['text_aligned']
    physio_aligned = data['physio_aligned']
    labels_aligned = data['labels_aligned']

    print(f"✓ Loaded data: {len(labels_aligned)} samples")
    print(f"  Text: {text_aligned.shape}")
    print(f"  Physio: {physio_aligned.shape}")

    # Run all validations
    print("\n" + "="*80)
    print("RUNNING RIGOROUS VALIDATION FRAMEWORK")
    print("="*80)

    # A. Shuffling control (CRITICAL)
    validator.id_shuffling_control(text_aligned, physio_aligned, labels_aligned)

    # B. Pairwise fusion ladder
    validator.pairwise_fusion_ladder(text_aligned, physio_aligned, labels_aligned)

    # C. Noise injection
    validator.noise_injection_control(text_aligned, physio_aligned, labels_aligned)

    # D. Calibration (for fusion model)
    if 'fusion' in validator.models:
        fusion_features = np.hstack([text_aligned, physio_aligned])
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            fusion_features, labels_aligned, test_size=0.2,
            stratify=labels_aligned, random_state=42
        )
        validator.confidence_calibration(validator.models['fusion'], X_test, y_test, "Fusion Model")

    # Generate report
    validator.generate_report()

    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
