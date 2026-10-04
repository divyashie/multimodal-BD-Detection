"""
Shuffled Label Sanity Check
Verifies models learn real patterns vs. memorizing artifacts

Trains models with:
1. Real labels (your current results)
2. Shuffled labels (should get random chance accuracy)

If shuffled accuracy ≈ random chance → model learns real patterns ✅
If shuffled accuracy >> random chance → possible data leakage ❌
"""

import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import Config
from models.flexible_multimodal_model import FlexibleMultimodalModel
from training.trainer import ImprovedTrainer
from training.evaluator import ImprovedEvaluator
from data.ablation_dataset import AblationDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class ShuffledLabelCheck:
    """Sanity check: Train with shuffled labels"""

    def __init__(self, data_dir='all_modalities', device='cpu'):
        self.data_dir = Path(data_dir)
        self.device = device
        self.results = {}

    def run_check(self, dataset_name: str, epochs: int = 10) -> Dict:
        """
        Run sanity check on a single dataset

        Trains twice:
        1. Normal labels (baseline)
        2. Shuffled labels (sanity check)

        Args:
            dataset_name: e.g., 'unimodal_T'
            epochs: Training epochs (use fewer for sanity check)

        Returns:
            dict with normal_acc, shuffled_acc, random_chance
        """
        logger.info("="*80)
        logger.info(f"SHUFFLED LABEL SANITY CHECK: {dataset_name}")
        logger.info("="*80)

        # Load data
        filepath = self.data_dir / f"{dataset_name}.pkl"
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Shuffle before splitting
        np.random.seed(42)
        indices = np.random.permutation(len(data))
        data_shuffled = [data[i] for i in indices]

        # Split
        n = len(data_shuffled)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        train_data = data_shuffled[:train_end]
        val_data = data_shuffled[train_end:val_end]
        test_data = data_shuffled[val_end:]

        logger.info(f"Dataset: {len(data):,} samples")
        logger.info(f"  Train: {len(train_data):,}")
        logger.info(f"  Val: {len(val_data):,}")
        logger.info(f"  Test: {len(test_data):,}")

        # Experiment 1: Normal labels
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT 1: NORMAL LABELS (Real Patterns)")
        logger.info("="*80)
        normal_acc, normal_f1 = self._train_and_eval(
            dataset_name, train_data, val_data, test_data,
            shuffle_labels=False, epochs=epochs
        )

        # Experiment 2: Shuffled labels
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT 2: SHUFFLED LABELS (Sanity Check)")
        logger.info("="*80)
        shuffled_acc, shuffled_f1 = self._train_and_eval(
            dataset_name, train_data, val_data, test_data,
            shuffle_labels=True, epochs=epochs
        )

        # Analysis
        logger.info("\n" + "="*80)
        logger.info("SANITY CHECK RESULTS")
        logger.info("="*80)

        # Calculate random chance
        labels = [s['label'] for s in test_data]
        num_classes = len(set(labels))
        random_chance = 1.0 / num_classes

        logger.info(f"Random Chance:       {random_chance*100:.2f}%")
        logger.info(f"Normal Accuracy:     {normal_acc*100:.2f}%")
        logger.info(f"Shuffled Accuracy:   {shuffled_acc*100:.2f}%")
        logger.info(f"Accuracy Drop:       {(normal_acc - shuffled_acc)*100:.2f}%")
        logger.info("")
        logger.info(f"Normal F1:           {normal_f1:.4f}")
        logger.info(f"Shuffled F1:         {shuffled_f1:.4f}")

        # Verdict
        if shuffled_acc < random_chance + 0.10:
            verdict = "✅ PASS: Model learns real patterns (shuffled ≈ random)"
        else:
            verdict = "❌ FAIL: Possible data leakage (shuffled >> random)"

        logger.info(f"\n{verdict}")
        logger.info("="*80)

        return {
            'dataset': dataset_name,
            'num_samples': len(data),
            'num_classes': num_classes,
            'random_chance': random_chance,
            'normal_acc': normal_acc,
            'normal_f1': normal_f1,
            'shuffled_acc': shuffled_acc,
            'shuffled_f1': shuffled_f1,
            'drop': normal_acc - shuffled_acc,
            'verdict': verdict
        }

    def _train_and_eval(self, dataset_name: str, train_data, val_data, test_data,
                        shuffle_labels: bool, epochs: int) -> tuple:
        """Train model and return test accuracy and F1"""

        # Create copies to avoid modifying original
        train_data_copy = [s.copy() for s in train_data]
        val_data_copy = [s.copy() for s in val_data]

        if shuffle_labels:
            logger.info("🔀 Shuffling training labels...")
            # Shuffle labels (breaks pattern between features and labels)
            train_labels = [s['label'] for s in train_data_copy]
            np.random.seed(999)  # Different seed than data split
            np.random.shuffle(train_labels)
            for i, sample in enumerate(train_data_copy):
                sample['label'] = train_labels[i]

            # Also shuffle val labels for consistent training
            val_labels = [s['label'] for s in val_data_copy]
            np.random.shuffle(val_labels)
            for i, sample in enumerate(val_data_copy):
                sample['label'] = val_labels[i]

        # Create config
        config = self._get_config(dataset_name)
        config.num_epochs = epochs

        # Create datasets
        train_dataset = AblationDataset(train_data_copy, config, mode='train')
        val_dataset = AblationDataset(val_data_copy, config, mode='val')
        test_dataset = AblationDataset(test_data, config, mode='test')  # NEVER shuffle test!

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Create model
        model = FlexibleMultimodalModel(config).to(self.device)

        # Train
        trainer = ImprovedTrainer(model, config)
        trainer.train(train_loader, val_loader)

        # Evaluate on test set (with REAL labels, never shuffled)
        evaluator = ImprovedEvaluator(model, config)
        test_metrics = evaluator.evaluate(test_loader, class_names=['Depression', 'Mania', 'Euthymia'])

        accuracy = test_metrics['accuracy']
        f1 = test_metrics['f1_weighted']

        logger.info(f"Final Test Accuracy: {accuracy*100:.2f}%")
        logger.info(f"Final Test F1: {f1:.4f}")

        return accuracy, f1

    def _get_config(self, dataset_name: str) -> Config:
        """Create config for dataset"""
        CONFIGS = {
            'unimodal_T': {'text_dim': 768, 'audio_dim': 0, 'video_dim': 0, 'physio_dim': 0},
            'unimodal_A': {'text_dim': 0, 'audio_dim': 88, 'video_dim': 0, 'physio_dim': 0},
            'unimodal_V': {'text_dim': 0, 'audio_dim': 0, 'video_dim': 2048, 'physio_dim': 0},
            'unimodal_P': {'text_dim': 0, 'audio_dim': 0, 'video_dim': 0, 'physio_dim': 64},
        }

        cfg = CONFIGS.get(dataset_name, CONFIGS['unimodal_T'])
        config = Config()
        config.text_dim = cfg['text_dim']
        config.audio_dim = cfg['audio_dim']
        config.video_dim = cfg['video_dim']
        config.physio_dim = cfg['physio_dim']

        config.num_classes = 3
        config.hidden_dim = 256
        config.dropout_rate = 0.4
        config.num_transformer_layers = 2
        config.num_heads = 8
        config.sequence_length = 10
        config.batch_size = 32
        config.learning_rate = 1e-4
        config.weight_decay = 1e-5
        config.device = self.device

        return config

    def run_all_unimodal(self):
        """Run sanity check on all 4 unimodal datasets"""
        datasets = ['unimodal_T', 'unimodal_A', 'unimodal_V', 'unimodal_P']

        results = {}
        for dataset in datasets:
            if (self.data_dir / f"{dataset}.pkl").exists():
                results[dataset] = self.run_check(dataset, epochs=10)
            else:
                logger.warning(f"Dataset not found: {dataset}.pkl")

        # Print summary table
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict):
        """Print summary table"""
        logger.info("\n" + "="*80)
        logger.info("SUMMARY: SHUFFLED LABEL SANITY CHECK")
        logger.info("="*80)
        logger.info(f"{'Dataset':<15} {'Normal Acc':<12} {'Shuffled Acc':<14} {'Random':<10} {'Verdict'}")
        logger.info("-"*80)

        for dataset, res in results.items():
            verdict_symbol = "✅" if "PASS" in res['verdict'] else "❌"
            logger.info(f"{dataset:<15} {res['normal_acc']*100:>6.2f}%      "
                       f"{res['shuffled_acc']*100:>6.2f}%         "
                       f"{res['random_chance']*100:>5.2f}%    {verdict_symbol}")

        logger.info("="*80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Shuffled Label Sanity Check')
    parser.add_argument('--data_dir', type=str, default='all_modalities')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset (e.g., unimodal_T) or None for all unimodal')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Training epochs (fewer for sanity check)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])

    args = parser.parse_args()

    checker = ShuffledLabelCheck(data_dir=args.data_dir, device=args.device)

    if args.dataset:
        results = checker.run_check(args.dataset, epochs=args.epochs)
    else:
        results = checker.run_all_unimodal()


if __name__ == '__main__':
    main()
