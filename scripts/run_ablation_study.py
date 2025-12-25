"""
Ablation Study Runner - Trains models on all 8 non-empty modality combinations

Automatically detects available datasets and trains flexible models for:
- 4 unimodal baselines (T, A, V, P)
- 3 bimodal combinations (TA, TV, AV)
- 1 trimodal fusion (TAV)

Usage:
    python scripts/run_ablation_study.py --data_dir all_modalities --output_dir ablation_results
"""

import os
import sys
import argparse
import pickle
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import Config
from models.flexible_multimodal_model import FlexibleMultimodalModel, SimpleFlexibleModel
from training.trainer import ImprovedTrainer
from training.evaluator import ImprovedEvaluator
from data.ablation_dataset import AblationDataset
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AblationStudy:
    """Manages complete ablation study across all modality combinations"""

    # Expected dataset configurations (matches AblationDatasetGenerator)
    # UPDATED: Audio=74D (COVAREP), Video=35D (OpenFace), Text=768D (BERT), Physio=64D (WESAD)
    DATASET_CONFIGS = {
        'unimodal_T': {
            'name': 'Text Only',
            'text_dim': 768, 'audio_dim': 0, 'video_dim': 0, 'physio_dim': 0,
            'description': 'Text baseline (BERT embeddings from Reddit + MOSEI)'
        },
        'unimodal_A': {
            'name': 'Audio Only',
            'text_dim': 0, 'audio_dim': 74, 'video_dim': 0, 'physio_dim': 0,
            'description': 'Audio baseline (COVAREP 74D acoustic features from MOSEI)'
        },
        'unimodal_V': {
            'name': 'Video Only',
            'text_dim': 0, 'audio_dim': 0, 'video_dim': 35, 'physio_dim': 0,
            'description': 'Video baseline (OpenFace 35D facial features from MOSEI)'
        },
        'unimodal_P': {
            'name': 'Physio Only',
            'text_dim': 0, 'audio_dim': 0, 'video_dim': 0, 'physio_dim': 64,
            'description': 'Physiological baseline (ECG/EDA from WESAD)'
        },
        'bimodal_TA': {
            'name': 'Text + Audio',
            'text_dim': 768, 'audio_dim': 74, 'video_dim': 0, 'physio_dim': 0,
            'description': 'Text-Audio fusion (BERT + COVAREP from MOSEI)'
        },
        'bimodal_TV': {
            'name': 'Text + Video',
            'text_dim': 768, 'audio_dim': 0, 'video_dim': 35, 'physio_dim': 0,
            'description': 'Text-Video fusion (BERT + OpenFace from MOSEI)'
        },
        'bimodal_AV': {
            'name': 'Audio + Video',
            'text_dim': 0, 'audio_dim': 74, 'video_dim': 35, 'physio_dim': 0,
            'description': 'Audio-Video fusion (COVAREP + OpenFace from MOSEI)'
        },
        'trimodal_TAV': {
            'name': 'Text + Audio + Video',
            'text_dim': 768, 'audio_dim': 74, 'video_dim': 35, 'physio_dim': 0,
            'description': 'Full multimodal fusion (BERT + COVAREP + OpenFace from MOSEI)'
        }
    }

    def __init__(self, data_dir: str, output_dir: str, use_simple_model: bool = False,
                 epochs: int = 30, batch_size: int = 32, device: str = 'cuda'):
        """
        Initialize ablation study

        Args:
            data_dir: Directory containing ablation datasets (*.pkl files)
            output_dir: Directory to save results
            use_simple_model: If True, use SimpleFlexibleModel instead of full model
            epochs: Training epochs per dataset
            batch_size: Batch size
            device: 'cuda' or 'cpu'
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.use_simple_model = use_simple_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results tracking
        self.results = {}
        self.available_datasets = self._detect_available_datasets()

        logger.info(f"Ablation Study Initialized")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Available datasets: {len(self.available_datasets)}")
        logger.info(f"  Model type: {'Simple' if use_simple_model else 'Full'}")
        logger.info(f"  Device: {self.device}")

    def _detect_available_datasets(self) -> List[str]:
        """Detect which ablation datasets are available and non-empty, and auto-detect dimensions"""
        available = []

        for dataset_name in self.DATASET_CONFIGS.keys():
            filepath = self.data_dir / f"{dataset_name}.pkl"

            if not filepath.exists():
                logger.warning(f"Dataset not found: {dataset_name}.pkl")
                continue

            # Check if non-empty and detect actual dimensions
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

                if len(data) == 0:
                    logger.warning(f"Dataset is empty: {dataset_name}.pkl")
                    continue

                # Auto-detect actual dimensions from first sample
                sample = data[0]
                actual_dims = {
                    'text_dim': sample['text'].shape[0] if torch.norm(sample['text']) > 0 else 0,
                    'audio_dim': sample['audio'].shape[0] if torch.norm(sample['audio']) > 0 else 0,
                    'video_dim': sample['video'].shape[0] if torch.norm(sample['video']) > 0 else 0,
                    'physio_dim': sample['physio'].shape[0] if torch.norm(sample['physio']) > 0 else 0,
                }

                # Update config with actual dimensions
                self.DATASET_CONFIGS[dataset_name].update(actual_dims)

                dims_str = ", ".join([f"{k.replace('_dim', '')}={v}D" for k, v in actual_dims.items() if v > 0])
                available.append(dataset_name)
                logger.info(f"✓ Found {dataset_name}.pkl ({len(data):,} samples, {dims_str})")

            except Exception as e:
                logger.error(f"Error loading {dataset_name}.pkl: {e}")
                continue

        return available

    def _create_config_for_dataset(self, dataset_name: str) -> Config:
        """Create Config object with appropriate dimensions for this dataset"""

        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        cfg = self.DATASET_CONFIGS[dataset_name]

        # Create config with modality dimensions
        config = Config()
        config.text_dim = cfg['text_dim']
        config.audio_dim = cfg['audio_dim']
        config.video_dim = cfg['video_dim']
        config.physio_dim = cfg['physio_dim']

        # Training hyperparameters
        config.num_classes = 3
        config.hidden_dim = 256
        config.dropout_rate = 0.4
        config.num_transformer_layers = 2
        config.num_heads = 8
        config.sequence_length = 10

        config.batch_size = self.batch_size
        config.num_epochs = self.epochs
        config.learning_rate = 1e-4
        config.weight_decay = 1e-5

        # Device
        config.device = self.device

        return config

    def _load_dataset(self, dataset_name: str) -> Tuple[List, List, List]:
        """Load dataset and split into train/val/test with proper shuffling"""

        filepath = self.data_dir / f"{dataset_name}.pkl"

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        logger.info(f"Loaded {dataset_name}: {len(data):,} samples")

        # IMPORTANT: Shuffle data before splitting to prevent class segregation
        # Set seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(len(data))
        data_shuffled = [data[i] for i in indices]

        # Split: 70% train, 15% val, 15% test
        n = len(data_shuffled)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        train_data = data_shuffled[:train_end]
        val_data = data_shuffled[train_end:val_end]
        test_data = data_shuffled[val_end:]

        logger.info(f"  Train: {len(train_data):,}")
        logger.info(f"  Val: {len(val_data):,}")
        logger.info(f"  Test: {len(test_data):,}")

        return train_data, val_data, test_data

    def _create_dataloaders(self, train_data: List, val_data: List,
                           test_data: List, config: Config, use_weighted_sampler: bool = False,
                           sample_weights: List = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch DataLoaders with optional weighted sampling"""

        train_dataset = AblationDataset(train_data, config, mode='train')
        val_dataset = AblationDataset(val_data, config, mode='val')
        test_dataset = AblationDataset(test_data, config, mode='test')

        # Use WeightedRandomSampler for class balancing if requested
        if use_weighted_sampler and sample_weights is not None:
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True  # Allow oversampling
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                sampler=sampler,  # Use sampler instead of shuffle
                num_workers=4,
                pin_memory=True if self.device == 'cuda' else False
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True if self.device == 'cuda' else False
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )

        return train_loader, val_loader, test_loader

    def train_single_dataset(self, dataset_name: str) -> Dict:
        """Train model on a single ablation dataset"""

        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING: {dataset_name}")
        logger.info(f"{'='*80}")

        # Create config
        config = self._create_config_for_dataset(dataset_name)

        # Load data
        train_data, val_data, test_data = self._load_dataset(dataset_name)

        # Compute class weights and adjust hyperparameters BEFORE creating model
        from collections import Counter
        train_labels = [item['label'] for item in train_data]
        label_counts = Counter(train_labels)
        total_samples = len(train_labels)

        logger.info(f"Class distribution in train set:")
        for i, count in sorted(label_counts.items()):
            logger.info(f"  Class {i} ({config.class_names[i] if hasattr(config, 'class_names') else i}): {count} ({100*count/total_samples:.1f}%)")

        # Adjust hyperparameters for small MOSEI datasets BEFORE model creation
        if total_samples < 5000:  # MOSEI has ~2300 train samples
            logger.info(f"\n⚠️  Small dataset detected ({total_samples} samples)")
            logger.info("Adjusting hyperparameters for better learning:")

            # Increase learning rate for small datasets
            original_lr = config.learning_rate
            config.learning_rate = 1e-4  # Higher LR for small datasets
            logger.info(f"  • Learning rate: {original_lr} → {config.learning_rate}")

            # Reduce regularization
            original_wd = config.weight_decay
            config.weight_decay = 1e-4  # Less regularization
            logger.info(f"  • Weight decay: {original_wd} → {config.weight_decay}")

            # Reduce dropout for weak signals
            original_dropout = config.dropout_rate
            config.dropout_rate = 0.3  # Lower dropout
            logger.info(f"  • Dropout rate: {original_dropout} → {config.dropout_rate}")

            # Reduce patience
            original_patience = config.patience
            config.patience = 10  # Stop earlier if not improving
            logger.info(f"  • Patience: {original_patience} → {config.patience}")

        # Compute class weights using Effective Number of Samples
        # More stable than inverse frequency for small imbalanced datasets
        beta = 0.99 if total_samples < 5000 else 0.9999  # Lower beta for small datasets

        effective_num = []
        for i in range(config.num_classes):
            n = label_counts.get(i, 0)
            if n == 0:
                effective_num.append(1.0)
            else:
                # Effective number: (1 - beta^n) / (1 - beta)
                eff_n = (1.0 - beta ** n) / (1.0 - beta)
                effective_num.append(eff_n)

        effective_num = torch.tensor(effective_num, dtype=torch.float32)

        # Compute class weights as inverse of effective number
        class_weights = 1.0 / effective_num
        class_weights = class_weights / class_weights.sum() * config.num_classes  # Normalize

        logger.info(f"Effective numbers: {effective_num.numpy()}")
        logger.info(f"Class weights (effective num): {class_weights.numpy()}")

        # For small imbalanced datasets, use weighted sampling to oversample minority classes
        use_weighted_sampler = False
        sample_weights = None

        if total_samples < 5000:  # Small datasets (MOSEI)
            # Compute imbalance ratio
            max_count = max(label_counts.values())
            min_count = min(label_counts.values())
            imbalance_ratio = max_count / min_count

            if imbalance_ratio > 2.0:  # Significant imbalance
                logger.info(f"\n📊 Severe class imbalance detected (ratio: {imbalance_ratio:.2f})")
                logger.info("Enabling weighted sampling to balance training batches")

                # Create per-sample weights based on class
                # Use square root to reduce aggressiveness (prevents collapse to single class)
                class_weight_dict = {}
                for cls, count in label_counts.items():
                    # Square root dampens the oversampling effect
                    class_weight_dict[cls] = np.sqrt(total_samples / (len(label_counts) * count))

                sample_weights = [class_weight_dict[item['label']] for item in train_data]
                use_weighted_sampler = True

        # Create dataloaders
        train_loader, val_loader, test_loader = self._create_dataloaders(
            train_data, val_data, test_data, config,
            use_weighted_sampler=use_weighted_sampler,
            sample_weights=sample_weights
        )

        # Create model AFTER adjusting config
        if self.use_simple_model:
            model = SimpleFlexibleModel(config)
        else:
            model = FlexibleMultimodalModel(config)

        model = model.to(self.device)

        # Create output directory for this dataset
        dataset_output_dir = self.output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # Update config with output directory
        config.model_save_path = str(dataset_output_dir / 'best_model.pt')

        # Train
        trainer = ImprovedTrainer(model=model, config=config)
        trainer.train(train_loader=train_loader, val_loader=val_loader, class_weights=class_weights)

        # Evaluate on test set
        evaluator = ImprovedEvaluator(model, config)
        class_names = ['Depression', 'Euthymia', 'Mania']  # FIXED: 0=Depression, 1=Euthymia, 2=Mania
        test_metrics = evaluator.evaluate(test_loader, class_names=class_names)

        # Convert numpy arrays to lists for JSON serialization
        test_metrics_serializable = {}
        for key, value in test_metrics.items():
            if isinstance(value, np.ndarray):
                test_metrics_serializable[key] = value.tolist()
            elif key == 'sequence_analysis' or key == 'classification_report':
                # Keep these as-is (they're already dict/serializable)
                test_metrics_serializable[key] = value
            else:
                test_metrics_serializable[key] = value

        # Save results
        result = {
            'dataset': dataset_name,
            'config': self.DATASET_CONFIGS[dataset_name],
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'best_val_f1': float(trainer.best_val_f1),
            'test_metrics': test_metrics_serializable,
            'training_history': {k: [float(x) for x in v] for k, v in trainer.history.items()}
        }

        # Save to JSON
        with open(dataset_output_dir / 'results.json', 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info(f"COMPLETED: {dataset_name}")
        logger.info(f"  Best Val F1: {trainer.best_val_f1:.4f}")
        logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        logger.info(f"{'='*80}\n")

        return result

    def run_all(self):
        """Run complete ablation study"""

        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING ABLATION STUDY")
        logger.info(f"Training {len(self.available_datasets)} datasets")
        logger.info(f"{'='*80}\n")

        start_time = datetime.now()

        for i, dataset_name in enumerate(self.available_datasets, 1):
            logger.info(f"\n[{i}/{len(self.available_datasets)}] Processing {dataset_name}...")

            try:
                result = self.train_single_dataset(dataset_name)
                self.results[dataset_name] = result

            except Exception as e:
                logger.error(f"Error training {dataset_name}: {e}", exc_info=True)
                self.results[dataset_name] = {'error': str(e)}

        end_time = datetime.now()
        duration = end_time - start_time

        # Generate summary report
        self._generate_summary_report(duration)

        logger.info(f"\n{'='*80}")
        logger.info(f"ABLATION STUDY COMPLETE")
        logger.info(f"Total time: {duration}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*80}\n")

    def _generate_summary_report(self, duration):
        """Generate comprehensive summary report"""

        logger.info(f"\n{'='*80}")
        logger.info(f"GENERATING SUMMARY REPORT")
        logger.info(f"{'='*80}")

        # Create summary table
        summary_data = []

        for dataset_name in self.available_datasets:
            if dataset_name not in self.results or 'error' in self.results[dataset_name]:
                continue

            result = self.results[dataset_name]
            summary_data.append({
                'Dataset': dataset_name,
                'Description': self.DATASET_CONFIGS[dataset_name]['name'],
                'Train Samples': result['train_samples'],
                'Test Samples': result['test_samples'],
                'Best Val F1': result['best_val_f1'],
                'Test Accuracy': result['test_metrics']['accuracy'],
                'Test F1': result['test_metrics']['f1_weighted'],
                'Test F1 (Depression)': result['test_metrics']['f1_per_class'][0],
                'Test F1 (Mania)': result['test_metrics']['f1_per_class'][1],
                'Test F1 (Euthymia)': result['test_metrics']['f1_per_class'][2]
            })

        df = pd.DataFrame(summary_data)

        # Check if we have any results
        if df.empty:
            logger.warning("No results to summarize - all datasets failed or no datasets were trained")
            return

        # Sort by modality type
        order = ['unimodal', 'bimodal', 'trimodal']
        df['type'] = df['Dataset'].apply(lambda x: x.split('_')[0])
        df['type'] = pd.Categorical(df['type'], categories=order, ordered=True)
        df = df.sort_values('type').drop('type', axis=1)

        # Save to CSV
        csv_path = self.output_dir / 'ablation_summary.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary CSV: {csv_path}")

        # Print summary table
        print("\n" + "="*120)
        print("ABLATION STUDY RESULTS")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120)

        # Save metadata
        metadata = {
            'study_date': datetime.now().isoformat(),
            'duration_seconds': duration.total_seconds(),
            'num_datasets': len(self.available_datasets),
            'model_type': 'SimpleFlexibleModel' if self.use_simple_model else 'FlexibleMultimodalModel',
            'epochs_per_dataset': self.epochs,
            'batch_size': self.batch_size,
            'device': self.device
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Run complete ablation study')

    parser.add_argument('--data_dir', type=str, default='all_modalities',
                       help='Directory containing ablation datasets')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                       help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs per dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--simple', action='store_true',
                       help='Use SimpleFlexibleModel instead of full model')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Train only specific dataset (e.g., unimodal_T)')

    args = parser.parse_args()

    # Create ablation study
    study = AblationStudy(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_simple_model=args.simple,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )

    # Run single dataset or all
    if args.dataset:
        if args.dataset not in study.available_datasets:
            logger.error(f"Dataset not available: {args.dataset}")
            logger.info(f"Available datasets: {study.available_datasets}")
            return

        result = study.train_single_dataset(args.dataset)
        logger.info(f"\nFinal result: {result}")
    else:
        study.run_all()


if __name__ == '__main__':
    main()
