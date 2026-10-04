#!/usr/bin/env python3
"""
Stage 1: Pre-train text encoder on full text dataset (14K samples)

This creates a strong text encoder that will be frozen during multimodal fine-tuning.
"""

import sys
import os
import torch
import torch.nn as nn
import pickle
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from data.ablation_dataset import AblationDataset
from training.trainer import ImprovedTrainer
from training.evaluator import ImprovedEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextOnlyModel(nn.Module):
    """
    Simple text-only model for pre-training
    This will later become the frozen text encoder in multimodal models
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        text_dim = getattr(config, 'text_dim', 768)
        hidden_dim = getattr(config, 'hidden_dim', 256)
        dropout_rate = getattr(config, 'dropout_rate', 0.3)
        num_classes = getattr(config, 'num_classes', 3)

        # Text encoder (will be frozen later)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, text_features, audio_features=None, video_features=None, physio_features=None):
        """
        Forward pass - only uses text
        Other modalities are ignored (for compatibility with trainer)
        """
        # Mean pool over sequence dimension
        # text_features: [batch, seq_len, 768]
        text_pooled = torch.mean(text_features, dim=1)  # [batch, 768]

        # Encode
        encoded = self.text_encoder(text_pooled)  # [batch, hidden_dim]

        # Classify
        logits = self.classifier(encoded)  # [batch, num_classes]

        return logits


def load_and_split_data(data_path: str):
    """Load unimodal_T dataset and split into train/val/test"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    logger.info(f"Loaded {len(data):,} samples from {data_path}")

    # Shuffle with fixed seed
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

    logger.info(f"Split: {len(train_data):,} train, {len(val_data):,} val, {len(test_data):,} test")

    return train_data, val_data, test_data


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Pre-train text encoder on full text dataset')
    parser.add_argument('--data_dir', type=str, default='all_modalities',
                        help='Directory containing datasets')
    parser.add_argument('--output_dir', type=str, default='pretrained_text_encoder',
                        help='Directory to save pre-trained model')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("STAGE 1: PRE-TRAINING TEXT ENCODER")
    logger.info("="*80)
    logger.info(f"  Data: {args.data_dir}/unimodal_T.pkl")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Device: {args.device}")
    logger.info("")

    # Load data
    data_path = Path(args.data_dir) / "unimodal_T.pkl"
    train_data, val_data, test_data = load_and_split_data(str(data_path))

    # Create config
    config = Config()
    config.text_dim = 768
    config.audio_dim = 0  # Not used
    config.video_dim = 0  # Not used
    config.physio_dim = 0  # Not used
    config.hidden_dim = args.hidden_dim
    config.dropout_rate = 0.3
    config.num_classes = 3
    config.batch_size = args.batch_size
    config.max_epochs = args.epochs
    config.learning_rate = 1e-3
    config.device = args.device

    # Create datasets
    train_dataset = AblationDataset(train_data, config, mode='train')
    val_dataset = AblationDataset(val_data, config, mode='val')
    test_dataset = AblationDataset(test_data, config, mode='test')

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create model
    model = TextOnlyModel(config)
    model = model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.text_encoder.parameters())

    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Text encoder parameters: {encoder_params:,}")
    logger.info("")

    # Train
    logger.info("Starting training...")
    trainer = ImprovedTrainer(model, config)

    # Set checkpoint path
    config.checkpoint_dir = str(output_path)

    trainer.train(train_loader, val_loader)

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    evaluator = ImprovedEvaluator(model, config)
    test_metrics = evaluator.evaluate(test_loader)

    logger.info("\n" + "="*80)
    logger.info("TEST SET RESULTS")
    logger.info("="*80)
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score: {test_metrics['f1_weighted']:.4f}")
    logger.info(f"  F1 per class: {test_metrics['f1_per_class']}")
    logger.info("")

    # Save encoder weights separately for easy loading
    encoder_path = output_path / "text_encoder.pt"
    torch.save({
        'encoder_state_dict': model.text_encoder.state_dict(),
        'config': {
            'text_dim': config.text_dim,
            'hidden_dim': config.hidden_dim,
            'dropout_rate': config.dropout_rate,
        },
        'performance': {
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1_weighted'],
            'test_f1_per_class': test_metrics['f1_per_class'],
        }
    }, encoder_path)

    logger.info(f"✅ Text encoder saved to: {encoder_path}")
    logger.info("")
    logger.info("="*80)
    logger.info("STAGE 1 COMPLETE")
    logger.info("="*80)
    logger.info("Next step: Run stage 2 fine-tuning with frozen encoder")
    logger.info(f"  python scripts/finetune_multimodal.py --encoder_path {encoder_path}")
    logger.info("")


if __name__ == "__main__":
    main()
