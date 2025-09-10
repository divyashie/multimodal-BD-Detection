"""
Enhanced Trainer with Improved Monitoring and Loss Functions
Clean implementation without boolean tensor issues
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from typing import Dict, List
from torch.utils.data import DataLoader
import os

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            # When alpha is None, no weighting applied
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ImprovedTrainer:
    """Enhanced trainer with better monitoring and stability"""

    def __init__(self, model: nn.Module, config):
        self.model = model
        self.config = config
        self.device = getattr(config, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self.model.to(self.device)
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': [],
            'train_acc': [], 'val_acc': []
        }
        
        self.best_val_f1 = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        
        self.best_model_path = getattr(config, 'model_save_path', 'best_model_improved.pth')
        os.makedirs(os.path.dirname(self.best_model_path) if os.path.dirname(self.best_model_path) else '.', exist_ok=True)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, class_weights: torch.Tensor = None) -> Dict[str, List[float]]:
        logger.info("Starting enhanced training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=getattr(self.config, 'learning_rate', 1e-4),
            weight_decay=getattr(self.config, 'weight_decay', 1e-4)
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

        if class_weights is not None:
            imbalance_ratio = class_weights.max() / class_weights.min()
            if imbalance_ratio > 5.0:
                criterion = FocalLoss(alpha=None, gamma=2.0)  # focal loss without alpha weighting
                logger.info("Using unweighted Focal Loss due to balanced sampling")
            else:
                criterion = nn.CrossEntropyLoss()  # standard loss without class weights
                logger.info("Using standard CrossEntropyLoss without class weights")
        else:
            criterion = nn.CrossEntropyLoss()
            logger.info("Using standard CrossEntropyLoss without class weights")

        num_epochs = getattr(self.config, 'num_epochs', 50)
        patience = getattr(self.config, 'patience', 15)
        min_delta = getattr(self.config, 'min_delta', 0.001)

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

            train_metrics = self._train_epoch(train_loader, optimizer, criterion)
            val_metrics = self._validate_epoch(val_loader, criterion)

            scheduler.step(val_metrics['f1'])

            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            self._update_history(train_metrics, val_metrics)

            if val_metrics['f1'] > self.best_val_f1 + min_delta:
                self.best_val_f1 = val_metrics['f1']
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                self._save_best_model()
                logger.info(f"New best model! Val F1: {self.best_val_f1:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"No improvement. Patience: {self.patience_counter}/{patience}")

            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with Val F1: {self.best_val_f1:.4f}")

        return self.history

    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        batch_size = getattr(self.config, 'batch_size', 16)
        accumulation_steps = max(1, 32 // batch_size)
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            try:
                text = batch['text'].to(self.device, non_blocking=True)
                audio = batch['audio'].to(self.device, non_blocking=True)
                video = batch['video'].to(self.device, non_blocking=True)
                physio = batch['physio'].to(self.device, non_blocking=True)  # Physio data
                labels = batch['sequence_label'].to(self.device, non_blocking=True)

                optimizer.zero_grad()
                outputs = self.model(text, audio, video, physio)
                loss = criterion(outputs, labels)

                loss = loss / accumulation_steps
                loss.backward()

                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.config, 'gradient_clip_norm', 0.5))

                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps
                with torch.no_grad():
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                if batch_idx % 10 == 0:
                    logger.info(f'  Batch {batch_idx}/{num_batches}, Loss: {loss.item() * accumulation_steps:.4f}')

            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue

        avg_loss = total_loss / max(len(all_labels), 1)
        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) if all_labels else 0.0

        return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1}

    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    text = batch['text'].to(self.device, non_blocking=True)
                    audio = batch['audio'].to(self.device, non_blocking=True)
                    video = batch['video'].to(self.device, non_blocking=True)
                    physio = batch['physio'].to(self.device, non_blocking=True)  # Physio data
                    labels = batch['sequence_label'].to(self.device, non_blocking=True)

                    outputs = self.model(text, audio, video, physio)
                    loss = criterion(outputs, labels)

                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue

        if all_labels:
            avg_loss = total_loss / max(len(val_loader), 1)
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        else:
            avg_loss = float('inf')
            accuracy = f1 = precision = recall = 0.0

        return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        logger.info(
            f"  Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, Acc: {train_metrics['accuracy']:.4f}"
        )
        logger.info(
            f"  Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}"
        )

    def _update_history(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_f1'].append(train_metrics['f1'])
        self.history['train_acc'].append(train_metrics['accuracy'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_f1'].append(val_metrics['f1'])
        self.history['val_acc'].append(val_metrics['accuracy'])

    def _save_best_model(self):
        try:
            # Save config as simple dict to avoid unserializable object issue
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('__')},
                'history': self.history,
                'best_val_f1': self.best_val_f1,
                'model_class': self.model.__class__.__name__
            }, self.best_model_path)
            logger.info(f"Best model saved to {self.best_model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_best_model(self) -> bool:
        try:
            if os.path.exists(self.best_model_path):
                checkpoint = torch.load(self.best_model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.history = checkpoint.get('history', self.history)
                self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)

                # Reconstruct config from dict (if you want to update any runtime config here)
                if 'config' in checkpoint:
                    loaded_config_dict = checkpoint['config']
                    # For example: update self.config values or re-create Config object if needed
                    # self.config = Config.from_dict(loaded_config_dict)  # If you implemented from_dict method

                logger.info("Best model loaded successfully")
                return True
            else:
                logger.warning(f"Best model file not found: {self.best_model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading best model: {e}")
            return False

    def plot_training_history(self):
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
