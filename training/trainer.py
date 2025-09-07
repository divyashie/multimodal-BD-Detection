"""
Enhanced Trainer with Improved Monitoring and Loss Functions
Clean implementation without boolean tensor issues
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from typing import Dict, List
from torch.utils.data import DataLoader
import os

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
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
            class_weights = class_weights.to(self.device)
            imbalance_ratio = class_weights.max() / class_weights.min()
            if imbalance_ratio > 5.0:
                criterion = FocalLoss(alpha=0.25, gamma=2.0)
                logger.info("Using Focal Loss due to high class imbalance")
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                logger.info("Using weighted CrossEntropy Loss")
        else:
            criterion = nn.CrossEntropyLoss()
            logger.info("Using standard CrossEntropy Loss")

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

                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.config, 'gradient_clip_norm', 0.5))
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
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
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
                logger.info("Best model loaded successfully")
                return True
            else:
                logger.warning(f"Best model file not found: {self.best_model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading best model: {e}")
            return False

    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, any]:
        logger.info("Evaluating model on test set...")

        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                try:
                    text = batch['text'].to(self.device, non_blocking=True)
                    audio = batch['audio'].to(self.device, non_blocking=True)
                    video = batch['video'].to(self.device, non_blocking=True)
                    physio = batch['physio'].to(self.device, non_blocking=True)
                    labels = batch['sequence_label'].to(self.device, non_blocking=True)

                    outputs = self.model(text, audio, video, physio)
                    probs = torch.softmax(outputs, dim=1)

                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                except Exception as e:
                    logger.error(f"Error in evaluation batch: {e}")
                    continue

        if not all_labels:
            logger.error("No valid predictions made during evaluation")
            return {}

        accuracy = accuracy_score(all_labels, all_preds)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

        class_names = ['Depression', 'Mania', 'Euthymia']
        class_report = classification_report(
            all_labels, all_preds, 
            target_names=class_names[:len(set(all_labels))],
            output_dict=True, 
            zero_division=0
        )

        conf_matrix = confusion_matrix(all_labels, all_preds)

        results = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }

        logger.info(f"Test Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 (weighted): {f1_weighted:.4f}")
        logger.info(f"  F1 (macro): {f1_macro:.4f}")

        for i, class_name in enumerate(class_names[:len(precision_per_class)]):
            logger.info(f"  {class_name}: P={precision_per_class[i]:.3f}, R={recall_per_class[i]:.3f}, F1={f1_per_class[i]:.3f}")

        return results

    def plot_training_history(self, save_path: str = 'training_history.png'):
        if not self.history['train_loss']:
            logger.warning("No training history to plot")
            return
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            epochs = range(1, len(self.history['train_loss']) + 1)
            axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
            axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(epochs, self.history['train_f1'], 'b-', label='Training F1', linewidth=2)
            axes[0, 1].plot(epochs, self.history['val_f1'], 'r-', label='Validation F1', linewidth=2)
            if self.best_val_f1 > 0:
                axes[0, 1].axhline(y=self.best_val_f1, color='g', linestyle='--', alpha=0.7, label=f'Best F1: {self.best_val_f1:.3f}')
            axes[0, 1].set_title('F1 Score Progress')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
            axes[1, 0].plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
            axes[1, 0].set_title('Accuracy Progress')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            if len(epochs) > 5:
                window = min(5, len(epochs) // 3)
                train_loss_smooth = np.convolve(self.history['train_loss'], np.ones(window)/window, mode='valid')
                val_loss_smooth = np.convolve(self.history['val_loss'], np.ones(window)/window, mode='valid')
                smooth_epochs = range(window, len(self.history['train_loss']) + 1)

                axes[1, 1].plot(smooth_epochs, train_loss_smooth, 'b-', label='Smoothed Train Loss', linewidth=2)
                axes[1, 1].plot(smooth_epochs, val_loss_smooth, 'r-', label='Smoothed Val Loss', linewidth=2)
                axes[1, 1].set_title('Smoothed Learning Curves')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient data for smoothing', ha='center', va='center', transform=axes[1, 1].transAxes)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating training plots: {e}")

# Compatibility class
class ImprovedEvaluator:
    """Enhanced evaluator for comprehensive model assessment"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = getattr(config, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        trainer = ImprovedTrainer(self.model, self.config)
        return trainer.evaluate_model(test_loader)
