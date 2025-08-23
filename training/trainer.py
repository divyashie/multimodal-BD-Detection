# ==============================================================================
# --- IMPROVED TRAINING AND EVALUATION ---
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from typing import Dict, List
from configs.config import Config
from torch.utils.data import DataLoader
from models.loss import FocalLoss
from torch.serialization import add_safe_globals

logger = logging.getLogger(__name__)

class ImprovedTrainer:
    """Improved trainer with better monitoring and early stopping."""

    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        self.best_model_path = 'best_model_improved.pth'

        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': [],
            'train_acc': [], 'val_acc': []
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              class_weights: torch.Tensor) -> Dict[str, List[float]]:
        """Enhanced training loop with better monitoring."""

        # Setup optimizer with weight decay
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )

        # Setup learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

        # Setup loss function - try both weighted CrossEntropy and Focal Loss
        use_focal_loss = class_weights.max() / class_weights.min() > 5.0  # Use focal loss if high imbalance

        if use_focal_loss:
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
            logger.info("Using Focal Loss due to high class imbalance")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.config.device))
            logger.info("Using weighted CrossEntropy Loss")

        # Early stopping variables
        best_val_f1 = 0.0
        patience_counter = 0

        logger.info("Starting improved training...")

        for epoch in range(self.config.num_epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader, optimizer, criterion)

            # Validation phase
            val_metrics = self._validate_epoch(val_loader, criterion)

            # Update learning rate
            scheduler.step(val_metrics['f1'])

            # Log metrics
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)

            # Update history
            self._update_history(train_metrics, val_metrics)

            # Early stopping check
            if val_metrics['f1'] > best_val_f1 + self.config.min_delta:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                self._save_best_model()
                logger.info(f"✅ New best model saved! Val F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        return self.history

    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                    criterion: nn.Module) -> Dict[str, float]:
        """Training epoch with gradient accumulation."""
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        # Gradient accumulation steps
        accumulation_steps = max(1, 32 // self.config.batch_size)

        for batch_idx, batch in enumerate(train_loader):
            text = batch['text'].to(self.config.device)
            audio = batch['audio'].to(self.config.device)
            video = batch['video'].to(self.config.device)
            labels = batch['sequence_label'].to(self.config.device)

            # Forward pass
            outputs = self.model(text, audio, video)
            loss = criterion(outputs, labels)

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Metrics
            total_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1}

    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Validation epoch."""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                text = batch['text'].to(self.config.device)
                audio = batch['audio'].to(self.config.device)
                video = batch['video'].to(self.config.device)
                labels = batch['sequence_label'].to(self.config.device)

                outputs = self.model(text, audio, video)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        return {
            'loss': avg_loss, 'accuracy': accuracy, 'f1': f1,
            'precision': precision, 'recall': recall
        }

    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float],
                          val_metrics: Dict[str, float]):
        """Log epoch metrics."""
        logger.info(
            f"Epoch {epoch+1}/{self.config.num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, Train F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

    def _update_history(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Update training history."""
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_f1'].append(train_metrics['f1'])
        self.history['train_acc'].append(train_metrics['accuracy'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_f1'].append(val_metrics['f1'])
        self.history['val_acc'].append(val_metrics['accuracy'])

    def _save_best_model(self):
        """Save the best model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history
        }, self.best_model_path)

    def load_best_model(self) -> bool:
        """Load the best model."""
        try:
            add_safe_globals([Config])  # Allow the Config class
            checkpoint = torch.load(self.best_model_path, map_location=self.config.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.history = checkpoint.get('history', self.history)
            logger.info("Best model loaded successfully")
            return True
        except FileNotFoundError:
            logger.warning("Best model file not found")
            return False

    def plot_training_history(self):
        """Plot training history."""
        if not self.history['train_loss']:
            logger.warning("No training history to plot")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # F1 Score plot
        ax2.plot(epochs, self.history['train_f1'], 'b-', label='Training F1')
        ax2.plot(epochs, self.history['val_f1'], 'r-', label='Validation F1')
        ax2.set_title('Training and Validation F1 Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True)

        # Accuracy plot
        ax3.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        ax3.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        ax3.set_title('Training and Validation Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True)

        # Learning curve smoothness
        if len(epochs) > 1:
            train_loss_smooth = np.convolve(self.history['train_loss'], np.ones(min(5, len(epochs)))/min(5, len(epochs)), mode='valid')
            val_loss_smooth = np.convolve(self.history['val_loss'], np.ones(min(5, len(epochs)))/min(5, len(epochs)), mode='valid')
            smooth_epochs = range(1, len(train_loss_smooth) + 1)

            ax4.plot(smooth_epochs, train_loss_smooth, 'b-', label='Smoothed Train Loss')
            ax4.plot(smooth_epochs, val_loss_smooth, 'r-', label='Smoothed Val Loss')
            ax4.set_title('Smoothed Learning Curves')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.legend()
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor smoothing',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Smoothed Learning Curves')

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

