# ==============================================================================
# --- IMPROVED EVALUATION WITH COMPREHENSIVE METRICS ---
# ==============================================================================
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from collections import Counter
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Any
from torch import nn
import configs.config as Config
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class ImprovedEvaluator:
    """Improved evaluation with comprehensive metrics."""

    def __init__(self, model: nn.Module, config: Config):
        """Initialize evaluator with model and configuration."""
        self.model = model
        self.config = config

    def evaluate(self, test_loader: DataLoader, class_names: List[str] = None,
                 baseline_preds: List[int] = None) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        if class_names is None:
            class_names = ['Depression', 'Mania', 'Euthymia']

        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        sequence_info = []

        with torch.no_grad():
            for batch in test_loader:
                text = batch['text'].to(self.config.device)
                audio = batch['audio'].to(self.config.device)
                video = batch['video'].to(self.config.device)
                labels = batch['sequence_label'].to(self.config.device)

                outputs = self.model(text, audio, video)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # Store sequence information for analysis
                sequence_info.extend([{
                    'user_id': batch['user_id'][i],
                    'sequence_type': batch['sequence_type'][i],
                    'predicted': predicted[i].item(),
                    'actual': labels[i].item(),
                    'confidence': probs[i].max().item()
                } for i in range(len(predicted))])

        # Calculate comprehensive metrics
        results = self._calculate_comprehensive_metrics(
            all_labels, all_preds, all_probs, class_names
        )

        # Add sequence-level analysis
        results['sequence_analysis'] = self._analyze_sequence_performance(sequence_info)

        # Print results
        self._print_evaluation_results(results, class_names)

        # Plot results
        self._plot_evaluation_results(all_labels, all_preds, all_probs, class_names)

        # Add baseline comparison if provided
        if baseline_preds is not None:
            from scipy.stats import mcnemar
            p_value = mcnemar(all_labels, all_preds, baseline_preds).pvalue
            logger.info(f"McNemar's test p-value vs. baseline: {p_value:.4f}")

        return results

    def _calculate_comprehensive_metrics(self, y_true: List[int], y_pred: List[int],
                                       y_probs: List[List[float]],
                                       class_names: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)

       # Determine which classes are present
        unique_labels = sorted(list(set(y_true)))
        present_class_names = [class_names[i] for i in unique_labels]

        # Per-class metrics
        f1_per_class = f1_score(y_true, y_pred, labels=unique_labels, average=None, zero_division=0)
        precision_per_class = precision_score(y_true, y_pred, labels=unique_labels, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, labels=unique_labels, average=None, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Classification report only for present classes
        report = classification_report(y_true, y_pred, labels=unique_labels,
                                    target_names=present_class_names,
                                    output_dict=True, zero_division=0)

        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_per_class': f1_per_class,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'confusion_matrix': cm,
            'classification_report': report
        }

    def _analyze_sequence_performance(self, sequence_info: List[Dict]) -> Dict[str, Any]:
        """Analyze performance by sequence characteristics."""

        # Performance by sequence type
        type_performance = {}
        for seq_type in ['user_based', 'pseudo_temporal']:
            type_data = [s for s in sequence_info if s['sequence_type'] == seq_type]
            if type_data:
                accuracy = sum(1 for s in type_data if s['predicted'] == s['actual']) / len(type_data)
                avg_confidence = sum(s['confidence'] for s in type_data) / len(type_data)
                type_performance[seq_type] = {
                    'count': len(type_data),
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence
                }

        # Confidence distribution
        confidences = [s['confidence'] for s in sequence_info]
        confidence_stats = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        }

        # Error analysis by confidence
        high_conf_errors = [s for s in sequence_info
                           if s['confidence'] > 0.8 and s['predicted'] != s['actual']]
        low_conf_correct = [s for s in sequence_info
                           if s['confidence'] < 0.5 and s['predicted'] == s['actual']]

        return {
            'type_performance': type_performance,
            'confidence_stats': confidence_stats,
            'high_confidence_errors': len(high_conf_errors),
            'low_confidence_correct': len(low_conf_correct)
        }

    def _print_evaluation_results(self, results: Dict[str, Any], class_names: List[str]):
        """Print comprehensive evaluation results."""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE EVALUATION RESULTS")
        logger.info("="*80)

        # Overall metrics
        logger.info(f"🎯 Overall Accuracy: {results['accuracy']:.4f}")
        logger.info(f"🎯 Weighted F1-Score: {results['f1_weighted']:.4f}")
        logger.info(f"🎯 Macro F1-Score: {results['f1_macro']:.4f}")
        logger.info(f"🎯 Weighted Precision: {results['precision_weighted']:.4f}")
        logger.info(f"🎯 Weighted Recall: {results['recall_weighted']:.4f}")

        # Per-class metrics
        logger.info("\n📊 Per-Class Performance:")
        for i, class_name in enumerate(class_names):
            if i < len(results['f1_per_class']):
                logger.info(f"  {class_name}:")
                logger.info(f"    - F1-Score: {results['f1_per_class'][i]:.3f}")
                logger.info(f"    - Precision: {results['precision_per_class'][i]:.3f}")
                logger.info(f"    - Recall (Sensitivity): {results['recall_per_class'][i]:.3f}")

        # Confusion Matrix
        logger.info(f"\n📊 Confusion Matrix:")
        logger.info(f"{results['confusion_matrix']}")

        # Sequence analysis
        if 'sequence_analysis' in results:
            seq_analysis = results['sequence_analysis']
            logger.info("\n📊 Sequence-Level Analysis:")
            for seq_type, performance in seq_analysis['type_performance'].items():
                logger.info(f"  {seq_type}:")
                logger.info(f"    - Count: {performance['count']}")
                logger.info(f"    - Accuracy: {performance['accuracy']:.3f}")
                logger.info(f"    - Avg Confidence: {performance['avg_confidence']:.3f}")

            conf_stats = seq_analysis['confidence_stats']
            logger.info(f"\n  Confidence Statistics:")
            logger.info(f"    - Mean: {conf_stats['mean']:.3f}")
            logger.info(f"    - Std: {conf_stats['std']:.3f}")
            logger.info(f"    - Range: [{conf_stats['min']:.3f}, {conf_stats['max']:.3f}]")

            logger.info(f"\n  Error Analysis:")
            logger.info(f"    - High-confidence errors: {seq_analysis['high_confidence_errors']}")
            logger.info(f"    - Low-confidence correct predictions: {seq_analysis['low_confidence_correct']}")

    def _plot_evaluation_results(self, y_true: List[int], y_pred: List[int],
                               y_probs: List[List[float]], class_names: List[str]):
        """Plot evaluation results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')

        # Class distribution
        true_counts = Counter(y_true)
        pred_counts = Counter(y_pred)

        x = np.arange(len(class_names))
        width = 0.35

        ax2.bar(x - width/2, [true_counts.get(i, 0) for i in range(len(class_names))],
               width, label='True', alpha=0.8)
        ax2.bar(x + width/2, [pred_counts.get(i, 0) for i in range(len(class_names))],
               width, label='Predicted', alpha=0.8)
        ax2.set_title('Class Distribution: True vs Predicted')
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Count')
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names)
        ax2.legend()

        # Confidence distribution
        confidences = [max(probs) for probs in y_probs]
        ax3.hist(confidences, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_title('Prediction Confidence Distribution')
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Frequency')
        ax3.axvline(np.mean(confidences), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax3.legend()

        # Per-class F1 scores
        f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Determine classes present in the batch
        unique_labels = sorted(list(set(y_true)))
        f1_scores = f1_score(y_true, y_pred, labels=unique_labels, average=None, zero_division=0)
        plot_class_names = [class_names[i] for i in unique_labels]

        bars = ax4.bar(plot_class_names, f1_scores, alpha=0.8)
        ax4.set_title('F1-Score per Class')
        ax4.set_xlabel('Classes')
        ax4.set_ylabel('F1-Score')
        ax4.set_ylim(0, 1)

        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.close()  # Prevent display clutter in non-interactive environments