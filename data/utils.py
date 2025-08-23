# ==============================================================================
# --- ENHANCED DATA SPLITTING ---
# ==============================================================================
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from collections import Counter
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple 

logger = logging.getLogger(__name__)

def create_improved_user_split(full_data: List[Dict], train_ratio: float = 0.7,
                              val_ratio: float = 0.15, test_ratio: float = 0.15,
                              min_samples_per_split: int = 100) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Create improved user-based data split with better balance."""

    # Analyze user distribution
    user_data = {}
    for item in full_data:
        user_id = item.get('user_id')
        if user_id is not None:
            if user_id not in user_data:
                user_data[user_id] = []
            user_data[user_id].append(item)

    # Separate users by number of posts
    multi_post_users = {uid: data for uid, data in user_data.items() if len(data) >= 3}
    single_post_users = {uid: data for uid, data in user_data.items() if len(data) < 3}

    logger.info(f"Multi-post users: {len(multi_post_users)}")
    logger.info(f"Single-post users: {len(single_post_users)}")

    # Split multi-post users first
    multi_post_user_ids = list(multi_post_users.keys())
    np.random.seed(42)
    np.random.shuffle(multi_post_user_ids)

    n_multi = len(multi_post_user_ids)
    n_train_multi = int(train_ratio * n_multi)
    n_val_multi = int(val_ratio * n_multi)

    train_multi_users = set(multi_post_user_ids[:n_train_multi])
    val_multi_users = set(multi_post_user_ids[n_train_multi:n_train_multi + n_val_multi])
    test_multi_users = set(multi_post_user_ids[n_train_multi + n_val_multi:])

    # Split single-post users
    single_post_user_ids = list(single_post_users.keys())
    np.random.shuffle(single_post_user_ids)

    n_single = len(single_post_user_ids)
    n_train_single = int(train_ratio * n_single)
    n_val_single = int(val_ratio * n_single)

    train_single_users = set(single_post_user_ids[:n_train_single])
    val_single_users = set(single_post_user_ids[n_train_single:n_train_single + n_val_single])
    test_single_users = set(single_post_user_ids[n_train_single + n_val_single:])

    # Combine and create final splits
    train_users = train_multi_users | train_single_users
    val_users = val_multi_users | val_single_users
    test_users = test_multi_users | test_single_users

    train_data = [item for item in full_data if item.get('user_id') in train_users]
    val_data = [item for item in full_data if item.get('user_id') in val_users]
    test_data = [item for item in full_data if item.get('user_id') in test_users]

    # Ensure minimum samples per split
    if len(train_data) < min_samples_per_split or len(val_data) < min_samples_per_split:
        logger.warning("Insufficient samples in splits, falling back to random split")
        return create_fallback_split(full_data, train_ratio, val_ratio)

    return train_data, val_data, test_data

def create_fallback_split(full_data: List[Dict], train_ratio: float,
                         val_ratio: float) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Fallback to stratified random split if user-based split fails."""

    # Group by labels for stratified split
    label_groups = {}
    for item in full_data:
        label = item.get('label')
        if label is not None:
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)

    train_data, val_data, test_data = [], [], []

    np.random.seed(42)
    for label, items in label_groups.items():
        np.random.shuffle(items)
        n_items = len(items)
        n_train = int(train_ratio * n_items)
        n_val = int(val_ratio * n_items)

        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])

    return train_data, val_data, test_data

# ==============================================================================
# --- ENHANCED DATA LOADER ---
# ==============================================================================
def create_improved_data_loader(dataset: Dataset, batch_size: int,
                               shuffle: bool = False,
                               use_sampler: bool = False) -> DataLoader:
    """Create improved data loader with better collation."""

    def collate_fn(batch):
        """Enhanced collate function with error handling."""
        try:
            return {
                'text': torch.stack([item['text'] for item in batch]),
                'audio': torch.stack([item['audio'] for item in batch]),
                'video': torch.stack([item['video'] for item in batch]),
                'labels': torch.stack([item['labels'] for item in batch]),
                'sequence_label': torch.stack([item['sequence_label'] for item in batch]),
                'sequence_type': [item['sequence_type'] for item in batch],
                'user_id': [item['user_id'] for item in batch]
            }
        except Exception as e:
            logger.error(f"Error in collate_fn: {e}")
            # Return a valid batch with single item to prevent crash
            if len(batch) > 0:
                return {
                    'text': batch[0]['text'].unsqueeze(0),
                    'audio': batch[0]['audio'].unsqueeze(0),
                    'video': batch[0]['video'].unsqueeze(0),
                    'labels': batch[0]['labels'].unsqueeze(0),
                    'sequence_label': batch[0]['sequence_label'].unsqueeze(0),
                    'sequence_type': [batch[0]['sequence_type']],
                    'user_id': [batch[0]['user_id']]
                }
            else:
                # Emergency fallback - should never happen
                return None

    sampler = None
    if use_sampler and hasattr(dataset, 'get_class_weights'):
        try:
            # Create weighted sampler for balanced training
            sequence_labels = []
            for i in range(len(dataset)):
                sequence_labels.append(dataset[i]['sequence_label'].item())

            class_weights = dataset.get_class_weights()
            sample_weights = [class_weights[label] for label in sequence_labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            shuffle = False  # Don't shuffle when using sampler
        except Exception as e:
            logger.warning(f"Could not create weighted sampler: {e}")
            sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )

# ==============================================================================
# --- UTILITY FUNCTIONS ---
# ==============================================================================
def analyze_data_distribution(data_list: List[Dict]) -> Dict[str, Any]:
    """Analyze the distribution of data across different dimensions."""

    analysis = {}

    # User distribution
    user_ids = [d.get('user_id') for d in data_list if d.get('user_id') is not None]
    unique_users = set(user_ids)
    user_post_counts = Counter(user_ids)

    analysis['users'] = {
        'total_unique': len(unique_users),
        'total_posts': len(user_ids),
        'avg_posts_per_user': len(user_ids) / len(unique_users) if unique_users else 0,
        'max_posts_per_user': max(user_post_counts.values()) if user_post_counts else 0,
        'min_posts_per_user': min(user_post_counts.values()) if user_post_counts else 0
    }

    # Label distribution
    labels = [d.get('label') for d in data_list if d.get('label') is not None]
    label_counts = Counter(labels)
    total_labeled = len(labels)

    analysis['labels'] = {
        'total_samples': total_labeled,
        'class_distribution': dict(label_counts),
        'class_percentages': {k: (v/total_labeled)*100 for k, v in label_counts.items()} if total_labeled > 0 else {}
    }

    # Temporal distribution
    timestamps = [d.get('timestamp') for d in data_list if d.get('timestamp') is not None]
    if timestamps:
        timestamps = np.array(timestamps)
        analysis['temporal'] = {
            'total_with_timestamps': len(timestamps),
            'timestamp_range': float(timestamps.max() - timestamps.min()) if len(timestamps) > 1 else 0.0,
            'timestamp_std': float(timestamps.std()) if len(timestamps) > 1 else 0.0
        }
    else:
        analysis['temporal'] = {
            'total_with_timestamps': 0,
            'timestamp_range': 0.0,
            'timestamp_std': 0.0
        }

    return analysis

def suggest_hyperparameters(data_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest hyperparameters based on data analysis."""

    suggestions = {}

    # Batch size based on data size
    total_samples = data_analysis.get('labels', {}).get('total_samples', 1000)
    if total_samples < 1000:
        suggestions['batch_size'] = 8
    elif total_samples < 10000:
        suggestions['batch_size'] = 16
    else:
        suggestions['batch_size'] = 32

    # Learning rate based on model complexity
    suggestions['learning_rate'] = 1e-4 if total_samples > 5000 else 5e-4

    # Sequence parameters based on user distribution
    user_stats = data_analysis.get('users', {})
    avg_posts = user_stats.get('avg_posts_per_user', 3)

    if avg_posts > 10:
        suggestions['sequence_length'] = 16
        suggestions['min_user_posts'] = 8
    elif avg_posts > 5:
        suggestions['sequence_length'] = 8
        suggestions['min_user_posts'] = 4
    else:
        suggestions['sequence_length'] = 4
        suggestions['min_user_posts'] = 2

    # Class imbalance handling
    class_dist = data_analysis.get('labels', {}).get('class_percentages', {})
    if class_dist:
        max_pct = max(class_dist.values()) if class_dist.values() else 0
        min_pct = min(class_dist.values()) if class_dist.values() else 1
        imbalance_ratio = max_pct / min_pct if min_pct > 0 else 10

        if imbalance_ratio > 5:
            suggestions['use_weighted_sampling'] = True
            suggestions['focal_loss_alpha'] = 0.25
            suggestions['focal_loss_gamma'] = 2.0

    return suggestions

def create_data_quality_dashboard(data_analysis: Dict[str, Any]) -> None:
    """Create a comprehensive data quality dashboard."""

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    # 1. Class distribution pie chart
    class_dist = data_analysis.get('labels', {}).get('class_distribution', {})
    if class_dist:
        labels = [f'Class {k}' for k in class_dist.keys()]
        values = list(class_dist.values())
        axes[0].pie(values, labels=labels, autopct='%1.1f%%')
        axes[0].set_title('Class Distribution')
    else:
        axes[0].text(0.5, 0.5, 'No class data available', ha='center', va='center')
        axes[0].set_title('Class Distribution')

    # 2. User post distribution histogram
    user_stats = data_analysis.get('users', {})
    if user_stats.get('total_unique', 0) > 0:
        # This is a simplified representation - in practice you'd want the actual distribution
        axes[1].bar(['Min Posts', 'Avg Posts', 'Max Posts'],
                   [user_stats.get('min_posts_per_user', 0),
                    user_stats.get('avg_posts_per_user', 0),
                    user_stats.get('max_posts_per_user', 0)])
        axes[1].set_title('Posts per User Statistics')
        axes[1].set_ylabel('Number of Posts')
    else:
        axes[1].text(0.5, 0.5, 'No user data available', ha='center', va='center')
        axes[1].set_title('Posts per User Statistics')

    # 3. Temporal distribution
    temporal_stats = data_analysis.get('temporal', {})
    if temporal_stats.get('total_with_timestamps', 0) > 0:
        axes[2].bar(['Range', 'Std Dev'],
                   [temporal_stats.get('timestamp_range', 0),
                    temporal_stats.get('timestamp_std', 0)])
        axes[2].set_title('Temporal Distribution Statistics')
        axes[2].set_ylabel('Value')
    else:
        axes[2].text(0.5, 0.5, 'No temporal data available', ha='center', va='center')
        axes[2].set_title('Temporal Distribution Statistics')

    # 4. Data completeness
    total_samples = data_analysis.get('labels', {}).get('total_samples', 0)
    total_users = user_stats.get('total_unique', 0)
    total_with_timestamps = temporal_stats.get('total_with_timestamps', 0)

    completeness_data = [total_samples, total_users, total_with_timestamps]
    completeness_labels = ['Total Samples', 'Users with ID', 'Samples with Timestamp']

    axes[3].bar(completeness_labels, completeness_data)
    axes[3].set_title('Data Completeness')
    axes[3].set_ylabel('Count')
    axes[3].tick_params(axis='x', rotation=45)

    # 5. Class imbalance visualization
    if class_dist:
        max_count = max(class_dist.values())
        normalized_counts = [v/max_count for v in class_dist.values()]
        class_labels = [f'Class {k}' for k in class_dist.keys()]

        axes[4].bar(class_labels, normalized_counts)
        axes[4].set_title('Class Imbalance (Normalized)')
        axes[4].set_ylabel('Relative Frequency')
        axes[4].set_ylim(0, 1)
    else:
        axes[4].text(0.5, 0.5, 'No class data available', ha='center', va='center')
        axes[4].set_title('Class Imbalance (Normalized)')

    # 6. Summary statistics
    summary_text = f"""
    Data Quality Summary:

    Total Samples: {total_samples}
    Unique Users: {total_users}
    Avg Posts/User: {user_stats.get('avg_posts_per_user', 0):.1f}

    Class Distribution:
    """

    if class_dist:
        for k, v in class_dist.items():
            pct = (v/total_samples)*100 if total_samples > 0 else 0
            summary_text += f"\n  Class {k}: {v} ({pct:.1f}%)"

    axes[5].text(0.1, 0.9, summary_text, transform=axes[5].transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=10)
    axes[5].set_title('Summary Statistics')
    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig('data_quality_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==============================================================================
# --- ENHANCED MAIN EXECUTION ---
# ==============================================================================
def main():
    """Enhanced main function with comprehensive pipeline."""

    # Validate configuration
    Config.validate()

    logger.info(f"Using device: {Config.device}")
    logger.info(f"Starting enhanced mental health classification pipeline...")

    try:
        # Load data
        logger.info(f"Loading data from {Config.data_path}")
        with open(Config.data_path, 'rb') as f:
            full_data = pickle.load(f)

        logger.info(f"Total samples loaded: {len(full_data)}")

        # Analyze data distribution
        data_analysis = analyze_data_distribution(full_data)
        logger.info("\nData Distribution Analysis:")
        logger.info(f"  Total unique users: {data_analysis['users']['total_unique']}")
        logger.info(f"  Average posts per user: {data_analysis['users']['avg_posts_per_user']:.2f}")
        logger.info(f"  Class distribution: {data_analysis['labels']['class_distribution']}")

        # Create data quality dashboard
        create_data_quality_dashboard(data_analysis)

        # Get hyperparameter suggestions
        suggestions = suggest_hyperparameters(data_analysis)
        logger.info(f"\nHyperparameter suggestions: {suggestions}")

        # Create improved data splits
        train_data, val_data, test_data = create_improved_user_split(full_data)

        logger.info("\n" + "="*60)
        logger.info("ENHANCED DATA SPLIT SUMMARY")
        logger.info("="*60)
        logger.info(f"Train samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        logger.info(f"Test samples: {len(test_data)}")

        # Create improved datasets
        train_dataset = ImprovedTemporalDataset(train_data, Config, mode='train')
        val_dataset = ImprovedTemporalDataset(val_data, Config, mode='val')
        test_dataset = ImprovedTemporalDataset(test_data, Config, mode='test')

        # Create improved data loaders
        train_loader = create_improved_data_loader(
            train_dataset, Config.batch_size, shuffle=True, use_sampler=True
        )
        val_loader = create_improved_data_loader(val_dataset, Config.batch_size)
        test_loader = create_improved_data_loader(test_dataset, Config.batch_size)

        logger.info(f"DataLoaders created: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")

        # Get class weights
        class_weights = train_dataset.get_class_weights()
        logger.info(f"Class weights: {class_weights}")

        # Initialize improved model
        model = ImprovedMultimodalModel(Config).to(Config.device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"\nModel Statistics:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")

        # Initialize trainer
        trainer = ImprovedTrainer(model, Config)

        # Start training
        logger.info("\n" + "="*80)
        logger.info("STARTING ENHANCED MODEL TRAINING")
        logger.info("="*80)

        history = trainer.train(train_loader, val_loader, class_weights)

        # Plot training history
        trainer.plot_training_history()

        # Load best model for evaluation
        trainer.load_best_model()

        # Initialize evaluator and run comprehensive evaluation
        evaluator = ImprovedEvaluator(model, Config)

        logger.info("\n" + "="*80)
        logger.info("STARTING COMPREHENSIVE EVALUATION")
        logger.info("="*80)

        evaluation_results = evaluator.evaluate(test_loader)

        # Save results
        results_summary = {
            'config': {
                'sequence_length': Config.sequence_length,
                'batch_size': Config.batch_size,
                'learning_rate': Config.learning_rate,
                'dropout_rate': Config.dropout_rate,
                'num_epochs': Config.num_epochs,
                'hidden_dim': Config.hidden_dim
            },
            'data_analysis': data_analysis,
            'training_history': history,
            'evaluation_results': evaluation_results,
            'model_stats': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            }
        }

        # Save as pickle
        with open('improved_results_summary.pkl', 'wb') as f:
            pickle.dump(results_summary, f)

        # Also save as JSON for easy reading
        import json
        json_results = {
            'config': results_summary['config'],
            'final_metrics': {
                'accuracy': float(evaluation_results['accuracy']),
                'f1_weighted': float(evaluation_results['f1_weighted']),
                'f1_macro': float(evaluation_results['f1_macro'])
            },
            'model_stats': results_summary['model_stats'],
            'data_stats': {
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'test_samples': len(test_data),
                'total_users': data_analysis['users']['total_unique']
            }
        }

        with open('improved_results_summary.json', 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info("\nResults saved:")
        logger.info("  - improved_results_summary.pkl (detailed results)")
        logger.info("  - improved_results_summary.json (summary)")
        logger.info("  - training_history.png (training curves)")
        logger.info("  - evaluation_results.png (evaluation plots)")
        logger.info("  - data_quality_dashboard.png (data analysis)")

        return results_summary

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        raise
