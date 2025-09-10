"""
Enhanced Dataset with Improved Sequence Creation
Clean implementation without boolean tensor issues
FIXED VERSION: Class-preserving label aggregation to prevent loss of Euthymia (class 2)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
import configs.config as Config

logger = logging.getLogger(__name__)

class ImprovedTemporalDataset(Dataset):
    """Enhanced dataset with improved sequence creation and data augmentation."""

    def __init__(self, data_list: List[Dict], config: Config.Config, mode: str = 'train'):
        self.config = config
        self.mode = mode

        # Ensure attributes exist
        self.config.min_user_posts = getattr(config, 'min_user_posts', 2)
        self.config.overlap_ratio = getattr(config, 'overlap_ratio', 0.25)
        self.config.use_data_augmentation = getattr(config, 'use_data_augmentation', True)
        self.config.noise_factor = getattr(config, 'noise_factor', 0.01)
        self.config.dropout_augmentation_rate = getattr(config, 'dropout_augmentation_rate', 0.1)

        # Process data
        self.data = self._clean_and_prepare_data(data_list)
        if len(self.data) == 0:
            raise ValueError(f"No valid data remaining after cleaning for {mode} dataset")

        self.sequences = self._create_improved_sequences()
        if len(self.sequences) == 0:
            raise ValueError(f"No valid sequences created for {mode} dataset")

        # DIAGNOSTIC: Log label distribution after sequence creation
        self._log_sequence_labels()

        logger.info(f"Dataset {mode} initialized with {len(self.sequences)} sequences")

    def _clean_and_prepare_data(self, data_list: List[Dict]) -> List[Dict]:
        """Clean and prepare data for sequence creation"""

        cleaned_data = []

        for item in data_list:
            try:
                # user_id fallback
                if 'user_id' not in item:
                    if 'User_ID' in item:
                        item['user_id'] = item['User_ID']
                    else:
                        item['user_id'] = 'unknown'

                # Ensure text features
                if 'text' not in item:
                    item['text'] = torch.zeros(self.config.text_dim)
                elif not isinstance(item['text'], torch.Tensor):
                    item['text'] = torch.tensor(item['text'], dtype=torch.float32)
                item['text'] = torch.nan_to_num(item['text'], nan=0.0, posinf=0.0, neginf=0.0)

                # Ensure audio features
                if 'audio' not in item:
                    item['audio'] = torch.zeros(self.config.audio_dim)
                elif not isinstance(item['audio'], torch.Tensor):
                    item['audio'] = torch.tensor(item['audio'], dtype=torch.float32)
                if torch.isnan(item['audio']).any() or torch.isinf(item['audio']).any():
                    logger.warning(f"NaN or Inf detected in audio features for user_id: {item.get('user_id')}, replacing with zeros.")
                item['audio'] = torch.nan_to_num(item['audio'], nan=0.0, posinf=0.0, neginf=0.0)

                # Ensure video features
                if 'video' not in item:
                    item['video'] = torch.zeros(self.config.video_dim)
                elif not isinstance(item['video'], torch.Tensor):
                    item['video'] = torch.tensor(item['video'], dtype=torch.float32)
                item['video'] = torch.nan_to_num(item['video'], nan=0.0, posinf=0.0, neginf=0.0)

                # NEW: Ensure physio features
                if 'physio' not in item:
                    item['physio'] = torch.zeros(self.config.physio_dim)
                elif not isinstance(item['physio'], torch.Tensor):
                    item['physio'] = torch.tensor(item['physio'], dtype=torch.float32)
                item['physio'] = torch.nan_to_num(item['physio'], nan=0.0, posinf=0.0, neginf=0.0)

                # Label fallback
                if 'label' not in item:
                    if 'Label' in item:
                        item['label'] = item['Label']
                    else:
                        item['label'] = 0

                # Timestamp fallback
                if 'timestamp' not in item:
                    if 'Timestamp' in item:
                        item['timestamp'] = item['Timestamp']
                    else:
                        item['timestamp'] = 0

                cleaned_data.append(item)

            except Exception as e:
                logger.warning(f"Error processing item: {e}")
                continue

        logger.info(f"Cleaned data: {len(cleaned_data)} items from {len(data_list)} original items")
        return cleaned_data

    def _create_improved_sequences(self) -> List[Dict[str, Any]]:
        """Create improved temporal sequences without boolean operations"""

        logger.info(f"Creating improved sequences for {self.mode} dataset...")

        # DIAGNOSTIC: Log original label distribution
        original_labels = [item['label'] for item in self.data]
        original_counts = Counter(original_labels)
        logger.info(f"Original label distribution before sequences: {dict(original_counts)}")

        # Group by user
        user_data = defaultdict(list)
        for i, item in enumerate(self.data):
            user_id = str(item['user_id'])
            user_data[user_id].append((i, item))

        sequences = []

        # ENHANCED: Process users with class-aware filtering
        multi_post_users = {}
        rare_class_users = {}

        for uid, data in user_data.items():
            user_labels = [item[1]['label'] for item in data]
            has_rare_class = 2 in user_labels  # Check for Euthymia

            if has_rare_class:
                # Keep users with rare class even if they have fewer posts
                if len(data) >= max(1, self.config.min_user_posts - 1):
                    rare_class_users[uid] = data
            elif len(data) >= self.config.min_user_posts:
                multi_post_users[uid] = data

        logger.info(f"Found {len(multi_post_users)} regular users with {self.config.min_user_posts}+ posts")
        logger.info(f"Found {len(rare_class_users)} users with rare class (Euthymia)")

        # Combine all eligible users
        all_eligible_users = {**multi_post_users, **rare_class_users}

        for user_id, user_posts in all_eligible_users.items():
            user_posts.sort(key=lambda x: x[1].get('timestamp', 0))
            indices = [x[0] for x in user_posts]

            # ENHANCED: Adaptive sequence length for rare class users
            user_labels = [x[1]['label'] for x in user_posts]
            has_rare_class = 2 in user_labels

            if has_rare_class:
                # More flexible sequence creation for rare class
                seq_len = min(self.config.sequence_length, max(2, len(indices)))
                min_seq_len = max(1, self.config.min_sequence_length - 1)
            else:
                seq_len = min(self.config.sequence_length, len(indices))
                min_seq_len = self.config.min_sequence_length

            if seq_len >= min_seq_len:
                step_size = max(1, int(seq_len * (1 - self.config.overlap_ratio)))

                for start in range(0, len(indices) - seq_len + 1, step_size):
                    sequence_indices = indices[start:start + seq_len]
                    while len(sequence_indices) < self.config.sequence_length:
                        sequence_indices.append(sequence_indices[-1])

                    # DIAGNOSTIC: Check what labels are in this sequence
                    seq_labels = [self.data[i]['label'] for i in sequence_indices[:seq_len]]

                    sequences.append({
                        'indices': sequence_indices,
                        'user_id': user_id,
                        'length': seq_len,
                        'type': 'rare_class_user' if has_rare_class else 'user_based',
                        'original_labels': seq_labels  # For debugging
                    })

        # Pseudo sequences creation (enhanced for rare classes)
        if len(sequences) < 1000:
            single_post_users = {uid: data for uid, data in user_data.items() 
                                if uid not in all_eligible_users}

            label_groups = defaultdict(list)
            for user_id, user_posts in single_post_users.items():
                for idx, item in user_posts:
                    label = item['label']
                    label_groups[label].append(idx)

            # ENHANCED: Give priority to rare classes in pseudo sequences
            for label in sorted(label_groups.keys(), key=lambda x: (x != 2, x)):  # Process class 2 first
                indices = label_groups[label]
                indices.sort(key=lambda x: self.data[x].get('timestamp', 0))
                np.random.shuffle(indices)

                min_seq_len = 1 if label == 2 else self.config.min_sequence_length  # More flexible for class 2

                for start in range(0, len(indices) - min_seq_len + 1,
                                self.config.sequence_length):
                    sequence_indices = indices[start:start + self.config.sequence_length]
                    if len(sequence_indices) >= min_seq_len:
                        while len(sequence_indices) < self.config.sequence_length:
                            sequence_indices.append(sequence_indices[-1])
                        sequences.append({
                            'indices': sequence_indices,
                            'user_id': f'pseudo_label_{label}',
                            'length': len([i for i in sequence_indices if i < len(self.data)]),
                            'type': f'pseudo_temporal_class_{label}',
                            'original_labels': [label] * len(sequence_indices)  # For debugging
                        })

        logger.info(f"Created {len(sequences)} total sequences")
        sequence_types = Counter([seq['type'] for seq in sequences])
        logger.info(f"Sequence types: {dict(sequence_types)}")

        return sequences

    def _log_sequence_labels(self):
        """Diagnostic function to log sequence label distribution"""
        try:
            predicted_labels = []
            for seq in self.sequences:
                seq_labels = [self.data[i]['label'] for i in seq['indices'] 
                            if i < len(self.data)]
                if seq_labels:
                    # Use the same logic as __getitem__ for consistency
                    label_counts = Counter(seq_labels)
                    if 2 in label_counts:  # Euthymia
                        sequence_label = 2
                    elif 0 in label_counts and 1 in label_counts:
                        sequence_label = label_counts.most_common(1)[0][0]
                    else:
                        sequence_label = label_counts.most_common(1)[0][0]
                    predicted_labels.append(sequence_label)

            final_counts = Counter(predicted_labels)
            logger.info(f"DIAGNOSTIC - Final sequence labels: {dict(final_counts)}")

            # Check for class 2 preservation
            if 2 in final_counts:
                logger.info(f"SUCCESS: Class 2 (Euthymia) preserved with {final_counts[2]} sequences")
            else:
                logger.warning("WARNING: Class 2 (Euthymia) not found in sequences!")

        except Exception as e:
            logger.error(f"Error in diagnostic logging: {e}")

    def _augment_sequence(self, text_features: torch.Tensor,
                         audio_features: torch.Tensor,
                         video_features: torch.Tensor,
                         physio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply data augmentation to sequences safely"""

        if not self.config.use_data_augmentation or self.mode != 'train':
            return text_features, audio_features, video_features, physio_features

        try:
            if np.random.random() < 0.5:
                noise_factor = self.config.noise_factor * 2
                text_features = text_features + torch.randn_like(text_features) * noise_factor
                audio_features = audio_features + torch.randn_like(audio_features) * noise_factor
                video_features = video_features + torch.randn_like(video_features) * noise_factor
                physio_features = physio_features + torch.randn_like(physio_features) * noise_factor

            if np.random.random() < 0.3:
                dropout_rate = self.config.dropout_augmentation_rate * 1.5
                keep_prob = 1.0 - dropout_rate

                def apply_mask(features):
                    mask = torch.bernoulli(torch.full_like(features, keep_prob))
                    return features * mask

                text_features = apply_mask(text_features)
                audio_features = apply_mask(audio_features)
                video_features = apply_mask(video_features)
                physio_features = apply_mask(physio_features)

        except Exception as e:
            logger.warning(f"Augmentation failed: {e}. Using original features.")

        return text_features, audio_features, video_features, physio_features

    def get_class_weights(self) -> torch.Tensor:
        """FIXED: Class-preserving weight calculation"""
        labels = []
        for seq in self.sequences:
            try:
                seq_labels = [self.data[i]['label'] for i in seq['indices'] if i < len(self.data)]
                if seq_labels:
                    # FIXED: Use class-preserving logic instead of simple majority voting
                    label_counts = Counter(seq_labels)

                    # Priority: Preserve rare classes (especially class 2 - Euthymia)
                    if 2 in label_counts:
                        majority_label = 2
                    elif 0 in label_counts and 1 in label_counts:
                        # Mixed depression/mania - use majority vote
                        majority_label = label_counts.most_common(1)[0][0]
                    else:
                        # Single class or clear majority
                        majority_label = label_counts.most_common(1)[0][0]

                    labels.append(majority_label)
            except Exception as e:
                logger.warning(f"Error getting sequence label: {e}")
                continue

        if not labels:
            return torch.ones(self.config.num_classes, dtype=torch.float32)

        class_counts = Counter(labels)
        logger.info(f"Class distribution for weight calculation: {dict(class_counts)}")

        total_samples = len(labels)
        weights = []
        for i in range(self.config.num_classes):
            count = class_counts.get(i, 1)
            weight = total_samples / (self.config.num_classes * count)
            weights.append(weight)

        weights = torch.tensor(weights, dtype=torch.float32)
        return weights / weights.sum() * self.config.num_classes

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """FIXED: Class-preserving label aggregation"""
        try:
            sequence_info = self.sequences[idx]
            indices = sequence_info['indices']

            text_features = []
            audio_features = []
            video_features = []
            physio_features = []
            labels = []

            for i in indices:
                if i < len(self.data):
                    item = self.data[i]
                    text_features.append(item['text'])
                    audio_features.append(item['audio'])
                    video_features.append(item['video'])
                    physio_features.append(item.get('physio', torch.zeros(self.config.physio_dim)))
                    labels.append(item['label'])
                else:
                    text_features.append(torch.zeros(self.config.text_dim))
                    audio_features.append(torch.zeros(self.config.audio_dim))
                    video_features.append(torch.zeros(self.config.video_dim))
                    physio_features.append(torch.zeros(self.config.physio_dim))
                    labels.append(0)

            text_features = torch.stack(text_features)
            audio_features = torch.stack(audio_features)
            video_features = torch.stack(video_features)
            physio_features = torch.stack(physio_features)
            labels = torch.tensor(labels, dtype=torch.long)

            text_features, audio_features, video_features, physio_features = self._augment_sequence(
                text_features, audio_features, video_features, physio_features
            )

            # FIXED: Class-preserving label aggregation logic
            label_counts = Counter(labels.tolist())

            # Priority-based label selection to preserve rare classes
            if 2 in label_counts:  # Euthymia (rare class)
                sequence_label = 2
            elif 0 in label_counts and 1 in label_counts:  # Mixed depression/mania
                # Use majority vote for mixed non-euthymia cases
                sequence_label = label_counts.most_common(1)[0][0]
            else:
                # Single class or clear majority
                sequence_label = label_counts.most_common(1)[0][0]

            return {
                'text': text_features,
                'audio': audio_features,
                'video': video_features,
                'physio': physio_features,
                'labels': labels,
                'sequence_label': torch.tensor(sequence_label, dtype=torch.long),
                'sequence_type': sequence_info.get('type', 'unknown'),
                'user_id': sequence_info.get('user_id', 'unknown')
            }
        except Exception as e:
            logger.error(f"Error getting item {idx}: {e}")
            return {
                'text': torch.zeros(self.config.sequence_length, self.config.text_dim),
                'audio': torch.zeros(self.config.sequence_length, self.config.audio_dim),
                'video': torch.zeros(self.config.sequence_length, self.config.video_dim),
                'physio': torch.zeros(self.config.sequence_length, self.config.physio_dim),
                'labels': torch.zeros(self.config.sequence_length, dtype=torch.long),
                'sequence_label': torch.tensor(0, dtype=torch.long),
                'sequence_type': 'error',
                'user_id': 'error'
            }


def create_improved_data_loader(dataset, batch_size, shuffle=False, use_sampler=False):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not use_sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )


def create_improved_user_split(full_data, train_ratio=0.7, val_ratio=0.15):
    user_data = defaultdict(list)
    for item in full_data:
        user_id = item.get('user_id', item.get('User_ID', 'unknown'))
        user_data[user_id].append(item)

    users = list(user_data.keys())
    np.random.seed(42)
    np.random.shuffle(users)

    n_users = len(users)
    train_end = int(train_ratio * n_users)
    val_end = int((train_ratio + val_ratio) * n_users)

    train_users = users[:train_end]
    val_users = users[train_end:val_end]
    test_users = users[val_end:]

    logger.info(f"User split - Train: {len(train_users)} users ({sum(len(user_data[u]) for u in train_users)} samples)")
    logger.info(f"User split - Val: {len(val_users)} users ({sum(len(user_data[u]) for u in val_users)} samples)")
    logger.info(f"User split - Test: {len(test_users)} users ({sum(len(user_data[u]) for u in test_users)} samples)")

    train_data = [item for u in train_users for item in user_data[u]]
    val_data = [item for u in val_users for item in user_data[u]]
    test_data = [item for u in test_users for item in user_data[u]]

    return train_data, val_data, test_data


def analyze_data_distribution(data):
    analysis = {
        'total_samples': len(data),
        'users': len(set(item.get('user_id', item.get('User_ID', 'unknown')) for item in data)),
        'labels': Counter(item.get('label', item.get('Label', 0)) for item in data)
    }

    logger.info(f"Data analysis: {analysis}")
    return analysis


def suggest_hyperparameters(analysis):
    suggestions = {
        'batch_size': min(32, max(8, analysis['total_samples'] // 100)),
        'learning_rate': 1e-4 if analysis['total_samples'] > 1000 else 5e-4,
        'epochs': min(100, max(20, analysis['total_samples'] // 50))
    }
    return suggestions


def create_data_quality_dashboard(analysis):
    logger.info("=== DATA QUALITY DASHBOARD ===")
    logger.info(f"Total samples: {analysis['total_samples']}")
    logger.info(f"Unique users: {analysis['users']}")
    logger.info(f"Label distribution: {dict(analysis['labels'])}")
    logger.info("=" * 30)
