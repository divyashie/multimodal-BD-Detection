# ==============================================================================
# --- ENHANCED DATASET WITH BETTER SEQUENCE CREATION ---
# ==============================================================================
import torch
from torch.utils.data import Dataset
from collections import Counter
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from .quality import DataQualityAnalyzer, EnhancedDataCleaner
from configs.config import Config

logger = logging.getLogger(__name__)

class ImprovedTemporalDataset(Dataset):
    """Improved dataset with better sequence creation and data augmentation."""

    def __init__(self, data_list: List[Dict], config: Config, mode: str = 'train'):
        self.config = config
        self.config.min_user_posts = 2  # Reduced to increase sequence count
        self.mode = mode
        self.quality_analyzer = DataQualityAnalyzer(config)
        self.data_cleaner = EnhancedDataCleaner(config)
        if mode == 'train':
            self.quality_report = self.quality_analyzer.analyze_data_quality(data_list)
        self.data = self.data_cleaner.clean_data(data_list, mode)
        if len(self.data) == 0:
            raise ValueError(f"No valid data remaining after cleaning for {mode} dataset")
        self.sequences = self._create_improved_sequences()
        if len(self.sequences) == 0:
            raise ValueError(f"No valid sequences created for {mode} dataset")
        logger.info(f"Dataset {mode} initialized with {len(self.sequences)} sequences")

    def _create_improved_sequences(self) -> List[Dict[str, Any]]:
        """Create improved temporal sequences."""
        logger.info(f"Creating improved sequences for {self.mode} dataset...")
        user_data = {}
        for i, item in enumerate(self.data):
            user_id = item['user_id']
            if user_id not in user_data:
                user_data[user_id] = []
            user_data[user_id].append((i, item))

        sequences = []
        multi_post_users = {uid: data for uid, data in user_data.items()
                           if len(data) >= self.config.min_user_posts}
        logger.info(f"Found {len(multi_post_users)} users with {self.config.min_user_posts}+ posts")

        for user_id, user_posts in multi_post_users.items():
            user_posts.sort(key=lambda x: x[1].get('timestamp', 0))
            indices = [x[0] for x in user_posts]
            seq_len = min(self.config.sequence_length, len(indices))
            if seq_len >= self.config.min_sequence_length:
                step_size = max(1, int(seq_len * (1 - self.config.overlap_ratio)))
                for start in range(0, len(indices) - seq_len + 1, step_size):
                    sequence_indices = indices[start:start + seq_len]
                    while len(sequence_indices) < self.config.sequence_length:
                        sequence_indices.append(sequence_indices[-1])  # Pad with last index
                    sequences.append({
                        'indices': sequence_indices,
                        'user_id': user_id,
                        'length': seq_len,
                        'type': 'user_based'
                    })

        # Enhanced pseudo-sequences with shuffling
        if len(sequences) < 2000:  # Lower threshold for more pseudo data
            single_post_users = {uid: data for uid, data in user_data.items()
                               if len(data) < self.config.min_user_posts}
            label_groups = {}
            for user_id, user_posts in single_post_users.items():
                for idx, item in user_posts:
                    label = item['label']
                    if label not in label_groups:
                        label_groups[label] = []
                    label_groups[label].append(idx)

            for label, indices in label_groups.items():
                indices.sort(key=lambda x: self.data[x].get('timestamp', 0))
                np.random.shuffle(indices)  # Add randomness to avoid temporal bias
                for start in range(0, len(indices) - self.config.min_sequence_length + 1,
                                self.config.sequence_length):
                    sequence_indices = indices[start:start + self.config.sequence_length]
                    if len(sequence_indices) >= self.config.min_sequence_length:
                        while len(sequence_indices) < self.config.sequence_length:
                            sequence_indices.append(sequence_indices[-1])
                        sequences.append({
                            'indices': sequence_indices,
                            'user_id': f'pseudo_label_{label}',
                            'length': len(sequence_indices),
                            'type': 'pseudo_temporal'
                        })

        logger.info(f"Created {len(sequences)} total sequences")
        logger.info(f"Sequence types: {Counter([seq['type'] for seq in sequences])}")
        return sequences

    def _augment_sequence(self, text_features: torch.Tensor,
                         audio_features: torch.Tensor,
                         video_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply data augmentation to sequences."""
        if not self.config.use_data_augmentation or self.mode != 'train':
            return text_features, audio_features, video_features
        if np.random.random() < 0.5:  # Increase chance to 50%
            noise_factor = self.config.noise_factor * 2  # Double noise for variety
            text_features += torch.randn_like(text_features) * noise_factor
            audio_features += torch.randn_like(audio_features) * noise_factor
            video_features += torch.randn_like(video_features) * noise_factor
        if np.random.random() < 0.3:  # Increase chance to 30%
            dropout_rate = self.config.dropout_augmentation_rate * 1.5
            text_mask = torch.rand_like(text_features) > dropout_rate
            audio_mask = torch.rand_like(audio_features) > dropout_rate
            video_mask = torch.rand_like(video_features) > dropout_rate
            text_features = text_features * text_mask
            audio_features = audio_features * audio_mask
            video_features = video_features * video_mask
        return text_features, audio_features, video_features

    def get_class_weights(self) -> torch.Tensor:
        """Calculate balanced class weights based on sequence labels."""
        labels = []
        for seq in self.sequences:
            # Get majority label from sequence
            seq_labels = [self.data[i]['label'] for i in seq['indices']]
            majority_label = Counter(seq_labels).most_common(1)[0][0]
            labels.append(majority_label)

        class_counts = Counter(labels)
        total_samples = len(labels)

        # Calculate inverse frequency weights
        weights = []
        for i in range(self.config.num_classes):
            count = class_counts.get(i, 1)  # Avoid division by zero
            weight = total_samples / (self.config.num_classes * count)
            weights.append(weight)

        weights = torch.tensor(weights, dtype=torch.float32)
        return weights / weights.sum() * self.config.num_classes  # Normalize

    def __len__(self) -> int:
        """Return the total number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sequence by index."""
        sequence_info = self.sequences[idx]
        indices = sequence_info['indices']

        # Extract features and labels
        text_features = torch.stack([self.data[i]['text'] for i in indices])
        audio_features = torch.stack([self.data[i]['audio'] for i in indices])
        video_features = torch.stack([self.data[i]['video'] for i in indices])
        labels = torch.tensor([self.data[i]['label'] for i in indices], dtype=torch.long)

        # Apply augmentation
        text_features, audio_features, video_features = self._augment_sequence(
            text_features, audio_features, video_features
        )

        # Determine sequence label (majority vote)
        sequence_label = Counter(labels.tolist()).most_common(1)[0][0]

        return {
            'text': text_features,
            'audio': audio_features,
            'video': video_features,
            'labels': labels,
            'sequence_label': torch.tensor(sequence_label, dtype=torch.long),
            'sequence_type': sequence_info.get('type', 'unknown'),
            'user_id': sequence_info.get('user_id', 'unknown')
        }