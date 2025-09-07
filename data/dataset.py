"""
Enhanced Dataset with Improved Sequence Creation
Clean implementation without boolean tensor issues
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
                
                # Ensure audio features
                if 'audio' not in item:
                    item['audio'] = torch.zeros(self.config.audio_dim)
                elif not isinstance(item['audio'], torch.Tensor):
                    item['audio'] = torch.tensor(item['audio'], dtype=torch.float32)
                
                # Ensure video features
                if 'video' not in item:
                    item['video'] = torch.zeros(self.config.video_dim)
                elif not isinstance(item['video'], torch.Tensor):
                    item['video'] = torch.tensor(item['video'], dtype=torch.float32)
                
                # NEW: Ensure physio features
                if 'physio' not in item:
                    item['physio'] = torch.zeros(self.config.physio_dim)
                elif not isinstance(item['physio'], torch.Tensor):
                    item['physio'] = torch.tensor(item['physio'], dtype=torch.float32)
                
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
        
        # Group by user
        user_data = defaultdict(list)
        for i, item in enumerate(self.data):
            user_id = str(item['user_id'])
            user_data[user_id].append((i, item))

        sequences = []
        
        # Process users with multiple posts
        multi_post_users = {uid: data for uid, data in user_data.items() if len(data) >= self.config.min_user_posts}
        
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
                        sequence_indices.append(sequence_indices[-1])
                    sequences.append({
                        'indices': sequence_indices,
                        'user_id': user_id,
                        'length': seq_len,
                        'type': 'user_based'
                    })

        # Pseudo sequences creation (optional)
        if len(sequences) < 1000:
            single_post_users = {uid: data for uid, data in user_data.items() if len(data) < self.config.min_user_posts}
            
            label_groups = defaultdict(list)
            for user_id, user_posts in single_post_users.items():
                for idx, item in user_posts:
                    label = item['label']
                    label_groups[label].append(idx)

            for label, indices in label_groups.items():
                indices.sort(key=lambda x: self.data[x].get('timestamp', 0))
                np.random.shuffle(indices)
                
                for start in range(0, len(indices) - self.config.min_sequence_length + 1,
                                self.config.sequence_length):
                    sequence_indices = indices[start:start + self.config.sequence_length]
                    if len(sequence_indices) >= self.config.min_sequence_length:
                        while len(sequence_indices) < self.config.sequence_length:
                            sequence_indices.append(sequence_indices[-1])
                        sequences.append({
                            'indices': sequence_indices,
                            'user_id': f'pseudo_label_{label}',
                            'length': len([i for i in sequence_indices if i < len(self.data)]),
                            'type': 'pseudo_temporal'
                        })

        logger.info(f"Created {len(sequences)} total sequences")
        sequence_types = Counter([seq['type'] for seq in sequences])
        logger.info(f"Sequence types: {dict(sequence_types)}")
        
        return sequences

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
        labels = []
        for seq in self.sequences:
            try:
                seq_labels = [self.data[i]['label'] for i in seq['indices'] if i < len(self.data)]
                if seq_labels:
                    majority_label = Counter(seq_labels).most_common(1)[0][0]
                    labels.append(majority_label)
            except Exception as e:
                logger.warning(f"Error getting sequence label: {e}")
                continue
        
        if not labels:
            return torch.ones(self.config.num_classes, dtype=torch.float32)
        
        class_counts = Counter(labels)
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

            sequence_label = Counter(labels.tolist()).most_common(1)[0][0]

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
    logger.info("="*30)
