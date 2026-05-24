"""
Simple Dataset for Ablation Study
Does NOT perform sequence grouping - expects pre-processed samples
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class AblationDataset(Dataset):
    """
    Simple dataset wrapper for ablation study

    Expects samples that are already processed with:
    - text: tensor of shape [seq_len, text_dim] or [text_dim]
    - audio: tensor of shape [seq_len, audio_dim] or [audio_dim]
    - video: tensor of shape [seq_len, video_dim] or [video_dim]
    - physio: tensor of shape [seq_len, physio_dim] or [physio_dim]
    - label: int (0, 1, or 2)

    Does NOT group by user or create sequences - treats each sample independently
    """

    def __init__(self, data_list: List[Dict], config, mode: str = 'train'):
        self.config = config
        self.mode = mode
        self.data = self._prepare_data(data_list)

        logger.info(f"AblationDataset {mode} initialized with {len(self.data)} samples")
        self._log_label_distribution()

    def _prepare_data(self, data_list: List[Dict]) -> List[Dict]:
        """
        Prepare data - ensure all tensors are correct shape and handle missing fields
        """
        prepared = []

        for item in data_list:
            try:
                # Ensure required fields exist
                sample = {}

                # Text features
                if 'text' in item:
                    text = item['text']
                    if not isinstance(text, torch.Tensor):
                        text = torch.tensor(text, dtype=torch.float32)
                    text = torch.nan_to_num(text, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    text = torch.zeros(self.config.text_dim, dtype=torch.float32)

                # Ensure it's 2D: [seq_len, dim]
                if text.dim() == 1:
                    text = text.unsqueeze(0)  # [dim] -> [1, dim]
                sample['text'] = text

                # Audio features
                if 'audio' in item:
                    audio = item['audio']
                    if not isinstance(audio, torch.Tensor):
                        audio = torch.tensor(audio, dtype=torch.float32)
                    audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    audio = torch.zeros(self.config.audio_dim, dtype=torch.float32)

                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                sample['audio'] = audio

                # Video features
                if 'video' in item:
                    video = item['video']
                    if not isinstance(video, torch.Tensor):
                        video = torch.tensor(video, dtype=torch.float32)
                    video = torch.nan_to_num(video, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    video = torch.zeros(self.config.video_dim, dtype=torch.float32)

                if video.dim() == 1:
                    video = video.unsqueeze(0)
                sample['video'] = video

                # Physio features
                if 'physio' in item:
                    physio = item['physio']
                    if not isinstance(physio, torch.Tensor):
                        physio = torch.tensor(physio, dtype=torch.float32)
                    physio = torch.nan_to_num(physio, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    physio = torch.zeros(self.config.physio_dim, dtype=torch.float32)

                if physio.dim() == 1:
                    physio = physio.unsqueeze(0)
                sample['physio'] = physio

                # Label
                sample['label'] = int(item.get('label', 0))

                # Metadata
                sample['user_id'] = item.get('user_id', 'unknown')
                sample['source'] = item.get('source', 'unknown')

                prepared.append(sample)

            except Exception as e:
                logger.warning(f"Error preparing sample: {e}")
                continue

        return prepared

    def _log_label_distribution(self):
        """Log class distribution"""
        labels = [item['label'] for item in self.data]
        unique, counts = np.unique(labels, return_counts=True)

        logger.info(f"Label distribution in {self.mode} set:")
        class_names = ['Depression', 'Mania', 'Euthymia']
        for label, count in zip(unique, counts):
            pct = count / len(labels) * 100
            logger.info(f"  Class {label} ({class_names[label]}): {count} ({pct:.1f}%)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a single sample

        Returns dict with:
        - text: [seq_len, text_dim]
        - audio: [seq_len, audio_dim]
        - video: [seq_len, video_dim]
        - physio: [seq_len, physio_dim]
        - sequence_label: scalar (for compatibility with trainer)
        - user_id: str
        - sequence_type: str (for compatibility with evaluator)
        """
        sample = self.data[idx]

        return {
            'text': sample['text'],
            'audio': sample['audio'],
            'video': sample['video'],
            'physio': sample['physio'],
            'sequence_label': torch.tensor(sample['label'], dtype=torch.long),
            'user_id': sample['user_id'],
            'sequence_type': 'ablation_sample'  # For evaluator compatibility
        }
