# ==============================================================================
# --- DATA QUALITY IMPROVEMENT ---
# ==============================================================================

import torch
import numpy as np
from collections import Counter
from typing import Dict, List, Any
import logging
from scipy import stats
from configs.config import Config

logger = logging.getLogger(__name__)

class DataQualityAnalyzer:
    """Comprehensive data quality analysis and cleaning."""

    def __init__(self, config: Config):
        self.config = config
        self.config.missing_threshold = 0.3  # Increased from 0.1 to retain more data
        self.quality_report = {}

    def analyze_data_quality(self, data_list: List[Dict]) -> Dict[str, Any]:
        """Comprehensive data quality analysis."""
        logger.info("Starting comprehensive data quality analysis...")

        report = {
            'total_samples': len(data_list),
            'missing_data': self._analyze_missing_data(data_list),
            'outliers': self._detect_outliers(data_list),
            'duplicates': self._detect_duplicates(data_list),
            'class_distribution': self._analyze_class_distribution(data_list),
            'temporal_analysis': self._analyze_temporal_distribution(data_list),
            'feature_statistics': self._analyze_feature_statistics(data_list)
        }

        self.quality_report = report
        self._print_quality_report(report)
        return report

    def _analyze_missing_data(self, data_list: List[Dict]) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_stats = {}
        required_fields = ['user_id', 'label']  # Reduced critical fields to minimize loss

        for field in required_fields:
            missing_count = sum(1 for item in data_list if item.get(field) is None)
            missing_stats[field] = {
                'count': missing_count,
                'percentage': (missing_count / len(data_list)) * 100
            }

        return missing_stats

    def _detect_outliers(self, data_list: List[Dict]) -> Dict[str, int]:
        """Detect outliers in numerical features."""
        outlier_counts = {'text': 0, 'audio': 0, 'video': 0}

        for modality in ['text', 'audio', 'video']:
            features = []
            for item in data_list:
                if item.get(modality) is not None:
                    try:
                        if isinstance(item[modality], torch.Tensor):
                            features.append(item[modality].numpy().flatten())
                        elif isinstance(item[modality], np.ndarray):
                            features.append(item[modality].flatten())
                    except:
                        continue

            if features and len(features) > 1:
                try:
                    features = np.vstack(features)
                    # Use robust statistics for outlier detection
                    z_scores = np.abs(stats.zscore(features, axis=0, nan_policy='omit'))
                    # Cap outliers at 4.0 instead of 3.0 to retain more data
                    outlier_counts[modality] = np.sum(np.any(z_scores > self.config.outlier_threshold * 1.33, axis=1))
                except:
                    outlier_counts[modality] = 0

        return outlier_counts
    
    def _impute_missing(self, data_list: List[Dict]) -> List[Dict]:
        """Impute missing modality features with zeros."""
        for item in data_list:
            for modality in ['text', 'audio', 'video']:
                if item.get(modality) is None:
                    item[modality] = torch.zeros(self.config.text_dim if modality == 'text' else
                                               self.config.audio_dim if modality == 'audio' else
                                               self.config.video_dim)
        return data_list
    
    def _detect_duplicates(self, data_list: List[Dict]) -> Dict[str, int]:
        """Detect duplicate entries."""
        # Create a simplified representation for duplicate detection
        simplified_data = []
        for item in data_list:
            key = (
                item.get('user_id'),
                item.get('timestamp'),
                item.get('label'),
                str(item.get('text', ''))[:50] if item.get('text') is not None else ''
            )
            simplified_data.append(key)

        unique_data = set(simplified_data)
        return {
            'total_duplicates': len(simplified_data) - len(unique_data),
            'unique_samples': len(unique_data)
        }

    def _analyze_class_distribution(self, data_list: List[Dict]) -> Dict[str, Any]:
        """Analyze class distribution and imbalance."""
        labels = [item.get('label') for item in data_list if item.get('label') is not None]
        class_counts = Counter(labels)
        total = len(labels)

        distribution = {}
        for class_id, count in class_counts.items():
            distribution[f'class_{class_id}'] = {
                'count': count,
                'percentage': (count / total) * 100
            }

        # Calculate imbalance ratio
        max_count = max(class_counts.values()) if class_counts else 1
        min_count = min(class_counts.values()) if class_counts else 1
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        distribution['imbalance_ratio'] = imbalance_ratio
        return distribution

    def _analyze_temporal_distribution(self, data_list: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal distribution of data."""
        timestamps = [item.get('timestamp', 0) for item in data_list
                     if item.get('timestamp') is not None]

        if not timestamps:
            return {'error': 'No valid timestamps found'}

        timestamps = np.array(timestamps)
        return {
            'min_timestamp': float(np.min(timestamps)),
            'max_timestamp': float(np.max(timestamps)),
            'timestamp_range': float(np.max(timestamps) - np.min(timestamps)),
            'median_timestamp': float(np.median(timestamps))
        }

    def _analyze_feature_statistics(self, data_list: List[Dict]) -> Dict[str, Any]:
        """Analyze statistical properties of features."""
        stats = {}

        for modality in ['text', 'audio', 'video']:
            modality_stats = {'valid_samples': 0, 'invalid_samples': 0}

            for item in data_list:
                feature = item.get(modality)
                if feature is not None:
                    try:
                        if isinstance(feature, torch.Tensor):
                            feature_array = feature.numpy()
                        else:
                            feature_array = np.array(feature)

                        # Check for NaN, Inf, or extreme values
                        if np.isnan(feature_array).any() or np.isinf(feature_array).any():
                            modality_stats['invalid_samples'] += 1
                        else:
                            modality_stats['valid_samples'] += 1
                    except:
                        modality_stats['invalid_samples'] += 1
                else:
                    modality_stats['invalid_samples'] += 1

            stats[modality] = modality_stats

        return stats

    def _print_quality_report(self, report: Dict[str, Any]):
        """Print formatted quality report."""
        logger.info("\n" + "="*60)
        logger.info("DATA QUALITY ANALYSIS REPORT")
        logger.info("="*60)

        logger.info(f"Total samples: {report['total_samples']}")

        logger.info("\nMissing Data:")
        for field, stats in report['missing_data'].items():
            logger.info(f"  {field}: {stats['count']} ({stats['percentage']:.1f}%)")

        logger.info("\nClass Distribution:")
        for class_name, stats in report['class_distribution'].items():
            if 'class_' in class_name:
                logger.info(f"  {class_name}: {stats['count']} ({stats['percentage']:.1f}%)")
        logger.info(f"  Imbalance ratio: {report['class_distribution']['imbalance_ratio']:.2f}")

        logger.info("\nFeature Validity:")
        for modality, stats in report['feature_statistics'].items():
            total = stats['valid_samples'] + stats['invalid_samples']
            valid_pct = (stats['valid_samples'] / total * 100) if total > 0 else 0
            logger.info(f"  {modality}: {valid_pct:.1f}% valid samples")

class EnhancedDataCleaner:
    """Enhanced data cleaning with multiple strategies."""

    def __init__(self, config: Config):
        self.config = config

    def clean_data(self, data_list: List[Dict], mode: str = 'train') -> List[Dict]:
        """Comprehensive data cleaning pipeline."""
        logger.info(f"Starting enhanced data cleaning for {mode} dataset...")

        original_count = len(data_list)
        
        # Add imputation before cleaning
        data_list = DataQualityAnalyzer(self.config)._impute_missing(data_list)
        
        # Step 1: Remove entries with missing critical fields
        data_list = self._remove_missing_critical_fields(data_list)
        logger.info(f"After removing missing critical fields: {len(data_list)} samples")

        # Step 2: Clean and validate feature tensors
        data_list = self._clean_feature_tensors(data_list)
        logger.info(f"After cleaning feature tensors: {len(data_list)} samples")

        # Step 3: Remove duplicates
        data_list = self._remove_duplicates(data_list)
        logger.info(f"After removing duplicates: {len(data_list)} samples")

        # Step 4: Handle outliers (only for training)
        if mode == 'train':
            data_list = self._handle_outliers(data_list)
            logger.info(f"After handling outliers: {len(data_list)} samples")

        # Step 5: Normalize timestamps
        data_list = self._normalize_timestamps(data_list)

        cleaned_count = len(data_list)
        removed_count = original_count - cleaned_count
        removal_percentage = (removed_count / original_count) * 100

        logger.info(f"Data cleaning complete for {mode}:")
        logger.info(f"  Original: {original_count} samples")
        logger.info(f"  Cleaned: {cleaned_count} samples")
        logger.info(f"  Removed: {removed_count} samples ({removal_percentage:.1f}%)")

        return data_list

    def _remove_missing_critical_fields(self, data_list: List[Dict]) -> List[Dict]:
        """Remove entries missing critical fields."""
        critical_fields = ['user_id', 'label']

        cleaned_data = []
        for item in data_list:
            if all(item.get(field) is not None for field in critical_fields):
                cleaned_data.append(item)

        return cleaned_data

    def _clean_feature_tensors(self, data_list: List[Dict]) -> List[Dict]:
        """Clean and validate feature tensors."""
        cleaned_data = []

        for item in data_list:
            is_valid = True
            cleaned_item = item.copy()

            for modality in ['text', 'audio', 'video']:
                feature = item.get(modality)

                if feature is not None:
                    try:
                        # Convert to tensor if needed
                        if not isinstance(feature, torch.Tensor):
                            feature = torch.tensor(feature, dtype=torch.float32)

                        # Check for NaN/Inf and replace with zeros
                        if torch.isnan(feature).any() or torch.isinf(feature).any():
                            logger.warning(f"Found NaN/Inf in {modality} feature, replacing with zeros")
                            feature = torch.where(torch.isnan(feature) | torch.isinf(feature),
                                                torch.zeros_like(feature), feature)

                        # Ensure correct dimensions
                        expected_dims = getattr(self.config, f'{modality}_dim')
                        if len(feature.shape) == 1:
                            if feature.shape[0] != expected_dims:
                                logger.warning(f"Incorrect {modality} dimension: {feature.shape[0]}, expected: {expected_dims}")
                                is_valid = False
                                break
                        else:
                            if feature.shape[-1] != expected_dims:
                                logger.warning(f"Incorrect {modality} dimension: {feature.shape[-1]}, expected: {expected_dims}")
                                is_valid = False
                                break

                        # Ensure tensor is 1D for consistency
                        if len(feature.shape) > 1:
                            feature = feature.flatten()
                            if len(feature) != expected_dims:
                                is_valid = False
                                break

                        cleaned_item[modality] = feature

                    except Exception as e:
                        logger.warning(f"Error processing {modality} feature: {e}")
                        is_valid = False
                        break
                else:
                    is_valid = False
                    break

            if is_valid:
                cleaned_data.append(cleaned_item)

        return cleaned_data

    def _remove_duplicates(self, data_list: List[Dict]) -> List[Dict]:
        """Remove duplicate entries."""
        seen_keys = set()
        unique_data = []

        for item in data_list:
            # Create a key for duplicate detection
            key = (
                item.get('user_id'),
                item.get('timestamp', 0),
                item.get('label'),
                # Use hash of tensor values for feature comparison
                hash(str(item.get('text', torch.tensor([])))) if item.get('text') is not None else 0
            )

            if key not in seen_keys:
                seen_keys.add(key)
                unique_data.append(item)

        return unique_data

    def _handle_outliers(self, data_list: List[Dict]) -> List[Dict]:
        """Handle outliers by capping extreme values."""
        for item in data_list:
            for modality in ['text', 'audio', 'video']:
                feature = item.get(modality)
                if feature is not None and isinstance(feature, torch.Tensor):
                    z_scores = torch.abs((feature - feature.mean()) / feature.std())
                    item[modality] = torch.where(z_scores > self.config.outlier_threshold * 1.33,
                                               torch.clamp(feature, -self.config.outlier_threshold * 1.33, self.config.outlier_threshold * 1.33),
                                               feature)
        return data_list

    def _normalize_timestamps(self, data_list: List[Dict]) -> List[Dict]:
        """Normalize timestamps to a consistent range."""
        timestamps = [item.get('timestamp', 0) for item in data_list]

        if timestamps:
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            ts_range = max_ts - min_ts

            if ts_range > 0:
                for item in data_list:
                    if item.get('timestamp') is not None:
                        # Normalize to [0, 1] range
                        item['timestamp'] = (item['timestamp'] - min_ts) / ts_range
            else:
                # All timestamps are the same
                for item in data_list:
                    item['timestamp'] = 0.0

        return data_list
