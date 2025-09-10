import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 

class Config:
    data_path: str = 'processed_mental_health_data_fixed.pkl'
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes: int = 3
    text_dim: int = 768
    audio_dim: int = 88
    video_dim: int = 2048
    physio_dim: int = 64  # Physiology input dimension from WESAD

    sequence_length: int = 8
    min_user_posts: int = 3
    min_sequence_length: int = 3
    overlap_ratio: float = 0.25

    hidden_dim: int = 256
    dropout_rate: float = 0.4
    num_heads: int = 8
    num_transformer_layers: int = 2

    num_epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 0.1

    patience: int = 15
    min_delta: float = 0.001

    outlier_threshold: float = 3.0
    missing_threshold: float = 0.1
    variance_threshold: float = 1e-6

    use_data_augmentation: bool = True
    noise_factor: float = 0.01
    dropout_augmentation_rate: float = 0.1

    @classmethod
    def validate(cls) -> None:
        """Validate configuration consistency."""
        assert cls.sequence_length > cls.min_sequence_length, \
            f"sequence_length ({cls.sequence_length}) must be greater than min_sequence_length ({cls.min_sequence_length})"
        assert cls.min_user_posts >= 1, "min_user_posts must be at least 1"
        assert 0 < cls.overlap_ratio < 1, "overlap_ratio must be between 0 and 1"
        assert 0 < cls.dropout_rate < 1, "dropout_rate must be between 0 and 1"
        assert cls.num_epochs > 0, "num_epochs must be positive"
        assert cls.batch_size > 0, "batch_size must be positive"
        assert cls.learning_rate > 0, "learning_rate must be positive"
        assert cls.num_classes > 0, "num_classes must be positive"
        assert hasattr(cls, "physio_dim") and cls.physio_dim > 0, "physio_dim must be set and positive"

        logging.info("✅ Configuration validated successfully")
