import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Enhanced model and training configuration with validation."""

    # Data configuration
    data_path = 'processed_mental_health_data_fixed.pkl'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model parameters
    num_classes = 3
    text_dim = 768
    audio_dim = 88
    video_dim = 2048

    # Sequence parameters
    sequence_length = 8
    min_user_posts = 3
    min_sequence_length = 3
    overlap_ratio = 0.25

    # Model architecture
    hidden_dim = 256
    dropout_rate = 0.4
    num_heads = 8
    num_transformer_layers = 2

    # Training configuration
    num_epochs = 50
    batch_size = 16
    learning_rate = 5e-5
    weight_decay = 1e-4
    gradient_clip_norm = 0.5
    patience = 15
    min_delta = 0.001

    # Data quality parameters
    outlier_threshold = 3.0
    missing_threshold = 0.1
    variance_threshold = 1e-6

    # Augmentation
    use_data_augmentation = True
    noise_factor = 0.01
    dropout_augmentation_rate = 0.1

    @classmethod
    def validate(cls):
        """Validate configuration parameters."""
        assert cls.sequence_length > cls.min_sequence_length
        assert cls.min_user_posts >= 1
        assert 0 < cls.overlap_ratio < 1
        assert 0 < cls.dropout_rate < 1
        logger.info("Configuration validated successfully")
