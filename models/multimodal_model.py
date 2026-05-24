import torch
from configs.config import Config
import torch.nn as nn
import torch.nn.functional as F

class ImprovedMultimodalModel(nn.Module):
    """Improved model with better regularization and architecture."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Improved encoders with residual connections
        self.text_encoder = self._create_encoder(config.text_dim, config.hidden_dim)
        self.audio_encoder = self._create_encoder(config.audio_dim, config.hidden_dim)
        self.video_encoder = self._create_encoder(config.video_dim, config.hidden_dim)

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads,
            dropout=config.dropout_rate, batch_first=True
        )

        # Fusion layer with residual connection
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )

        # Temporal modeling
        self.temporal_encoder = self._create_temporal_encoder(config)

        # Improved classifier with more regularization
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 4, config.num_classes)
        )

        self._initialize_weights()

    def _create_encoder(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """Create improved encoder with residual connections."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate)
        )

    def _create_temporal_encoder(self, config: Config) -> nn.Module:
        """Create temporal encoder with proper configuration."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 2,  # Reduced from 4
            dropout=config.dropout_rate,
            activation='gelu',  # Better activation for transformers
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=config.num_transformer_layers)

    def _initialize_weights(self):
        """Improved weight initialization."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, text_features: torch.Tensor,
                audio_features: torch.Tensor,
                video_features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = text_features.shape

        # Encode each modality
        text_encoded = self.text_encoder(text_features)
        audio_encoded = self.audio_encoder(audio_features)
        video_encoded = self.video_encoder(video_features)

        # Cross-modal attention (text attending to audio)
        attended_features, _ = self.cross_attention(
            text_encoded, audio_encoded, audio_encoded
        )

        # Fusion with residual connection
        fused = torch.cat([text_encoded, attended_features, video_encoded], dim=-1)
        fused = self.fusion_layer(fused.view(-1, self.config.hidden_dim * 3))
        fused = fused.view(batch_size, seq_len, self.config.hidden_dim)

        # Temporal modeling
        temporal_output = self.temporal_encoder(fused)

        # Global average pooling with attention weights
        attention_weights = torch.softmax(
            torch.mean(temporal_output, dim=-1), dim=-1
        ).unsqueeze(-1)
        pooled_output = torch.sum(temporal_output * attention_weights, dim=1)

        # Classification
        logits = self.classifier(pooled_output)

        return logits