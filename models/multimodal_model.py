# ==============================================================================
# --- IMPROVED MODEL ARCHITECTURE ---
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import configs.config as Config

class ImprovedMultimodalModel(nn.Module):
    """Improved model with better regularization and architecture."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.config.dropout_rate = 0.5  # Increased for regularization
        self.text_encoder = self._create_encoder(config.text_dim, config.hidden_dim)
        self.audio_encoder = self._create_encoder(config.audio_dim, config.hidden_dim)
        self.video_encoder = self._create_encoder(config.video_dim, config.hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads,
            dropout=config.dropout_rate, batch_first=True
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),  # Adjusted for 4 inputs
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate * 1.5)  # Extra dropout
        )

        self.temporal_encoder = self._create_temporal_encoder(config)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.num_classes),
            nn.Softmax(dim=-1)  # Ensure proper probability output
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
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=False  # Resolved nested tensor warning
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

        # Bidirectional cross-attention
        attended_text, _ = self.cross_attention(text_encoded, audio_encoded, audio_encoded)
        attended_audio, _ = self.cross_attention(audio_encoded, video_encoded, video_encoded)

        # Fusion with four inputs (text, attended_text, video, attended_audio)
        fused = torch.cat([text_encoded, attended_text, video_encoded, attended_audio], dim=-1)
        fused = self.fusion_layer(fused.view(-1, self.config.hidden_dim * 4))
        fused = fused.view(batch_size, seq_len, self.config.hidden_dim)

        # Temporal modeling
        temporal_output = self.temporal_encoder(fused)
        attention_weights = torch.softmax(torch.mean(temporal_output, dim=-1), dim=-1).unsqueeze(-1)
        pooled_output = torch.sum(temporal_output * attention_weights, dim=1)

        # Classification
        logits = self.classifier(pooled_output)
        return logits