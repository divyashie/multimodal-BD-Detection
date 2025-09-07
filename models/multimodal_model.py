"""
Enhanced Multimodal Model with Improved Architecture
Clean implementation without boolean tensor issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from configs.config import Config

logger = logging.getLogger(__name__)

class ImprovedMultimodalModel(nn.Module):
    """Enhanced multimodal model with improved architecture and stability"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.dropout_rate = getattr(config, 'dropout_rate', 0.5)
        
        # Create enhanced encoders for all modalities including physiology
        self.text_encoder = self._create_enhanced_encoder(config.text_dim, config.hidden_dim, 'text')
        self.audio_encoder = self._create_enhanced_encoder(config.audio_dim, config.hidden_dim, 'audio')
        self.video_encoder = self._create_enhanced_encoder(config.video_dim, config.hidden_dim, 'video')
        self.physio_encoder = self._create_enhanced_encoder(config.physio_dim, config.hidden_dim, 'physio')

        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim, 
            config.num_heads,
            dropout=self.dropout_rate, 
            batch_first=True
        )

        # Updated fusion layer: 5 inputs (text_encoded, attended_text, video_encoded, attended_audio, physio_encoded)
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_dim * 5, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(self.dropout_rate * 1.2)
        )

        self.temporal_encoder = self._create_safe_temporal_encoder(config)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )

        self._initialize_weights()

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Enhanced multimodal model initialized with {total_params:,} trainable parameters")

    def _create_enhanced_encoder(self, input_dim: int, hidden_dim: int, modality: str) -> nn.Module:
        activation = nn.GELU() if modality == 'text' else nn.ReLU()
        encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation,
            nn.Dropout(self.dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation,
            nn.Dropout(self.dropout_rate),

            nn.Linear(hidden_dim, hidden_dim) if modality in ['video', 'text', 'physio'] else nn.Identity(),
            nn.LayerNorm(hidden_dim) if modality in ['video', 'text', 'physio'] else nn.Identity(),
            activation if modality in ['video', 'text', 'physio'] else nn.Identity(),
            nn.Dropout(self.dropout_rate) if modality in ['video', 'text', 'physio'] else nn.Identity(),
        )
        return encoder

    def _create_safe_temporal_encoder(self, config: Config) -> nn.Module:
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 2,
                dropout=self.dropout_rate,
                activation='gelu',
                batch_first=True,
                norm_first=False
            )
            return nn.TransformerEncoder(encoder_layer, num_layers=config.num_transformer_layers)
        except Exception as e:
            logger.warning(f"Failed to create transformer encoder: {e}. Using simple alternative.")
            return nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate)
            )

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'classifier' in name:
                    nn.init.normal_(module.weight, 0, 0.01)
                elif 'attention' in name.lower():
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.MultiheadAttention):
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight)

    def forward(self, text_features: torch.Tensor, 
                audio_features: torch.Tensor, 
                video_features: torch.Tensor,
                physio_features: torch.Tensor) -> torch.Tensor:
        
        batch_size, seq_len, _ = text_features.shape

        text_encoded = self.text_encoder(text_features)
        audio_encoded = self.audio_encoder(audio_features)
        video_encoded = self.video_encoder(video_features)
        physio_encoded = self.physio_encoder(physio_features)

        try:
            attended_text, _ = self.cross_attention(text_encoded, audio_encoded, audio_encoded)
            attended_audio, _ = self.cross_attention(audio_encoded, video_encoded, video_encoded)
        except Exception as e:
            logger.warning(f"Cross-attention failed: {e}. Using identity mapping.")
            attended_text = text_encoded
            attended_audio = audio_encoded

        fused = torch.cat([text_encoded, attended_text, video_encoded, attended_audio, physio_encoded], dim=-1)
        fused_reshaped = fused.view(-1, self.config.hidden_dim * 5)
        fused_features = self.fusion_layer(fused_reshaped)
        fused_features = fused_features.view(batch_size, seq_len, self.config.hidden_dim)

        try:
            temporal_output = self.temporal_encoder(fused_features)
        except Exception as e:
            logger.warning(f"Temporal encoding failed: {e}. Using mean pooling.")
            temporal_output = fused_features

        try:
            attention_scores = torch.mean(temporal_output, dim=-1)
            attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)
            pooled_output = torch.sum(temporal_output * attention_weights, dim=1)
        except Exception as e:
            logger.warning(f"Attention pooling failed: {e}. Using mean pooling.")
            pooled_output = torch.mean(temporal_output, dim=1)

        logits = self.classifier(pooled_output)
        return logits

    def get_attention_weights(self, text_features: torch.Tensor, 
                             audio_features: torch.Tensor, 
                             video_features: torch.Tensor,
                             physio_features: torch.Tensor) -> dict:
        try:
            with torch.no_grad():
                self.eval()
                
                text_encoded = self.text_encoder(text_features)
                audio_encoded = self.audio_encoder(audio_features)
                video_encoded = self.video_encoder(video_features)
                physio_encoded = self.physio_encoder(physio_features)
                
                _, text_attention = self.cross_attention(text_encoded, audio_encoded, audio_encoded)
                _, audio_attention = self.cross_attention(audio_encoded, video_encoded, video_encoded)
                
                return {
                    'text_attention': text_attention.cpu().numpy(),
                    'audio_attention': audio_attention.cpu().numpy()
                }
                
        except Exception as e:
            logger.warning(f"Could not extract attention weights: {e}")
            return {}

    def freeze_encoders(self):
        for encoder in [self.text_encoder, self.audio_encoder, self.video_encoder, self.physio_encoder]:
            for param in encoder.parameters():
                param.requires_grad = False
        logger.info("Encoder parameters frozen")

    def unfreeze_encoders(self):
        for encoder in [self.text_encoder, self.audio_encoder, self.video_encoder, self.physio_encoder]:
            for param in encoder.parameters():
                param.requires_grad = True
        logger.info("Encoder parameters unfrozen")


# Backward compatibility classes
class MultimodalTransformer(ImprovedMultimodalModel):
    pass

class EnhancedMultimodalModel(ImprovedMultimodalModel):
    pass


# Simple fallback model for critical failures
class SimpleMultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        hidden_dim = getattr(config, 'hidden_dim', 256)
        num_classes = getattr(config, 'num_classes', 3)
        dropout_rate = getattr(config, 'dropout_rate', 0.4)

        self.text_encoder = nn.Linear(config.text_dim, hidden_dim)
        self.audio_encoder = nn.Linear(config.audio_dim, hidden_dim)
        self.video_encoder = nn.Linear(config.video_dim, hidden_dim)
        self.physio_encoder = nn.Linear(config.physio_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_features, audio_features, video_features, physio_features):
        text_pooled = torch.mean(self.text_encoder(text_features), dim=1)
        audio_pooled = torch.mean(self.audio_encoder(audio_features), dim=1)
        video_pooled = torch.mean(self.video_encoder(video_features), dim=1)
        physio_pooled = torch.mean(self.physio_encoder(physio_features), dim=1)
        
        fused = torch.cat([text_pooled, audio_pooled, video_pooled, physio_pooled], dim=-1)
        logits = self.fusion(fused)
        return logits


if __name__ == "__main__":
    cfg = Config()
    model = ImprovedMultimodalModel(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: Total={total_params:,}, Trainable={trainable_params:,}")
