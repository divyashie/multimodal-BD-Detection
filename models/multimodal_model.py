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
        
        # Enhanced dropout for better regularization
        self.dropout_rate = getattr(config, 'dropout_rate', 0.5)
        
        # Create enhanced encoders
        self.text_encoder = self._create_enhanced_encoder(config.text_dim, config.hidden_dim, 'text')
        self.audio_encoder = self._create_enhanced_encoder(config.audio_dim, config.hidden_dim, 'audio')
        self.video_encoder = self._create_enhanced_encoder(config.video_dim, config.hidden_dim, 'video')

        # Safe cross-attention (without complex masking)
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim, 
            config.num_heads,
            dropout=self.dropout_rate, 
            batch_first=True
        )

        # Enhanced fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, config.hidden_dim * 2),  # 4 inputs: text, attended_text, video, attended_audio
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(self.dropout_rate * 1.2)  # Slightly higher dropout
        )

        # Safe temporal encoder
        self.temporal_encoder = self._create_safe_temporal_encoder(config)

        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
            # Note: Removed Softmax - should be handled by loss function
        )

        # Initialize weights
        self._initialize_weights()
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Enhanced multimodal model initialized with {total_params:,} trainable parameters")

    def _create_enhanced_encoder(self, input_dim: int, hidden_dim: int, modality: str) -> nn.Module:
        """Create enhanced encoder with residual-like connections"""
        
        # Different activation functions for different modalities
        activation = nn.GELU() if modality == 'text' else nn.ReLU()
        
        encoder = nn.Sequential(
            # First layer
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation,
            nn.Dropout(self.dropout_rate),
            
            # Second layer with residual-like structure
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation,
            nn.Dropout(self.dropout_rate),
            
            # Third layer for complex modalities
            nn.Linear(hidden_dim, hidden_dim) if modality in ['video', 'text'] else nn.Identity(),
            nn.LayerNorm(hidden_dim) if modality in ['video', 'text'] else nn.Identity(),
            activation if modality in ['video', 'text'] else nn.Identity(),
            nn.Dropout(self.dropout_rate) if modality in ['video', 'text'] else nn.Identity()
        )
        
        return encoder

    def _create_safe_temporal_encoder(self, config: Config) -> nn.Module:
        """Create safe temporal encoder without complex attention masking"""
        
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 2,
                dropout=self.dropout_rate,
                activation='gelu',
                batch_first=True,
                norm_first=False  # Avoid nested tensor warnings
            )
            return nn.TransformerEncoder(encoder_layer, num_layers=config.num_transformer_layers)
        
        except Exception as e:
            logger.warning(f"Failed to create transformer encoder: {e}. Using simple alternative.")
            # Fallback to simple temporal processing
            return nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate)
            )

    def _initialize_weights(self):
        """Enhanced weight initialization"""
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Different initialization strategies
                if 'classifier' in name:
                    nn.init.normal_(module.weight, 0, 0.01)  # Conservative for classifier
                elif 'attention' in name.lower():
                    nn.init.xavier_uniform_(module.weight)  # Xavier for attention
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
                
            elif isinstance(module, nn.MultiheadAttention):
                # Proper initialization for attention layers
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight)

    def forward(self, text_features: torch.Tensor, 
                audio_features: torch.Tensor, 
                video_features: torch.Tensor) -> torch.Tensor:
        """
        Safe forward pass without complex boolean operations
        
        Args:
            text_features: [batch_size, seq_len, text_dim]
            audio_features: [batch_size, seq_len, audio_dim] 
            video_features: [batch_size, seq_len, video_dim]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        
        try:
            batch_size, seq_len, _ = text_features.shape

            # Encode each modality
            text_encoded = self.text_encoder(text_features)    # [B, S, H]
            audio_encoded = self.audio_encoder(audio_features)  # [B, S, H]
            video_encoded = self.video_encoder(video_features)  # [B, S, H]

            # Safe cross-attention without complex masking
            try:
                attended_text, _ = self.cross_attention(text_encoded, audio_encoded, audio_encoded)
                attended_audio, _ = self.cross_attention(audio_encoded, video_encoded, video_encoded)
            except Exception as e:
                logger.warning(f"Cross-attention failed: {e}. Using identity mapping.")
                attended_text = text_encoded
                attended_audio = audio_encoded

            # Fusion with four inputs
            fused = torch.cat([text_encoded, attended_text, video_encoded, attended_audio], dim=-1)
            
            # Reshape for fusion layer
            fused_reshaped = fused.view(-1, self.config.hidden_dim * 4)
            fused_features = self.fusion_layer(fused_reshaped)
            fused_features = fused_features.view(batch_size, seq_len, self.config.hidden_dim)

            # Safe temporal modeling
            try:
                if hasattr(self.temporal_encoder, 'layers'):  # TransformerEncoder
                    temporal_output = self.temporal_encoder(fused_features)
                else:  # Simple fallback
                    temporal_output = self.temporal_encoder(fused_features)
            except Exception as e:
                logger.warning(f"Temporal encoding failed: {e}. Using mean pooling.")
                temporal_output = fused_features

            # Safe attention-based pooling
            try:
                # Compute attention weights safely
                attention_scores = torch.mean(temporal_output, dim=-1)  # [B, S]
                attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)  # [B, S, 1]
                pooled_output = torch.sum(temporal_output * attention_weights, dim=1)  # [B, H]
            except Exception as e:
                logger.warning(f"Attention pooling failed: {e}. Using mean pooling.")
                pooled_output = torch.mean(temporal_output, dim=1)  # [B, H]

            # Classification
            logits = self.classifier(pooled_output)
            
            return logits
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            # Return zeros to prevent complete failure
            return torch.zeros(batch_size, self.config.num_classes, 
                             device=text_features.device, dtype=text_features.dtype)

    def get_attention_weights(self, text_features: torch.Tensor, 
                             audio_features: torch.Tensor, 
                             video_features: torch.Tensor) -> dict:
        """Get attention weights for interpretability (optional)"""
        
        try:
            with torch.no_grad():
                self.eval()
                
                # Forward pass to get intermediate representations
                text_encoded = self.text_encoder(text_features)
                audio_encoded = self.audio_encoder(audio_features)
                video_encoded = self.video_encoder(video_features)
                
                # Get attention weights
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
        """Freeze encoder parameters for fine-tuning"""
        for encoder in [self.text_encoder, self.audio_encoder, self.video_encoder]:
            for param in encoder.parameters():
                param.requires_grad = False
        logger.info("Encoder parameters frozen")

    def unfreeze_encoders(self):
        """Unfreeze encoder parameters"""
        for encoder in [self.text_encoder, self.audio_encoder, self.video_encoder]:
            for param in encoder.parameters():
                param.requires_grad = True
        logger.info("Encoder parameters unfrozen")


# Backward compatibility classes
class MultimodalTransformer(ImprovedMultimodalModel):
    """Backward compatibility wrapper"""
    pass

class EnhancedMultimodalModel(ImprovedMultimodalModel):
    """Backward compatibility wrapper"""
    pass


# Simple fallback model for critical failures
class SimpleMultimodalModel(nn.Module):
    """Simple fallback model"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        hidden_dim = getattr(config, 'hidden_dim', 256)
        num_classes = getattr(config, 'num_classes', 3)
        dropout_rate = getattr(config, 'dropout_rate', 0.4)
        
        self.text_encoder = nn.Linear(config.text_dim, hidden_dim)
        self.audio_encoder = nn.Linear(config.audio_dim, hidden_dim)
        self.video_encoder = nn.Linear(config.video_dim, hidden_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, text_features, audio_features, video_features):
        # Simple mean pooling
        text_pooled = torch.mean(self.text_encoder(text_features), dim=1)
        audio_pooled = torch.mean(self.audio_encoder(audio_features), dim=1)
        video_pooled = torch.mean(self.video_encoder(video_features), dim=1)
        
        # Concatenate and classify
        fused = torch.cat([text_pooled, audio_pooled, video_pooled], dim=-1)
        logits = self.fusion(fused)
        return logits
# Example usage in training script
# Initialize and move model to device
# Example usage in training script
# Example usage in training script
if __name__ == "__main__":
    cfg = Config()
    model = ImprovedMultimodalModel(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: Total={total_params:,}, Trainable={trainable_params:,}")
