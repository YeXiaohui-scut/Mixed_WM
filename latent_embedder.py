"""
Latent Embedder (C) - Stage II
Embeds latent watermark patterns into image latent space.
Critical: Zero-initialized to ensure no impact at training start.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroInitializedConv(nn.Conv2d):
    """
    Convolution layer with zero initialization.
    Critical for stable training - no impact on image quality initially.
    """
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(ZeroInitializedConv, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        # Zero initialize weights and bias
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class ResidualBlock(nn.Module):
    """
    Residual block for feature transformation.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, channels)  # GroupNorm for better stability
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = self.relu(out)
        
        return out


class LatentEmbedder(nn.Module):
    """
    Latent Embedder (C) for Stage II training.
    
    Architecture:
        Input: 
            - image_latent: [B, 4, 64, 64] - SD VAE encoded image
            - pattern_latent: [B, 1, 64, 64] - Encoded watermark pattern
        
        Process:
            1. Map 1-channel pattern to 4-channel
            2. Residual fusion with image latent
            3. Zero-initialized final layer
        
        Output: [B, 4, 64, 64] - Watermarked latent
    
    Key Features:
        - Zero-initialized final convolution (critical for quality preservation)
        - Residual connection for stable training
        - Compatible with SD latent space (4 channels)
    """
    def __init__(self, image_channels=4, pattern_channels=1, hidden_channels=64):
        super(LatentEmbedder, self).__init__()
        
        # Pattern projection: 1ch -> 64ch
        self.pattern_projection = nn.Sequential(
            nn.Conv2d(pattern_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Conv2d(image_channels + hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels)
        )
        
        # Zero-initialized output layer (CRITICAL)
        # This ensures watermarked_latent ≈ image_latent at training start
        self.output_layer = ZeroInitializedConv(hidden_channels, image_channels, 3, padding=1)
        
    def forward(self, image_latent, pattern_latent):
        """
        Args:
            image_latent: [B, 4, 64, 64] - Original image latent
            pattern_latent: [B, 1, 64, 64] - Watermark pattern latent
        
        Returns:
            watermarked_latent: [B, 4, 64, 64] - Watermarked image latent
        """
        # Project pattern to higher dimension
        pattern_features = self.pattern_projection(pattern_latent)  # [B, 64, 64, 64]
        
        # Concatenate with image latent
        combined = torch.cat([image_latent, pattern_features], dim=1)  # [B, 68, 64, 64]
        
        # Fusion
        fusion_features = self.fusion(combined)  # [B, 64, 64, 64]
        
        # Zero-initialized residual
        residual = self.output_layer(fusion_features)  # [B, 4, 64, 64]
        
        # Add residual to original image latent
        watermarked_latent = image_latent + residual
        
        return watermarked_latent


# Testing
if __name__ == "__main__":
    print("Testing Latent Embedder...")
    
    # Create model
    embedder = LatentEmbedder(image_channels=4, pattern_channels=1, hidden_channels=64)
    
    # Count parameters
    total_params = sum(p.numel() for p in embedder.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    image_latent = torch.randn(batch_size, 4, 64, 64)
    pattern_latent = torch.randn(batch_size, 1, 64, 64)
    
    with torch.no_grad():
        watermarked_latent = embedder(image_latent, pattern_latent)
    
    print(f"Image latent shape: {image_latent.shape}")
    print(f"Pattern latent shape: {pattern_latent.shape}")
    print(f"Watermarked latent shape: {watermarked_latent.shape}")
    
    # Test zero initialization effect
    # At initialization, output should be very close to input
    diff = torch.abs(watermarked_latent - image_latent).mean()
    print(f"Initial difference (should be ~0): {diff:.6f}")
    
    assert watermarked_latent.shape == (batch_size, 4, 64, 64), "Output shape mismatch!"
    assert diff < 0.01, "Zero initialization not working properly!"
    print("\n✅ Latent Embedder test passed!")