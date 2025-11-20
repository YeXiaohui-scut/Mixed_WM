"""
Pattern Encoder (E_p) - Stage I
Encodes QR code patterns into latent representations with self-attention.
Critical: Self-attention provides holographic properties for crop resistance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """
    Multi-head Self-Attention module for global feature extraction.
    This is CRITICAL for holographic properties - local features contain global info.
    """
    def __init__(self, channels, num_heads=8):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=False
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Reshape to [H*W, B, C] for attention
        x_flat = x.view(B, C, H * W).permute(2, 0, 1)  # [H*W, B, C]
        
        # Apply self-attention
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        
        # Residual connection + layer norm
        out = self.norm(x_flat + attn_out)
        
        # Reshape back to [B, C, H, W]
        out = out.permute(1, 2, 0).view(B, C, H, W)
        
        return out


class ResBlock(nn.Module):
    """
    Residual Block with optional attention.
    """
    def __init__(self, channels, use_attention=False, num_heads=8):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(channels, num_heads)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.use_attention:
            out = self.attention(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class PatternEncoder(nn.Module):
    """
    Pattern Encoder (E_p) for Stage I training.
    
    Architecture:
        Input: [B, 1, 256, 256] - QR code pattern
        - Downsampling: 256 -> 128 -> 64 -> 64
        - ResBlocks with Self-Attention in middle layers
        - Output: [B, 1, 64, 64] - Latent pattern with Tanh activation
    
    Key Features:
        - Self-Attention for holographic properties (crop resistance)
        - Progressive downsampling to match SD latent space
        - High-frequency information preservation for QR codes
    """
    def __init__(self, in_channels=1, latent_channels=64, num_res_blocks=6, num_heads=8):
        super(PatternEncoder, self).__init__()
        
        # Initial convolution: 1 -> 64 channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, 7, padding=3),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling: 256 -> 128 -> 64
        self.down1 = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels * 2),
            nn.ReLU(inplace=True)
        )  # 256 -> 128
        
        self.down2 = nn.Sequential(
            nn.Conv2d(latent_channels * 2, latent_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels * 4),
            nn.ReLU(inplace=True)
        )  # 128 -> 64
        
        # ResBlocks with Self-Attention
        # Critical: Attention in middle layers for holographic properties
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            # Add attention to middle blocks (indices 2, 3, 4)
            use_attention = (i >= num_res_blocks // 3) and (i < 2 * num_res_blocks // 3 + 1)
            self.res_blocks.append(
                ResBlock(latent_channels * 4, use_attention=use_attention, num_heads=num_heads)
            )
        
        # Final projection to 1 channel latent pattern
        self.final_conv = nn.Sequential(
            nn.Conv2d(latent_channels * 4, latent_channels, 3, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, 1, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, 1, 256, 256] - QR code pattern
        Returns:
            latent_pattern: [B, 1, 64, 64] - Encoded latent pattern
        """
        # Initial feature extraction
        x = self.initial_conv(x)  # [B, 64, 256, 256]
        
        # Downsampling
        x = self.down1(x)  # [B, 128, 128, 128]
        x = self.down2(x)  # [B, 256, 64, 64]
        
        # ResBlocks with Self-Attention
        for res_block in self.res_blocks:
            x = res_block(x)  # [B, 256, 64, 64]
        
        # Final projection
        latent_pattern = self.final_conv(x)  # [B, 1, 64, 64]
        
        return latent_pattern


# Testing
if __name__ == "__main__":
    print("Testing Pattern Encoder...")
    
    # Create model
    encoder = PatternEncoder(in_channels=1, latent_channels=64, num_res_blocks=6)
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    qr_code = torch.randn(batch_size, 1, 256, 256)
    
    with torch.no_grad():
        latent_pattern = encoder(qr_code)
    
    print(f"Input shape: {qr_code.shape}")
    print(f"Output shape: {latent_pattern.shape}")
    print(f"Output range: [{latent_pattern.min():.3f}, {latent_pattern.max():.3f}]")
    
    assert latent_pattern.shape == (batch_size, 1, 64, 64), "Output shape mismatch!"
    print("\n✅ Pattern Encoder test passed!")