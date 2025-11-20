"""
Latent Extractor (DC) - Stage II
Extracts watermark patterns from watermarked image latents.
Uses Small U-Net architecture to separate weak watermark from strong semantic background.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    """
    Downsampling block for U-Net encoder.
    """
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Return both pooled and skip connection
        skip = self.conv(x)
        pooled = self.pool(skip)
        return pooled, skip


class UpBlock(nn.Module):
    """
    Upsampling block for U-Net decoder with skip connections.
    """
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),  # *2 for skip connection
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class LatentExtractor(nn.Module):
    """
    Latent Extractor (DC) for Stage II training.
    
    Architecture: Small U-Net
        Input: [B, 4, 64, 64] - Watermarked (possibly attacked) image latent
        
        Encoder:
            64x64 -> 32x32 -> 16x16 (bottleneck)
        
        Decoder:
            16x16 -> 32x32 -> 64x64
        
        Output: [B, 1, 64, 64] - Extracted watermark pattern latent
    
    Key Features:
        - U-Net with skip connections for multi-scale feature fusion
        - Designed to separate weak watermark signal from strong semantic background
        - Robust to latent-space distortions
    """
    def __init__(self, in_channels=4, base_channels=32):
        super(LatentExtractor, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder (downsampling path)
        self.down1 = DownBlock(base_channels, base_channels * 2)      # 64 -> 32
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)  # 32 -> 16
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.GroupNorm(8, base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.GroupNorm(8, base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling path)
        self.up1 = UpBlock(base_channels * 8, base_channels * 4)  # 16 -> 32
        self.up2 = UpBlock(base_channels * 4, base_channels * 2)  # 32 -> 64
        
        # Final upsampling to original resolution
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output layer: project to 1-channel pattern
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, 3, padding=1),
            nn.Tanh()  # Match pattern encoder output range [-1, 1]
        )
        
    def forward(self, watermarked_latent):
        """
        Args:
            watermarked_latent: [B, 4, 64, 64] - Watermarked image latent (possibly attacked)
        
        Returns:
            extracted_pattern: [B, 1, 64, 64] - Extracted watermark pattern latent
        """
        # Initial features
        x = self.initial_conv(watermarked_latent)  # [B, 32, 64, 64]
        
        # Encoder with skip connections
        x, skip1 = self.down1(x)  # [B, 64, 32, 32], skip: [B, 64, 64, 64]
        x, skip2 = self.down2(x)  # [B, 128, 16, 16], skip: [B, 128, 32, 32]
        
        # Bottleneck
        x = self.bottleneck(x)  # [B, 256, 16, 16]
        
        # Decoder with skip connections
        x = self.up1(x, skip2)  # [B, 128, 32, 32]
        x = self.up2(x, skip1)  # [B, 64, 64, 64]
        
        # Final upsampling (if needed, currently already at 64x64)
        # x = self.final_up(x)  # Not needed since we're already at target size
        
        # Output pattern
        extracted_pattern = self.output_conv(x)  # [B, 1, 64, 64]
        
        return extracted_pattern


# Testing
if __name__ == "__main__":
    print("Testing Latent Extractor...")
    
    # Create model
    extractor = LatentExtractor(in_channels=4, base_channels=32)
    
    # Count parameters
    total_params = sum(p.numel() for p in extractor.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    watermarked_latent = torch.randn(batch_size, 4, 64, 64)
    
    with torch.no_grad():
        extracted_pattern = extractor(watermarked_latent)
    
    print(f"Input shape: {watermarked_latent.shape}")
    print(f"Output shape: {extracted_pattern.shape}")
    print(f"Output range: [{extracted_pattern.min():.3f}, {extracted_pattern.max():.3f}]")
    
    assert extracted_pattern.shape == (batch_size, 1, 64, 64), "Output shape mismatch!"
    print("\n✅ Latent Extractor test passed!")