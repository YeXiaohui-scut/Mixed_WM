"""
Pattern Decoder (D_p) - Stage I
Decodes latent patterns back to QR codes using U-Net style upsampling.

MODIFIED: Output raw LOGITS (without Sigmoid) for BCEWithLogitsLoss.
This is required for safe autocast (AMP) compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleBlock(nn.Module):
    """
    Upsampling block with residual connection.
    """
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super(UpsampleBlock, self).__init__()
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout2d(0.3)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        
        if self.use_dropout:
            x = self.dropout(x)
        
        return x


class PatternDecoder(nn.Module):
    """
    Pattern Decoder (D_p) for Stage I training.
    
    Architecture:
        Input: [B, 1, 64, 64] - Latent pattern
        - Upsampling: 64 -> 128 -> 256
        - U-Net style with residual connections
        - Output: [B, 1, 256, 256] - Raw LOGITS (NO Sigmoid)
    
    Key Features:
        - Progressive upsampling to restore QR code resolution
        - Residual connections for better gradient flow
        - **Outputs LOGITS for BCEWithLogitsLoss compatibility**
    
    IMPORTANT CHANGE:
        - Removed final Sigmoid activation
        - Now outputs raw logits for use with BCEWithLogitsLoss
        - This is required for autocast (AMP) safety
    """
    def __init__(self, in_channels=1, hidden_channels=64):
        super(PatternDecoder, self).__init__()
        
        # Initial projection
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels * 4, 3, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling blocks: 64 -> 128 -> 256
        self.up1 = UpsampleBlock(hidden_channels * 4, hidden_channels * 2, use_dropout=True)  # 64 -> 128
        self.up2 = UpsampleBlock(hidden_channels * 2, hidden_channels, use_dropout=True)      # 128 -> 256
        
        # Refinement blocks
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer - OUTPUTS LOGITS (NO Sigmoid)
        self.final_conv = nn.Conv2d(hidden_channels // 2, 1, 3, padding=1)
        # NOTE: Removed Sigmoid! Loss function will handle it.
        
    def forward(self, x):
        """
        Args:
            x: [B, 1, 64, 64] - Latent pattern (possibly distorted)
        Returns:
            logits: [B, 1, 256, 256] - Raw logits (NO Sigmoid applied)
        """
        # Initial feature extraction
        x = self.initial_conv(x)  # [B, 256, 64, 64]
        
        # Progressive upsampling
        x = self.up1(x)  # [B, 128, 128, 128]
        x = self.up2(x)  # [B, 64, 256, 256]
        
        # Refinement
        x = self.refine(x)  # [B, 32, 256, 256]
        
        # Final reconstruction - LOGITS
        logits = self.final_conv(x)  # [B, 1, 256, 256]
        
        return logits  # Return raw logits, NOT sigmoid(logits)


# Testing
if __name__ == "__main__":
    print("Testing Pattern Decoder (with logits output)...")
    
    # Create model
    decoder = PatternDecoder(in_channels=1, hidden_channels=64)
    
    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    latent_pattern = torch.randn(batch_size, 1, 64, 64)
    
    with torch.no_grad():
        logits = decoder(latent_pattern)
    
    print(f"Input shape: {latent_pattern.shape}")
    print(f"Output shape (logits): {logits.shape}")
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Test sigmoid
    reconstructed_qr = torch.sigmoid(logits)
    print(f"After sigmoid range: [{reconstructed_qr.min():.3f}, {reconstructed_qr.max():.3f}]")
    
    assert logits.shape == (batch_size, 1, 256, 256), "Output shape mismatch!"
    print("\n✅ Pattern Decoder test passed (now outputs logits)!")