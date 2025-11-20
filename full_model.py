"""
Full Latent-WOFA-Seal Model
Combines all components for complete watermark embedding and extraction pipeline.
"""

import torch
import torch.nn as nn
from .pattern_encoder import PatternEncoder
from .pattern_decoder import PatternDecoder
from .latent_embedder import LatentEmbedder
from .latent_extractor import LatentExtractor


class LatentWOFASeal(nn.Module):
    """
    Complete Latent-WOFA-Seal watermarking system.
    
    Components:
        - Pattern Encoder (E_p): QR code -> Latent pattern
        - Pattern Decoder (D_p): Latent pattern -> QR code
        - Latent Embedder (C): Embed pattern into image latent
        - Latent Extractor (DC): Extract pattern from watermarked latent
    
    Training Stages:
        Stage I: Train E_p and D_p with WOFA progressive distortion
        Stage II: Train C and DC with frozen E_p, D_p, and VAE
    """
    def __init__(self, 
                 encoder_config=None,
                 decoder_config=None,
                 embedder_config=None,
                 extractor_config=None):
        super(LatentWOFASeal, self).__init__()
        
        # Default configurations
        encoder_config = encoder_config or {}
        decoder_config = decoder_config or {}
        embedder_config = embedder_config or {}
        extractor_config = extractor_config or {}
        
        # Stage I components
        self.pattern_encoder = PatternEncoder(**encoder_config)
        self.pattern_decoder = PatternDecoder(**decoder_config)
        
        # Stage II components
        self.latent_embedder = LatentEmbedder(**embedder_config)
        self.latent_extractor = LatentExtractor(**extractor_config)
        
    def encode_pattern(self, qr_code):
        """
        Encode QR code to latent pattern (Stage I).
        
        Args:
            qr_code: [B, 1, 256, 256]
        Returns:
            latent_pattern: [B, 1, 64, 64]
        """
        return self.pattern_encoder(qr_code)
    
    def decode_pattern(self, latent_pattern):
        """
        Decode latent pattern to QR code (Stage I).
        
        Args:
            latent_pattern: [B, 1, 64, 64]
        Returns:
            reconstructed_qr: [B, 1, 256, 256]
        """
        return self.pattern_decoder(latent_pattern)
    
    def embed_watermark(self, image_latent, pattern_latent):
        """
        Embed watermark pattern into image latent (Stage II).
        
        Args:
            image_latent: [B, 4, 64, 64]
            pattern_latent: [B, 1, 64, 64]
        Returns:
            watermarked_latent: [B, 4, 64, 64]
        """
        return self.latent_embedder(image_latent, pattern_latent)
    
    def extract_watermark(self, watermarked_latent):
        """
        Extract watermark pattern from watermarked latent (Stage II).
        
        Args:
            watermarked_latent: [B, 4, 64, 64]
        Returns:
            extracted_pattern: [B, 1, 64, 64]
        """
        return self.latent_extractor(watermarked_latent)
    
    def forward_stage1(self, qr_code):
        """
        Stage I forward pass: QR -> Latent -> QR
        
        Args:
            qr_code: [B, 1, 256, 256]
        Returns:
            latent_pattern: [B, 1, 64, 64]
            reconstructed_qr: [B, 1, 256, 256]
        """
        latent_pattern = self.encode_pattern(qr_code)
        reconstructed_qr = self.decode_pattern(latent_pattern)
        return latent_pattern, reconstructed_qr
    
    def forward_stage2(self, image_latent, qr_code):
        """
        Stage II forward pass: Full watermark embedding and extraction
        
        Args:
            image_latent: [B, 4, 64, 64]
            qr_code: [B, 1, 256, 256]
        Returns:
            watermarked_latent: [B, 4, 64, 64]
            extracted_pattern: [B, 1, 64, 64]
            pattern_latent: [B, 1, 64, 64]
        """
        # Encode QR to latent pattern
        with torch.no_grad():  # Encoder is frozen in Stage II
            pattern_latent = self.encode_pattern(qr_code)
        
        # Embed into image latent
        watermarked_latent = self.embed_watermark(image_latent, pattern_latent)
        
        # Extract watermark
        extracted_pattern = self.extract_watermark(watermarked_latent)
        
        return watermarked_latent, extracted_pattern, pattern_latent
    
    def freeze_stage1(self):
        """Freeze Stage I components (E_p and D_p) for Stage II training."""
        for param in self.pattern_encoder.parameters():
            param.requires_grad = False
        for param in self.pattern_decoder.parameters():
            param.requires_grad = False
        print("✓ Stage I components (Encoder & Decoder) frozen.")
    
    def unfreeze_stage1(self):
        """Unfreeze Stage I components."""
        for param in self.pattern_encoder.parameters():
            param.requires_grad = True
        for param in self.pattern_decoder.parameters():
            param.requires_grad = True
        print("✓ Stage I components (Encoder & Decoder) unfrozen.")
    
    def get_stage1_parameters(self):
        """Get parameters for Stage I training."""
        params = list(self.pattern_encoder.parameters()) + \
                 list(self.pattern_decoder.parameters())
        return params
    
    def get_stage2_parameters(self):
        """Get parameters for Stage II training."""
        params = list(self.latent_embedder.parameters()) + \
                 list(self.latent_extractor.parameters())
        return params


# Testing
if __name__ == "__main__":
    print("Testing Full Latent-WOFA-Seal Model...")
    
    # Create model
    model = LatentWOFASeal()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    stage1_params = sum(p.numel() for p in model.get_stage1_parameters())
    stage2_params = sum(p.numel() for p in model.get_stage2_parameters())
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Stage I parameters: {stage1_params:,}")
    print(f"Stage II parameters: {stage2_params:,}")
    
    # Test Stage I
    print("\n--- Testing Stage I ---")
    batch_size = 4
    qr_code = torch.randn(batch_size, 1, 256, 256)
    
    with torch.no_grad():
        latent_pattern, reconstructed_qr = model.forward_stage1(qr_code)
    
    print(f"QR input shape: {qr_code.shape}")
    print(f"Latent pattern shape: {latent_pattern.shape}")
    print(f"Reconstructed QR shape: {reconstructed_qr.shape}")
    
    # Test Stage II
    print("\n--- Testing Stage II ---")
    image_latent = torch.randn(batch_size, 4, 64, 64)
    
    model.freeze_stage1()
    
    with torch.no_grad():
        watermarked_latent, extracted_pattern, pattern_latent = model.forward_stage2(
            image_latent, qr_code
        )
    
    print(f"Image latent shape: {image_latent.shape}")
    print(f"Watermarked latent shape: {watermarked_latent.shape}")
    print(f"Extracted pattern shape: {extracted_pattern.shape}")
    print(f"Pattern latent shape: {pattern_latent.shape}")
    
    print("\n✅ Full model test passed!")