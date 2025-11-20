"""
Loss Functions for Latent-WOFA-Seal Training

Stage I Losses:
- BCE Loss: Force binary output (critical for QR codes)
- MSE Loss: Pixel-level accuracy
- SSIM Loss: Structural similarity (better than LPIPS for geometric patterns)
- Edge Loss: Optional edge consistency

Stage II Losses:
- Quality Loss (MSE in latent space)
- Recovery Loss (MSE between patterns)
- Perceptual Loss (optional, for image quality)

FIXED: Use BCEWithLogitsLoss to be safe with autocast (AMP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim, SSIM


class BCEWithLogitsLossWrapper(nn.Module):
    """
    Binary Cross Entropy Loss with Logits for QR code reconstruction.
    
    IMPORTANT: This is safe to use with autocast (AMP).
    The decoder should output LOGITS (without Sigmoid).
    
    L_BCE = -[V * log(σ(logits)) + (1-V) * log(1-σ(logits))]
    
    where σ is the sigmoid function.
    """
    def __init__(self, reduction='mean'):
        super(BCEWithLogitsLossWrapper, self).__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(self, logits, target):
        """
        Args:
            logits: [B, 1, H, W] - Raw logits from decoder (NO Sigmoid)
            target: [B, 1, H, W] - Ground truth QR (binary, 0 or 1)
        
        Returns:
            loss: BCE loss
        """
        return self.bce_with_logits(logits, target)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Loss for QR code reconstruction.
    Better than LPIPS for geometric patterns like QR codes.
    
    L_SSIM = (1 - SSIM(V, V̂)) / 2
    
    NOTE: Requires sigmoid of logits for proper comparison.
    """
    def __init__(self, data_range=1.0, channel=1):
        super(SSIMLoss, self).__init__()
        self.data_range = data_range
        self.channel = channel
    
    def forward(self, logits, target):
        """
        Args:
            logits: [B, C, H, W] - Raw logits from decoder
            target: [B, C, H, W] - Ground truth image
        
        Returns:
            loss: 1 - SSIM (lower is better)
        """
        # Apply sigmoid to logits to get predictions in [0, 1]
        pred = torch.sigmoid(logits)
        
        # ssim returns value in [0, 1], where 1 means identical
        ssim_value = ssim(
            pred, target,
            data_range=self.data_range,
            size_average=True,
            nonnegative_ssim=True
        )
        return 1.0 - ssim_value


class EdgeLoss(nn.Module):
    """
    Edge Consistency Loss using Sobel operator.
    Ensures edges in reconstructed QR match original.
    
    Optional: Can be disabled in early training.
    
    NOTE: Requires sigmoid of logits for proper edge detection.
    """
    def __init__(self, reduction='mean'):
        super(EdgeLoss, self).__init__()
        self.reduction = reduction
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def get_edges(self, x):
        """Apply Sobel operator to get edge map."""
        # x: [B, C, H, W]
        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_y = F.conv2d(x, self.sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edge
    
    def forward(self, logits, target):
        """
        Args:
            logits: [B, 1, H, W] - Raw logits from decoder
            target: [B, 1, H, W] - Ground truth QR
        
        Returns:
            loss: Edge consistency loss
        """
        # Apply sigmoid to logits
        pred = torch.sigmoid(logits)
        
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        
        loss = F.mse_loss(pred_edges, target_edges, reduction=self.reduction)
        return loss


class QualityLoss(nn.Module):
    """
    Quality preservation loss for Stage II.
    Ensures watermarked latent is close to original image latent.
    
    L_quality = MSE(z_img, z_w)
    """
    def __init__(self, reduction='mean'):
        super(QualityLoss, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, image_latent, watermarked_latent):
        """
        Args:
            image_latent: [B, 4, 64, 64] - Original image latent
            watermarked_latent: [B, 4, 64, 64] - Watermarked latent
        
        Returns:
            loss: Scalar tensor
        """
        return self.mse(watermarked_latent, image_latent)


class RecoveryLoss(nn.Module):
    """
    Pattern recovery loss for Stage II.
    Ensures extracted pattern matches original pattern latent.
    
    L_recovery = MSE(P, P_hat)
    """
    def __init__(self, reduction='mean'):
        super(RecoveryLoss, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, pattern_latent, extracted_pattern):
        """
        Args:
            pattern_latent: [B, 1, 64, 64] - Original pattern from Encoder
            extracted_pattern: [B, 1, 64, 64] - Extracted pattern from DC
        
        Returns:
            loss: Scalar tensor
        """
        return self.mse(extracted_pattern, pattern_latent)


class DecoderLoss(nn.Module):
    """
    Optional decoder loss for Stage II.
    Verifies that extracted pattern can be decoded back to QR code.
    
    L_decoder = MSE(V, V_hat)
    """
    def __init__(self, reduction='mean'):
        super(DecoderLoss, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, original_qr, reconstructed_qr):
        """
        Args:
            original_qr: [B, 1, 256, 256] - Original QR code
            reconstructed_qr: [B, 1, 256, 256] - Reconstructed QR from extracted pattern
        
        Returns:
            loss: Scalar tensor
        """
        return self.mse(reconstructed_qr, original_qr)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features.
    For Stage II to improve visual quality of watermarked images.
    
    NOT recommended for Stage I (QR codes are not natural images).
    """
    def __init__(self, feature_layers=None, reduction='mean'):
        super(PerceptualLoss, self).__init__()
        
        from torchvision.models import vgg16, VGG16_Weights
        
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        if feature_layers is None:
            feature_layers = [3, 8, 15, 22]
        
        self.feature_layers = feature_layers
        self.feature_extractors = nn.ModuleList()
        
        prev_layer = 0
        for layer_idx in feature_layers:
            self.feature_extractors.append(
                nn.Sequential(*list(vgg.children())[prev_layer:layer_idx+1])
            )
            prev_layer = layer_idx + 1
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.reduction = reduction
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)
        
        loss = 0.0
        for extractor in self.feature_extractors:
            x = extractor(x)
            y = extractor(y)
            layer_loss = F.mse_loss(y, x, reduction=self.reduction)
            loss += layer_loss
        
        return loss


class CombinedStageLoss(nn.Module):
    """
    Combined loss for each training stage with configurable weights.
    
    Stage I: BCEWithLogits + MSE + SSIM (+ Edge)
    Stage II: Quality + Recovery + Decoder (+ Perceptual)
    
    FIXED: Uses BCEWithLogitsLoss which is safe with autocast.
    """
    def __init__(self, stage=1, loss_weights=None):
        """
        Args:
            stage: 1 or 2 (training stage)
            loss_weights: Dict of loss weights
        """
        super(CombinedStageLoss, self).__init__()
        
        self.stage = stage
        
        if stage == 1:
            # Stage I: Multi-objective for QR code reconstruction
            self.bce_loss = BCEWithLogitsLossWrapper()  # FIXED: Use BCEWithLogits
            self.mse_loss = nn.MSELoss()
            self.ssim_loss = SSIMLoss(data_range=1.0, channel=1)
            self.edge_loss = EdgeLoss()
            
            self.weights = loss_weights or {
                'bce': 1.0,      # Main loss: force binary output
                'mse': 0.5,      # Auxiliary: pixel accuracy
                'ssim': 0.1,     # Auxiliary: structural similarity
                'edge': 0.0      # Optional: edge consistency (can enable later)
            }
            
        elif stage == 2:
            # Stage II: Quality + Recovery + optional Decoder
            self.quality_loss = QualityLoss()
            self.recovery_loss = RecoveryLoss()
            self.decoder_loss = DecoderLoss()
            
            self.weights = loss_weights or {
                'quality': 1.0,
                'recovery': 10.0,
                'decoder': 1.0,
                'perceptual': 0.0
            }
            
            if self.weights.get('perceptual', 0.0) > 0:
                self.perceptual_loss = PerceptualLoss()
            else:
                self.perceptual_loss = None
        
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")
    
    def forward_stage1(self, original_qr, decoder_output):
        """
        Stage I loss computation with multi-objective optimization.
        
        Args:
            original_qr: [B, 1, 256, 256] - Ground truth (binary, 0 or 1)
            decoder_output: [B, 1, 256, 256] - Raw LOGITS from decoder (NO Sigmoid)
        
        Returns:
            total_loss: Scalar
            loss_dict: Dict of individual losses
        """
        # 1. BCE Loss (main): Force binary output
        # BCEWithLogitsLoss handles sigmoid internally
        loss_bce = self.bce_loss(decoder_output, original_qr)
        
        # 2. MSE Loss (auxiliary): Pixel-level accuracy
        # Need to apply sigmoid for MSE comparison
        reconstructed_qr = torch.sigmoid(decoder_output)
        loss_mse = self.mse_loss(reconstructed_qr, original_qr)
        
        # 3. SSIM Loss (auxiliary): Structural similarity
        # SSIM loss handles sigmoid internally
        loss_ssim = self.ssim_loss(decoder_output, original_qr)
        
        # 4. Edge Loss (optional): Edge consistency
        # Edge loss handles sigmoid internally
        if self.weights['edge'] > 0:
            loss_edge = self.edge_loss(decoder_output, original_qr)
        else:
            loss_edge = torch.tensor(0.0, device=decoder_output.device)
        
        # Combined loss
        total_loss = (
            self.weights['bce'] * loss_bce +
            self.weights['mse'] * loss_mse +
            self.weights['ssim'] * loss_ssim +
            self.weights['edge'] * loss_edge
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'bce': loss_bce.item(),
            'mse': loss_mse.item(),
            'ssim': loss_ssim.item(),
            'edge': loss_edge.item() if self.weights['edge'] > 0 else 0.0
        }
        
        return total_loss, loss_dict
    
    def forward_stage2(self, 
                      image_latent, 
                      watermarked_latent,
                      pattern_latent,
                      extracted_pattern,
                      original_qr=None,
                      reconstructed_qr=None,
                      original_image=None,
                      watermarked_image=None):
        """Stage II loss computation."""
        loss_quality = self.quality_loss(image_latent, watermarked_latent)
        loss_recovery = self.recovery_loss(pattern_latent, extracted_pattern)
        
        total_loss = (self.weights['quality'] * loss_quality + 
                     self.weights['recovery'] * loss_recovery)
        
        loss_dict = {
            'quality': loss_quality.item(),
            'recovery': loss_recovery.item()
        }
        
        if (self.weights.get('decoder', 0) > 0 and 
            original_qr is not None and 
            reconstructed_qr is not None):
            loss_decoder = self.decoder_loss(original_qr, reconstructed_qr)
            total_loss += self.weights['decoder'] * loss_decoder
            loss_dict['decoder'] = loss_decoder.item()
        
        if (self.perceptual_loss is not None and 
            original_image is not None and 
            watermarked_image is not None):
            loss_percept = self.perceptual_loss(original_image, watermarked_image)
            total_loss += self.weights['perceptual'] * loss_percept
            loss_dict['perceptual'] = loss_percept.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def forward(self, *args, **kwargs):
        """Route to appropriate stage forward function."""
        if self.stage == 1:
            return self.forward_stage1(*args, **kwargs)
        else:
            return self.forward_stage2(*args, **kwargs)


# Testing
if __name__ == "__main__":
    print("Testing Stage I losses with BCEWithLogitsLoss...")
    
    # Create Stage I loss
    stage1_loss = CombinedStageLoss(
        stage=1,
        loss_weights={
            'bce': 1.0,
            'mse': 0.5,
            'ssim': 0.1,
            'edge': 0.05
        }
    )
    
    # Test data
    original_qr = torch.randint(0, 2, (4, 1, 256, 256)).float()  # Binary
    decoder_logits = torch.randn(4, 1, 256, 256)  # Raw logits (NO Sigmoid)
    
    loss, loss_dict = stage1_loss(original_qr, decoder_logits)
    
    print(f"\nTotal loss: {loss.item():.4f}")
    print("Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"  {key:10s}: {value:.4f}")
    
    print("\n✅ Stage I loss test passed (BCEWithLogits is safe with autocast)!")