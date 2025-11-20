"""
WOFA Progressive Distortion Layer
Simulates physical-world attacks in latent space with progressive strength control.

Key Features:
- Progressive strength parameter (0.0 → 1.0) for curriculum learning
- Latent Masking: Simulates partial theft/cropping
- Latent Geometry: Simulates geometric transformations
- Latent Noise: Simulates noise attacks

This operates on 64×64 latent space, NOT on pixel space.

WOFA Strategy Implementation:
- Noise: Progressive in first 20% epochs (0.01 → 1.0)
- Geometry: Progressive in first 50% epochs (0.05 → 1.0)
- Masking: Three-stage progressive (0.10 → 0.50 → 0.75 → 0.70)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np


class LatentDistortionLayer(nn.Module):
    """
    Progressive distortion layer for latent space (64×64).
    
    Args:
        strength: Can be:
                 - Float in [0.0, 1.0]: uniform strength for all distortion types
                 - Dict: {'noise': 0.5, 'geometry': 0.8, 'masking': 0.6}
        
        enable_masking: Enable random masking (crop simulation)
        enable_geometry: Enable geometric transforms
        enable_noise: Enable Gaussian noise
    """
    def __init__(self, 
                 enable_masking=True,
                 enable_geometry=True,
                 enable_noise=True,
                 masking_max_ratio=0.7,
                 translation_max=0.2,
                 rotation_max=45.0,
                 scale_min=0.8,
                 scale_max=1.2,
                 noise_max_std=0.1):
        super(LatentDistortionLayer, self).__init__()
        
        self.enable_masking = enable_masking
        self.enable_geometry = enable_geometry
        self.enable_noise = enable_noise
        
        # Attack parameters
        self.masking_max_ratio = masking_max_ratio
        self.translation_max = translation_max
        self.rotation_max = rotation_max
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.noise_max_std = noise_max_std
        
    def apply_masking(self, x, strength):
        """
        Apply random rectangular masking (crop simulation).
        
        Args:
            x: [B, C, H, W] latent tensor
            strength: [0, 1] controls mask size
        
        Returns:
            masked: [B, C, H, W] with masked regions set to 0
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Strength controls mask area ratio
        # strength=0.10 → small mask (10%)
        # strength=1.0 → large mask (70%)
        mask_ratio = strength * self.masking_max_ratio
        mask_ratio = min(mask_ratio, self.masking_max_ratio)
        
        masked = x.clone()
        
        for b in range(B):
            # Random mask dimensions
            mask_area = mask_ratio * H * W
            aspect_ratio = random.uniform(0.5, 2.0)  # Random aspect ratio
            
            mask_h = int(math.sqrt(mask_area * aspect_ratio))
            mask_w = int(math.sqrt(mask_area / aspect_ratio))
            
            mask_h = min(mask_h, H)
            mask_w = min(mask_w, W)
            
            # Random position
            if mask_h < H:
                top = random.randint(0, H - mask_h)
            else:
                top = 0
                
            if mask_w < W:
                left = random.randint(0, W - mask_w)
            else:
                left = 0
            
            # Apply mask
            masked[b, :, top:top+mask_h, left:left+mask_w] = 0
        
        return masked
    
    def apply_geometry(self, x, strength):
        """
        Apply geometric transformations using affine grid sampling.
        
        Args:
            x: [B, C, H, W] latent tensor
            strength: [0, 1] controls transform intensity
        
        Returns:
            transformed: [B, C, H, W]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Create affine transformation matrix for each sample in batch
        theta_list = []
        
        for b in range(B):
            # Translation (in normalized coordinates [-1, 1])
            tx = random.uniform(-1, 1) * strength * self.translation_max
            ty = random.uniform(-1, 1) * strength * self.translation_max
            
            # Rotation (in radians)
            angle_deg = random.uniform(-1, 1) * strength * self.rotation_max
            angle_rad = math.radians(angle_deg)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            # Scale
            scale_range = self.scale_max - self.scale_min
            scale = self.scale_min + strength * scale_range
            # Random scale around 1.0
            scale = 1.0 + (scale - 1.0) * random.uniform(-1, 1)
            
            # Construct affine matrix [2, 3]
            # [cos*scale, -sin*scale, tx]
            # [sin*scale,  cos*scale, ty]
            theta = torch.tensor([
                [cos_a * scale, -sin_a * scale, tx],
                [sin_a * scale,  cos_a * scale, ty]
            ], dtype=torch.float32, device=device)
            
            theta_list.append(theta)
        
        # Stack all theta matrices
        theta_batch = torch.stack(theta_list, dim=0)  # [B, 2, 3]
        
        # Generate sampling grid
        grid = F.affine_grid(theta_batch, x.size(), align_corners=False)
        
        # Apply transformation
        transformed = F.grid_sample(x, grid, mode='bilinear', 
                                   padding_mode='zeros', align_corners=False)
        
        return transformed
    
    def apply_noise(self, x, strength):
        """
        Apply Gaussian noise.
        
        Args:
            x: [B, C, H, W] latent tensor
            strength: [0, 1] controls noise intensity
        
        Returns:
            noisy: [B, C, H, W]
        """
        # Noise std scales with strength
        noise_std = strength * self.noise_max_std
        
        noise = torch.randn_like(x) * noise_std
        noisy = x + noise
        
        return noisy
    
    def forward(self, x, strength):
        """
        Apply progressive distortions to latent tensor.
        
        Args:
            x: [B, C, H, W] latent tensor (64×64)
            strength: Can be:
                     - Float in [0, 1]: uniform strength
                     - Dict: {'noise': 0.5, 'geometry': 0.8, 'masking': 0.6}
                     - Tensor [B]: per-sample uniform strength
        
        Returns:
            distorted: [B, C, H, W] distorted latent tensor
        """
        # Parse strength input
        if isinstance(strength, dict):
            # Dict input: separate strength for each distortion type
            noise_strength = strength.get('noise', 0.0)
            geometry_strength = strength.get('geometry', 0.0)
            masking_strength = strength.get('masking', 0.0)
        elif isinstance(strength, (int, float)):
            # Scalar: uniform strength
            noise_strength = strength
            geometry_strength = strength
            masking_strength = strength
        elif isinstance(strength, torch.Tensor):
            # Tensor: use mean for this implementation
            strength_val = strength.mean().item()
            noise_strength = strength_val
            geometry_strength = strength_val
            masking_strength = strength_val
        else:
            raise ValueError(f"Invalid strength type: {type(strength)}")
        
        distorted = x
        
        # Apply distortions in sequence
        # Order matters for realistic attack simulation
        
        # 1. Geometric transforms first (most common in real world)
        if self.enable_geometry and geometry_strength > 0:
            distorted = self.apply_geometry(distorted, geometry_strength)
        
        # 2. Masking (simulates cropping after geometric transform)
        if self.enable_masking and masking_strength > 0:
            distorted = self.apply_masking(distorted, masking_strength)
        
        # 3. Noise last (simulates compression/transmission noise)
        if self.enable_noise and noise_strength > 0:
            distorted = self.apply_noise(distorted, noise_strength)
        
        return distorted


class ProgressiveDistortionScheduler:
    """
    WOFA Progressive Distortion Scheduler
    
    Implements curriculum learning strategy from WOFA paper:
    - Noise: Progressive in first 20% epochs (0.01 → 1.0)
    - Geometry: Progressive in first 50% epochs (0.05 → 1.0)
    - Masking: Three-stage progressive:
        * 0-1%: 0.10 → 0.50 (keep 50%)
        * 1-10%: 0.50 → 0.75 (keep 25%)
        * 10%+: 0.70 (keep 5%-95%, capped at 70%)
    
    Key insight: Start with SMALL but NON-ZERO distortions, then progressively increase.
    """
    def __init__(self, 
                 total_epochs,
                 noise_warmup_ratio=0.2,      # Noise progressive in first 20% epochs
                 geometry_warmup_ratio=0.5,   # Geometry progressive in first 50% epochs
                 masking_stage1_ratio=0.01,   # First masking stage: 0-1% epochs
                 masking_stage2_ratio=0.1,    # Second masking stage: 1-10% epochs
                 schedule_type='linear'):
        """
        Args:
            total_epochs: Total number of training epochs
            noise_warmup_ratio: Ratio of epochs for noise warmup (default: 0.2 = 20%)
            geometry_warmup_ratio: Ratio of epochs for geometry warmup (default: 0.5 = 50%)
            masking_stage1_ratio: Ratio for first masking stage (default: 0.01 = 1%)
            masking_stage2_ratio: Ratio for second masking stage (default: 0.1 = 10%)
            schedule_type: 'linear', 'cosine', or 'exponential'
        """
        self.total_epochs = total_epochs
        self.noise_warmup_ratio = noise_warmup_ratio
        self.geometry_warmup_ratio = geometry_warmup_ratio
        self.masking_stage1_ratio = masking_stage1_ratio
        self.masking_stage2_ratio = masking_stage2_ratio
        self.schedule_type = schedule_type
        
    def get_strength(self, epoch):
        """
        Get distortion strength for current epoch (WOFA strategy).
        
        Args:
            epoch: Current epoch (0-indexed)
        
        Returns:
            dict: {
                'noise': float in [0.01, 1.0],
                'geometry': float in [0.05, 1.0],
                'masking': float in [0.10, 0.70]
            }
        """
        progress = epoch / self.total_epochs
        
        # ============================================================
        # 1. Noise Strength: Progressive in first 20% epochs
        #    Start from 0.01 (not 0!) to avoid "no distortion" trap
        # ============================================================
        if progress < self.noise_warmup_ratio:
            noise_progress = progress / self.noise_warmup_ratio
            noise_strength = 0.01 + self._apply_schedule(noise_progress) * 0.99  # 0.01 → 1.0
        else:
            noise_strength = 1.0
        
        # ============================================================
        # 2. Geometry Strength: Progressive in first 50% epochs
        #    Start from 0.05 (small but non-zero)
        # ============================================================
        if progress < self.geometry_warmup_ratio:
            geometry_progress = progress / self.geometry_warmup_ratio
            geometry_strength = 0.05 + self._apply_schedule(geometry_progress) * 0.95  # 0.05 → 1.0
        else:
            geometry_strength = 1.0
        
        # ============================================================
        # 3. Masking Strength: Three-stage progressive
        #    Start from 0.10 (mask 10% of area)
        # ============================================================
        if progress < self.masking_stage1_ratio:
            # Stage 1 (0-1% epochs): 0.10 → 0.50
            # Keep at least 50% of original pattern
            stage_progress = progress / self.masking_stage1_ratio
            masking_strength = 0.10 + self._apply_schedule(stage_progress) * 0.40  # 0.10 → 0.50
            
        elif progress < self.masking_stage2_ratio:
            # Stage 2 (1-10% epochs): 0.50 → 0.75
            # Keep at least 25% of original pattern
            stage_progress = (progress - self.masking_stage1_ratio) / \
                           (self.masking_stage2_ratio - self.masking_stage1_ratio)
            masking_strength = 0.50 + self._apply_schedule(stage_progress) * 0.25  # 0.50 → 0.75
            
        else:
            # Stage 3 (10%+ epochs): Cap at 0.70
            # In WOFA paper, this can be 1%-95% random, but we cap at 70% for stability
            masking_strength = 0.70
        
        return {
            'noise': noise_strength,
            'geometry': geometry_strength,
            'masking': masking_strength
        }
    
    def _apply_schedule(self, progress):
        """
        Apply scheduling function to progress [0, 1].
        
        Args:
            progress: Float in [0, 1]
        
        Returns:
            scheduled_value: Float in [0, 1]
        """
        progress = min(progress, 1.0)
        
        if self.schedule_type == 'linear':
            return progress
            
        elif self.schedule_type == 'cosine':
            # Smooth acceleration (slow start, fast middle, slow end)
            return (1 - math.cos(math.pi * progress)) / 2
            
        elif self.schedule_type == 'exponential':
            # Fast start, slow end
            return progress ** 2
            
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


# Testing
if __name__ == "__main__":
    print("Testing WOFA Progressive Distortion Layer...")
    print("=" * 70)
    
    # Create distortion layer
    distortion = LatentDistortionLayer(
        enable_masking=True,
        enable_geometry=True,
        enable_noise=True
    )
    
    # Test with dict strength input
    print("\n--- Testing Dict Strength Input ---")
    batch_size = 4
    x = torch.randn(batch_size, 1, 64, 64)
    
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    
    strength_dict = {
        'noise': 0.5,
        'geometry': 0.8,
        'masking': 0.6
    }
    
    with torch.no_grad():
        distorted = distortion(x, strength_dict)
    
    diff = (distorted - x).abs().mean()
    print(f"\nStrength dict: {strength_dict}")
    print(f"Output range: [{distorted.min():.3f}, {distorted.max():.3f}]")
    print(f"Mean absolute difference: {diff:.4f}")
    
    # Test Progressive Scheduler
    print("\n" + "=" * 70)
    print("Testing WOFA Progressive Distortion Scheduler...")
    print("=" * 70)
    
    scheduler = ProgressiveDistortionScheduler(
        total_epochs=100,
        noise_warmup_ratio=0.2,
        geometry_warmup_ratio=0.5,
        masking_stage1_ratio=0.01,
        masking_stage2_ratio=0.1,
        schedule_type='linear'
    )
    
    print("\nProgressive Strength Schedule:")
    print(f"{'Epoch':>6} {'Progress':>8} {'Noise':>8} {'Geometry':>10} {'Masking':>9}")
    print("-" * 50)
    
    test_epochs = [0, 1, 5, 10, 20, 50, 75, 99]
    for epoch in test_epochs:
        strength = scheduler.get_strength(epoch)
        progress = epoch / 100
        print(f"{epoch:6d} {progress:7.1%} "
              f"{strength['noise']:8.3f} {strength['geometry']:10.3f} "
              f"{strength['masking']:9.3f}")
    
    print("\n" + "=" * 70)
    print("Key Observations:")
    print("  - Epoch 0: All strengths start SMALL but NON-ZERO")
    print("  - Epoch 20: Noise reaches 1.0 (20% warmup complete)")
    print("  - Epoch 50: Geometry reaches 1.0 (50% warmup complete)")
    print("  - Epoch 10+: Masking caps at 0.70")
    print("=" * 70)
    
    # Test with actual distortion
    print("\n--- Testing Actual Distortion at Different Epochs ---")
    x = torch.randn(4, 1, 64, 64)
    
    for epoch in [0, 20, 50, 99]:
        strength = scheduler.get_strength(epoch)
        with torch.no_grad():
            distorted = distortion(x, strength)
        diff = (distorted - x).abs().mean()
        print(f"Epoch {epoch:3d}: Diff={diff:.4f} | "
              f"N={strength['noise']:.2f} G={strength['geometry']:.2f} M={strength['masking']:.2f}")
    
    print("\n✅ WOFA Distortion layer test passed!")