"""
Stage I Training (Cached captions variant with WOFA Progressive Strategy)

Modified to use multi-objective loss function:
- BCE Loss: Force binary output (main loss)
- MSE Loss: Pixel-level accuracy (auxiliary)
- SSIM Loss: Structural similarity (auxiliary)
- Edge Loss: Edge consistency (optional)

This implements the correct WOFA progressive distortion strategy:
- Noise: 0.01 → 1.0 in first 20% epochs
- Geometry: 0.05 → 1.0 in first 50% epochs
- Masking: 0.10 → 0.50 → 0.75 → 0.70 in three stages
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import os
import sys
import time
from tqdm import tqdm
import numpy as np
import argparse

from config import Config
# Use cached data provider (reads precomputed captions JSON)
from data_provider_cached import get_dataloader
from models import LatentWOFASeal
from distortions import LatentDistortionLayer, ProgressiveDistortionScheduler

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from losses import CombinedStageLoss


class Stage1Trainer:
    """
    Stage I Trainer: Robustness Pre-training with WOFA
    Now using multi-objective loss: BCE + MSE + SSIM + Edge
    """
    def __init__(self, config):
        self.config = config
        
        # Create timestamp for this experiment
        self.timestamp = config.get_timestamp()
        print(f"Experiment timestamp: {self.timestamp}")
        
        # Setup directories
        self.checkpoint_dir = config.get_stage1_checkpoint_dir(self.timestamp)
        self.log_dir = config.get_stage1_log_dir(self.timestamp)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Log directory: {self.log_dir}")
        
        # Setup device
        self.device = torch.device(config.DEVICE)
        print(f"Using device: {self.device}")
        
        # Build model
        print("Building model...")
        self.model = LatentWOFASeal(
            encoder_config=config.PATTERN_ENCODER_CONFIG,
            decoder_config=config.PATTERN_DECODER_CONFIG,
            embedder_config=config.LATENT_EMBEDDER_CONFIG,
            extractor_config=config.LATENT_EXTRACTOR_CONFIG
        ).to(self.device)
        
        # Only train Stage I components
        parameters = self.model.get_stage1_parameters()
        total_params = sum(p.numel() for p in parameters)
        print(f"Stage I trainable parameters: {total_params:,}")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            parameters,
            lr=config.STAGE1_LEARNING_RATE,
            weight_decay=config.STAGE1_WEIGHT_DECAY,
            betas=config.STAGE1_BETAS
        )
        
        # Setup learning rate scheduler
        self.scheduler = self._build_scheduler()
        
        # Setup multi-objective loss function
        print("\nSetting up multi-objective loss function:")
        loss_weights = {
            'bce': 1.0,      # Main: Force binary output
            'mse': 0.5,      # Auxiliary: Pixel accuracy
            'ssim': 0.1,     # Auxiliary: Structural similarity
            'edge': 0.0      # Optional: Edge consistency (disabled initially)
        }
        print(f"  BCE weight:  {loss_weights['bce']:.2f} (main)")
        print(f"  MSE weight:  {loss_weights['mse']:.2f} (auxiliary)")
        print(f"  SSIM weight: {loss_weights['ssim']:.2f} (auxiliary)")
        print(f"  Edge weight: {loss_weights['edge']:.2f} (optional)")
        
        self.criterion = CombinedStageLoss(stage=1, loss_weights=loss_weights)
        self.criterion.to(self.device)
        
        # Setup distortion layer
        self.distortion = LatentDistortionLayer(
            enable_masking=config.DISTORTION_MASKING_ENABLED,
            enable_geometry=config.DISTORTION_GEOMETRY_ENABLED,
            enable_noise=config.DISTORTION_NOISE_ENABLED,
            masking_max_ratio=config.DISTORTION_MASKING_MAX_RATIO,
            translation_max=config.DISTORTION_TRANSLATION_MAX,
            rotation_max=config.DISTORTION_ROTATION_MAX,
            scale_min=config.DISTORTION_SCALE_MIN,
            scale_max=config.DISTORTION_SCALE_MAX,
            noise_max_std=config.DISTORTION_NOISE_MAX_STD
        ).to(self.device)
        
        # Setup WOFA progressive distortion scheduler
        print("\nSetting up WOFA Progressive Distortion Scheduler:")
        print(f"  - Noise warmup: first {int(config.STAGE1_EPOCHS * config.WOFA_NOISE_WARMUP_RATIO)} epochs ({config.WOFA_NOISE_WARMUP_RATIO:.0%})")
        print(f"  - Geometry warmup: first {int(config.STAGE1_EPOCHS * config.WOFA_GEOMETRY_WARMUP_RATIO)} epochs ({config.WOFA_GEOMETRY_WARMUP_RATIO:.0%})")
        print(f"  - Masking stage 1: first {int(config.STAGE1_EPOCHS * config.WOFA_MASKING_STAGE1_RATIO)} epochs ({config.WOFA_MASKING_STAGE1_RATIO:.0%})")
        print(f"  - Masking stage 2: epochs {int(config.STAGE1_EPOCHS * config.WOFA_MASKING_STAGE1_RATIO)}-{int(config.STAGE1_EPOCHS * config.WOFA_MASKING_STAGE2_RATIO)} ({config.WOFA_MASKING_STAGE2_RATIO:.0%})")
        
        self.distortion_scheduler = ProgressiveDistortionScheduler(
            total_epochs=config.STAGE1_EPOCHS,
            noise_warmup_ratio=config.WOFA_NOISE_WARMUP_RATIO,
            geometry_warmup_ratio=config.WOFA_GEOMETRY_WARMUP_RATIO,
            masking_stage1_ratio=config.WOFA_MASKING_STAGE1_RATIO,
            masking_stage2_ratio=config.WOFA_MASKING_STAGE2_RATIO,
            schedule_type=config.WOFA_SCHEDULE_TYPE
        )
        
        # Setup AMP
        self.use_amp = config.USE_AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup TensorBoard
        if config.TENSORBOARD_ENABLED:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        if self.config.STAGE1_LR_SCHEDULER == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.STAGE1_EPOCHS,
                eta_min=self.config.STAGE1_LR_MIN
            )
        elif self.config.STAGE1_LR_SCHEDULER == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.STAGE1_EPOCHS // 3,
                gamma=0.1
            )
        else:
            return None
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with WOFA progressive distortion."""
        self.model.train()
        
        # Get current epoch's distortion strength (WOFA strategy)
        distortion_strength = self.distortion_scheduler.get_strength(epoch)
        
        # Print distortion strengths at the beginning of each epoch
        print(f"\nEpoch {epoch+1}/{self.config.STAGE1_EPOCHS} Distortion Strengths:")
        print(f"  Noise:    {distortion_strength['noise']:.4f}")
        print(f"  Geometry: {distortion_strength['geometry']:.4f}")
        print(f"  Masking:  {distortion_strength['masking']:.4f}")
        
        # Accumulators for epoch metrics
        epoch_loss_dict = {
            'total': 0.0,
            'bce': 0.0,
            'mse': 0.0,
            'ssim': 0.0,
            'edge': 0.0
        }
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.STAGE1_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get QR codes
            qr_codes = batch['watermark'].to(self.device)  # [B, 1, 256, 256]
            
            # # Forward pass with AMP: do model forward in autocast but compute
            # # BCE-based losses outside autocast to avoid unsafe autocast on BCE.
            # with autocast(enabled=self.use_amp):
            #     # Encode QR to latent pattern
            #     latent_pattern = self.model.encode_pattern(qr_codes)  # [B, 1, 64, 64]

            #     # Apply WOFA progressive distortion (dict input with separate strengths)
            #     distorted_pattern = self.distortion(latent_pattern, distortion_strength)

            #     # Decode back to QR code
            #     reconstructed_qr = self.model.decode_pattern(distorted_pattern)  # [B, 1, 256, 256]

            # # Ensure reconstruction is float32 before computing BCE/MSE/SSIM (loss functions expect float32)
            # if reconstructed_qr.dtype != torch.float32:
            #     reconstructed_qr = reconstructed_qr.float()

            # # Compute multi-objective loss outside autocast to keep BCE computation in float32
            # loss, loss_dict = self.criterion(qr_codes, reconstructed_qr)
                        # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                # Encode QR to latent pattern
                latent_pattern = self.model.encode_pattern(qr_codes)  # [B, 1, 64, 64]
                
                # Apply WOFA progressive distortion (dict input with separate strengths)
                distorted_pattern = self.distortion(latent_pattern, distortion_strength)
                
                # Decode back to QR code
                reconstructed_qr = self.model.decode_pattern(distorted_pattern)  # [B, 1, 256, 256]
                
                # Compute multi-objective loss
                loss, loss_dict = self.criterion(qr_codes, reconstructed_qr)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.config.GRAD_CLIP_NORM > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_stage1_parameters(),
                        self.config.GRAD_CLIP_NORM
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_stage1_parameters(),
                        self.config.GRAD_CLIP_NORM
                    )
                self.optimizer.step()
            
            # Update metrics
            for key in epoch_loss_dict.keys():
                if key in loss_dict:
                    epoch_loss_dict[key] += loss_dict[key]
            
            self.global_step += 1
            
            # Update progress bar with all loss components
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'BCE': f'{loss_dict["bce"]:.3f}',
                'MSE': f'{loss_dict["mse"]:.3f}',
                'SSIM': f'{loss_dict["ssim"]:.3f}',
                'N': f'{distortion_strength["noise"]:.2f}',
                'G': f'{distortion_strength["geometry"]:.2f}',
                'M': f'{distortion_strength["masking"]:.2f}'
            })
            
            # TensorBoard logging
            if self.writer and self.global_step % self.config.TENSORBOARD_LOG_INTERVAL == 0:
                # Log all loss components
                self.writer.add_scalar('Stage1/Loss/total', loss_dict['total'], self.global_step)
                self.writer.add_scalar('Stage1/Loss/bce', loss_dict['bce'], self.global_step)
                self.writer.add_scalar('Stage1/Loss/mse', loss_dict['mse'], self.global_step)
                self.writer.add_scalar('Stage1/Loss/ssim', loss_dict['ssim'], self.global_step)
                if loss_dict['edge'] > 0:
                    self.writer.add_scalar('Stage1/Loss/edge', loss_dict['edge'], self.global_step)
                
                # Log distortion strengths
                self.writer.add_scalar('Stage1/Distortion/Noise', 
                                      distortion_strength['noise'], self.global_step)
                self.writer.add_scalar('Stage1/Distortion/Geometry', 
                                      distortion_strength['geometry'], self.global_step)
                self.writer.add_scalar('Stage1/Distortion/Masking', 
                                      distortion_strength['masking'], self.global_step)
                
                # Log learning rate
                self.writer.add_scalar('Stage1/Learning_Rate', 
                                      self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Log images
            if self.writer and self.global_step % self.config.TENSORBOARD_IMAGE_INTERVAL == 0:
                self._log_images(qr_codes, latent_pattern, distorted_pattern, reconstructed_qr)
            
            # Debug mode: limit iterations
            if self.config.DEBUG_MODE and batch_idx >= self.config.DEBUG_ITERATIONS:
                break
        
        # Average epoch losses
        num_batches = len(dataloader) if not self.config.DEBUG_MODE else self.config.DEBUG_ITERATIONS
        for key in epoch_loss_dict.keys():
            epoch_loss_dict[key] /= num_batches
        
        return epoch_loss_dict
    
    def _log_images(self, qr_codes, latent_pattern, distorted_pattern, reconstructed_qr):
        """Log visualization to TensorBoard."""
        with torch.no_grad():
            # Select first few samples from batch
            n = min(self.config.VIS_BATCH_SIZE, qr_codes.size(0))
            
            # Original QR codes
            self.writer.add_images('Stage1/1_Original_QR', 
                                  qr_codes[:n], self.global_step, dataformats='NCHW')
            
            # Latent patterns (normalize for visualization)
            latent_vis = (latent_pattern[:n] + 1) / 2  # [-1,1] -> [0,1]
            self.writer.add_images('Stage1/2_Latent_Pattern', 
                                  latent_vis, self.global_step, dataformats='NCHW')
            
            # Distorted latent patterns
            distorted_vis = (distorted_pattern[:n] + 1) / 2
            self.writer.add_images('Stage1/3_Distorted_Pattern', 
                                  distorted_vis, self.global_step, dataformats='NCHW')
            
            # Reconstructed QR codes
            self.writer.add_images('Stage1/4_Reconstructed_QR', 
                                  reconstructed_qr[:n], self.global_step, dataformats='NCHW')
            
            # Difference map
            diff = torch.abs(qr_codes[:n] - reconstructed_qr[:n])
            self.writer.add_images('Stage1/5_Difference_Map', 
                                  diff, self.global_step, dataformats='NCHW')
            
            # Binarized reconstruction (threshold at 0.5)
            binarized = (reconstructed_qr[:n] > 0.5).float()
            self.writer.add_images('Stage1/6_Binarized_QR',
                                  binarized, self.global_step, dataformats='NCHW')
    
    def save_checkpoint(self, epoch, metrics=None, is_best=False):
        """Save model checkpoint."""
        # Make a plain dict snapshot of the Config class
        try:
            config_snapshot = {k: v for k, v in self.config.__dict__.items() if k.isupper()}
        except Exception:
            config_snapshot = {}

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': config_snapshot,
            'timestamp': self.timestamp
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch:04d}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
        
        # Save latest model (for easy resuming)
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
    
    def train(self, dataloader):
        """Main training loop."""
        print("\n" + "="*80)
        print("Starting Stage I Training: WOFA Progressive Pre-training")
        print("="*80)
        print("\nKey Strategy:")
        print("  ✓ Multi-objective loss: BCE + MSE + SSIM + Edge")
        print("  ✓ Start with SMALL but NON-ZERO distortions from Epoch 0")
        print("  ✓ Progressively increase distortion strength over epochs")
        print("  ✓ Different distortion types have different warmup schedules")
        print("="*80)
        
        best_loss = float('inf')
        
        for epoch in range(self.config.STAGE1_EPOCHS):
            self.current_epoch = epoch
            
            # Train one epoch
            epoch_start = time.time()
            loss_dict = self.train_epoch(dataloader, epoch)
            epoch_time = time.time() - epoch_start
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log epoch metrics
            print(f"\nEpoch {epoch+1}/{self.config.STAGE1_EPOCHS} Summary:")
            print(f"  Total Loss: {loss_dict['total']:.4f}")
            print(f"  BCE Loss:   {loss_dict['bce']:.4f}")
            print(f"  MSE Loss:   {loss_dict['mse']:.4f}")
            print(f"  SSIM Loss:  {loss_dict['ssim']:.4f}")
            if loss_dict['edge'] > 0:
                print(f"  Edge Loss:  {loss_dict['edge']:.4f}")
            print(f"  Time:       {epoch_time:.2f}s")
            
            if self.writer:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'Stage1/Loss_Epoch/{key}', value, epoch)
            
            # Save checkpoint
            is_best = loss_dict['total'] < best_loss
            if is_best:
                best_loss = loss_dict['total']
                print(f"  ✓ New best loss: {best_loss:.4f}")
            
            if (epoch + 1) % self.config.STAGE1_SAVE_INTERVAL == 0:
                self.save_checkpoint(epoch, metrics=loss_dict, is_best=is_best)
        
        # Save final model
        self.save_checkpoint(self.config.STAGE1_EPOCHS - 1, metrics=loss_dict, is_best=False)
        
        print("\n" + "="*80)
        print("Stage I Training Completed!")
        print(f"Best Loss: {best_loss:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"TensorBoard logs: {self.log_dir}")
        print("="*80)
        
        if self.writer:
            self.writer.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Stage1 training with WOFA progressive strategy and multi-objective loss')
    parser.add_argument('--captions_json', default=None,
                        help='Path to precomputed captions JSON (overrides default)')
    args = parser.parse_args()

    # Print configuration
    Config.print_config()
    
    # Set random seed
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # Determine captions JSON path
    if args.captions_json:
        captions_json = args.captions_json
    else:
        # Default location inside the project
        captions_json = os.path.join(Config.BASE_DIR, 'data_json', 'captions.json')

    print("\nLoading cached dataset (precomputed captions)...")
    print(f"Captions JSON: {captions_json}")
    dataloader = get_dataloader(
        captions_json,
        batch_size=Config.STAGE1_BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        shuffle=True
    )
    print(f"Dataset loaded: {len(dataloader)} batches")
    
    # Create trainer
    trainer = Stage1Trainer(Config)
    
    # Start training
    trainer.train(dataloader)


if __name__ == "__main__":
    main()