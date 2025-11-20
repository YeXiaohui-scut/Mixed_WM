"""
Stage I Training: Robustness Pre-training with WOFA Strategy

Training Flow:
1. QR Code (V) → Pattern Encoder (E_p) → Latent Pattern (P)
2. Apply Progressive Distortion: P → P' (strength increases over epochs)
3. P' → Pattern Decoder (D_p) → Reconstructed QR (V_hat)
4. Loss: MSE(V, V_hat)

Goal: Train E_p and D_p to be robust against latent-space distortions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import os
import time
from tqdm import tqdm
import numpy as np

from config import Config
from data_provider import get_dataloader
from models import LatentWOFASeal
from distortions import LatentDistortionLayer, ProgressiveDistortionScheduler


class Stage1Trainer:
    """
    Stage I Trainer: Robustness Pre-training with WOFA
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
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
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
        
        # Setup progressive distortion scheduler
        self.distortion_scheduler = ProgressiveDistortionScheduler(
            total_epochs=config.STAGE1_EPOCHS,
            start_strength=config.STAGE1_DISTORTION_START_STRENGTH,
            end_strength=config.STAGE1_DISTORTION_END_STRENGTH,
            warmup_epochs=config.STAGE1_LR_WARMUP_EPOCHS,
            schedule_type='cosine'
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
        """Train for one epoch."""
        self.model.train()
        
        # Get current distortion strength
        distortion_strength = self.distortion_scheduler.get_strength(epoch)
        
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.STAGE1_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get QR codes
            qr_codes = batch['watermark'].to(self.device)  # [B, 1, 256, 256]
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                # Encode QR to latent pattern
                latent_pattern = self.model.encode_pattern(qr_codes)  # [B, 1, 64, 64]
                
                # Apply progressive distortion
                distorted_pattern = self.distortion(latent_pattern, distortion_strength)
                
                # Decode back to QR code
                reconstructed_qr = self.model.decode_pattern(distorted_pattern)  # [B, 1, 256, 256]
                
                # Compute loss
                loss = self.criterion(reconstructed_qr, qr_codes)
            
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
            epoch_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dist_strength': f'{distortion_strength:.3f}'
            })
            
            # TensorBoard logging
            if self.writer and self.global_step % self.config.TENSORBOARD_LOG_INTERVAL == 0:
                self.writer.add_scalar('Stage1/Loss/train', loss.item(), self.global_step)
                self.writer.add_scalar('Stage1/Distortion_Strength', distortion_strength, self.global_step)
                self.writer.add_scalar('Stage1/Learning_Rate', 
                                      self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Log images
            if self.writer and self.global_step % self.config.TENSORBOARD_IMAGE_INTERVAL == 0:
                self._log_images(qr_codes, latent_pattern, distorted_pattern, reconstructed_qr)
            
            # Debug mode: limit iterations
            if self.config.DEBUG_MODE and batch_idx >= self.config.DEBUG_ITERATIONS:
                break
        
        avg_loss = epoch_loss / len(dataloader)
        return avg_loss
    
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
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config.__dict__,
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
        print("Starting Stage I Training: Robustness Pre-training with WOFA")
        print("="*80)
        
        best_loss = float('inf')
        
        for epoch in range(self.config.STAGE1_EPOCHS):
            self.current_epoch = epoch
            
            # Train one epoch
            epoch_start = time.time()
            avg_loss = self.train_epoch(dataloader, epoch)
            epoch_time = time.time() - epoch_start
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log epoch metrics
            print(f"\nEpoch {epoch+1}/{self.config.STAGE1_EPOCHS} - "
                  f"Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")
            
            if self.writer:
                self.writer.add_scalar('Stage1/Loss/epoch', avg_loss, epoch)
            
            # Save checkpoint
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                print(f"New best loss: {best_loss:.4f}")
            
            if (epoch + 1) % self.config.STAGE1_SAVE_INTERVAL == 0:
                self.save_checkpoint(epoch, is_best=is_best)
        
        # Save final model
        self.save_checkpoint(self.config.STAGE1_EPOCHS - 1, is_best=False)
        
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
    # Print configuration
    Config.print_config()
    
    # Set random seed
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # Create dataloader
    print("\nLoading dataset...")
    dataloader = get_dataloader(
        root=Config.COCO_ROOT,
        #annFile=Config.COCO_ANN_FILE,
        batch_size=Config.STAGE1_BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        use_blip=Config.USE_BLIP,
        device=Config.DEVICE
    )
    print(f"Dataset loaded: {len(dataloader)} batches")
    
    # Create trainer
    trainer = Stage1Trainer(Config)
    
    # Start training
    trainer.train(dataloader)


if __name__ == "__main__":
    main()