"""
Stage II Training: Quality & Fusion Training

Training Flow:
1. Image → VAE Encode → z_img (frozen VAE)
2. QR → Pattern Encoder (E_p) → P (frozen encoder)
3. z_w = Embedder(z_img, P)
4. z_w' = Distortion(z_w, random_strength)
5. P_hat = Extractor(z_w')
6. V_hat = Decoder(P_hat) (optional verification)

Losses (Decoupled Design):
- L_quality = MSE(z_img, z_w)  → Preserve image quality
- L_recovery = MSE(P, P_hat)   → Extract watermark accurately
- L_decoder = MSE(V, V_hat)    → Optional QR reconstruction verification

Goal: Train Embedder and Extractor to embed/extract robust watermarks
      while preserving image quality.
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
from diffusers import AutoencoderKL

from config import Config
from data_provider import get_dataloader
from models import LatentWOFASeal
from distortions import LatentDistortionLayer
from utils.losses import CombinedStageLoss
from utils.metrics import MetricsTracker, calculate_psnr, calculate_ssim, calculate_qr_accuracy
from utils.visualizer import Visualizer


class Stage2Trainer:
    """
    Stage II Trainer: Quality & Fusion Training
    """
    def __init__(self, config, stage1_checkpoint_path=None):
        self.config = config
        
        # Create timestamp for this experiment
        self.timestamp = config.get_timestamp()
        print(f"Experiment timestamp: {self.timestamp}")
        
        # Setup directories
        self.checkpoint_dir = config.get_stage2_checkpoint_dir(self.timestamp)
        self.log_dir = config.get_stage2_log_dir(self.timestamp)
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
        
        # Load Stage I checkpoint
        if stage1_checkpoint_path:
            print(f"Loading Stage I checkpoint: {stage1_checkpoint_path}")
            self.load_stage1_checkpoint(stage1_checkpoint_path)
        else:
            print("WARNING: No Stage I checkpoint provided. Starting from scratch.")
        
        # Freeze Stage I components
        self.model.freeze_stage1()
        
        # Only train Stage II components
        parameters = self.model.get_stage2_parameters()
        total_params = sum(p.numel() for p in parameters)
        print(f"Stage II trainable parameters: {total_params:,}")
        
        # Setup Stable Diffusion VAE (frozen)
        print(f"Loading Stable Diffusion VAE: {config.SD_VAE_MODEL}")
        self.vae = AutoencoderKL.from_pretrained(config.SD_VAE_MODEL).to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        print("VAE loaded and frozen.")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            parameters,
            lr=config.STAGE2_LEARNING_RATE,
            weight_decay=config.STAGE2_WEIGHT_DECAY,
            betas=config.STAGE2_BETAS
        )
        
        # Setup learning rate scheduler
        self.scheduler = self._build_scheduler()
        
        # Setup loss function
        loss_weights = {
            'quality': config.STAGE2_LOSS_QUALITY,
            'recovery': config.STAGE2_LOSS_RECOVERY,
            'decoder': config.STAGE2_LOSS_DECODER,
            'perceptual': 0.0  # Can enable if needed
        }
        self.criterion = CombinedStageLoss(stage=2, loss_weights=loss_weights)
        
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
        
        # Setup AMP
        self.use_amp = config.USE_AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup TensorBoard
        if config.TENSORBOARD_ENABLED:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.visualizer = Visualizer(
                save_dir=os.path.join(self.log_dir, 'visualizations'),
                tensorboard_writer=self.writer
            )
        else:
            self.writer = None
            self.visualizer = None
        
        # Setup metrics tracker
        self.metrics_tracker = MetricsTracker(metrics=config.EVAL_METRICS)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        if self.config.STAGE2_LR_SCHEDULER == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.STAGE2_EPOCHS,
                eta_min=self.config.STAGE2_LR_MIN
            )
        elif self.config.STAGE2_LR_SCHEDULER == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.STAGE2_EPOCHS // 3,
                gamma=0.1
            )
        else:
            return None
    
    def load_stage1_checkpoint(self, checkpoint_path):
        """Load Stage I trained weights."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load only Stage I components
        model_state = checkpoint['model_state_dict']
        
        # Filter Stage I keys
        stage1_state = {}
        for key, value in model_state.items():
            if key.startswith('pattern_encoder') or key.startswith('pattern_decoder'):
                stage1_state[key] = value
        
        # Load state dict
        self.model.load_state_dict(stage1_state, strict=False)
        print(f"Loaded Stage I weights from epoch {checkpoint.get('epoch', 'unknown')}")
    
    def encode_image_to_latent(self, image):
        """
        Encode image to latent using VAE.
        
        Args:
            image: [B, 3, H, W] normalized to [-1, 1]
        
        Returns:
            latent: [B, 4, H//8, W//8]
        """
        with torch.no_grad():
            # VAE encode
            latent_dist = self.vae.encode(image).latent_dist
            latent = latent_dist.sample()
            # Scale latent (SD VAE specific)
            latent = latent * 0.18215
        return latent
    
    def decode_latent_to_image(self, latent):
        """
        Decode latent to image using VAE.
        
        Args:
            latent: [B, 4, H//8, W//8]
        
        Returns:
            image: [B, 3, H, W]
        """
        with torch.no_grad():
            # Unscale latent
            latent = latent / 0.18215
            # VAE decode
            image = self.vae.decode(latent).sample
        return image
    
    def get_random_distortion_strength(self):
        """Get random distortion strength for Stage II."""
        if self.config.STAGE2_RANDOM_DISTORTION:
            # Random strength in [0.3, 0.9]
            return np.random.uniform(0.3, 0.9)
        else:
            return self.config.STAGE2_DISTORTION_STRENGTH
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.model.train()
        self.model.freeze_stage1()  # Ensure Stage I stays frozen
        
        epoch_loss_dict = {
            'total': 0.0,
            'quality': 0.0,
            'recovery': 0.0,
            'decoder': 0.0
        }
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.STAGE2_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            images = batch['image'].to(self.device)  # [B, 3, 256, 256]
            qr_codes = batch['watermark'].to(self.device)  # [B, 1, 256, 256]
            
            # Get random distortion strength
            distortion_strength = self.get_random_distortion_strength()
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                # 1. Encode image to latent (frozen VAE)
                image_latent = self.encode_image_to_latent(images)  # [B, 4, 64, 64]
                
                # 2. Encode QR to pattern latent (frozen encoder)
                with torch.no_grad():
                    pattern_latent = self.model.encode_pattern(qr_codes)  # [B, 1, 64, 64]
                
                # 3. Embed watermark into image latent
                watermarked_latent = self.model.embed_watermark(
                    image_latent, pattern_latent
                )  # [B, 4, 64, 64]
                
                # 4. Apply distortion to watermarked latent
                distorted_latent = self.distortion(
                    watermarked_latent, distortion_strength
                )
                
                # 5. Extract watermark from distorted latent
                extracted_pattern = self.model.extract_watermark(
                    distorted_latent
                )  # [B, 1, 64, 64]
                
                # 6. Optional: Decode extracted pattern to QR (frozen decoder)
                with torch.no_grad():
                    reconstructed_qr = self.model.decode_pattern(
                        extracted_pattern
                    )  # [B, 1, 256, 256]
                
                # Compute losses
                loss, loss_dict = self.criterion(
                    image_latent=image_latent,
                    watermarked_latent=watermarked_latent,
                    pattern_latent=pattern_latent,
                    extracted_pattern=extracted_pattern,
                    original_qr=qr_codes,
                    reconstructed_qr=reconstructed_qr
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.config.GRAD_CLIP_NORM > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_stage2_parameters(),
                        self.config.GRAD_CLIP_NORM
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_stage2_parameters(),
                        self.config.GRAD_CLIP_NORM
                    )
                self.optimizer.step()
            
            # Update metrics
            for key in epoch_loss_dict.keys():
                if key in loss_dict:
                    epoch_loss_dict[key] += loss_dict[key]
            
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'quality': f'{loss_dict.get("quality", 0):.4f}',
                'recovery': f'{loss_dict.get("recovery", 0):.4f}',
                'dist_str': f'{distortion_strength:.2f}'
            })
            
            # TensorBoard logging
            if self.writer and self.global_step % self.config.TENSORBOARD_LOG_INTERVAL == 0:
                self.visualizer.log_loss_curves(loss_dict, self.global_step, stage=2)
                self.writer.add_scalar('Stage2/Distortion_Strength', 
                                      distortion_strength, self.global_step)
                self.writer.add_scalar('Stage2/Learning_Rate',
                                      self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Log images
            if self.visualizer and self.global_step % self.config.TENSORBOARD_IMAGE_INTERVAL == 0:
                with torch.no_grad():
                    # Decode latents to images for visualization
                    original_images = self.decode_latent_to_image(image_latent)
                    watermarked_images = self.decode_latent_to_image(watermarked_latent)
                    
                    self.visualizer.log_stage2_batch(
                        original_images=original_images,
                        watermarked_images=watermarked_images,
                        qr_codes=qr_codes,
                        pattern_latents=pattern_latent,
                        extracted_patterns=extracted_pattern,
                        reconstructed_qrs=reconstructed_qr,
                        global_step=self.global_step
                    )
            
            # Debug mode: limit iterations
            if self.config.DEBUG_MODE and batch_idx >= self.config.DEBUG_ITERATIONS:
                break
        
        # Average epoch losses
        num_batches = len(dataloader) if not self.config.DEBUG_MODE else self.config.DEBUG_ITERATIONS
        for key in epoch_loss_dict.keys():
            epoch_loss_dict[key] /= num_batches
        
        return epoch_loss_dict
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate model on validation set."""
        self.model.eval()
        self.metrics_tracker.reset()
        
        print("\nEvaluating...")
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = batch['image'].to(self.device)
            qr_codes = batch['watermark'].to(self.device)
            
            # Encode to latent
            image_latent = self.encode_image_to_latent(images)
            pattern_latent = self.model.encode_pattern(qr_codes)
            
            # Embed and extract
            watermarked_latent = self.model.embed_watermark(image_latent, pattern_latent)
            
            # Apply fixed distortion for evaluation
            distorted_latent = self.distortion(watermarked_latent, 0.7)
            
            # Extract
            extracted_pattern = self.model.extract_watermark(distorted_latent)
            reconstructed_qr = self.model.decode_pattern(extracted_pattern)
            
            # Decode to images
            original_images = self.decode_latent_to_image(image_latent)
            watermarked_images = self.decode_latent_to_image(watermarked_latent)
            
            # Calculate metrics
            psnr = calculate_psnr(original_images, watermarked_images)
            ssim = calculate_ssim(original_images, watermarked_images)
            mse_latent = torch.mean((image_latent - watermarked_latent) ** 2).item()
            
            # QR accuracy (if enabled)
            if self.config.EVAL_QR_DECODE_ENABLED:
                qr_acc, _, _ = calculate_qr_accuracy(qr_codes, reconstructed_qr)
            else:
                qr_acc = 0.0
            
            # Update tracker
            self.metrics_tracker.update('PSNR', psnr)
            self.metrics_tracker.update('SSIM', ssim)
            self.metrics_tracker.update('MSE', mse_latent)
            self.metrics_tracker.update('QR_Accuracy', qr_acc)
            
            # Limit evaluation batches
            if batch_idx >= 50:  # Evaluate on 50 batches
                break
        
        # Get results
        results = self.metrics_tracker.compute_all()
        return results
    
    def save_checkpoint(self, epoch, metrics=None, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
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
        
        # Save latest model
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
    
    def train(self, train_dataloader, val_dataloader=None):
        """Main training loop."""
        print("\n" + "="*80)
        print("Starting Stage II Training: Quality & Fusion Training")
        print("="*80)
        
        best_psnr = 0.0
        
        for epoch in range(self.config.STAGE2_EPOCHS):
            self.current_epoch = epoch
            
            # Train one epoch
            epoch_start = time.time()
            loss_dict = self.train_epoch(train_dataloader, epoch)
            epoch_time = time.time() - epoch_start
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log epoch metrics
            print(f"\nEpoch {epoch+1}/{self.config.STAGE2_EPOCHS}")
            print(f"  Total Loss: {loss_dict['total']:.4f}")
            print(f"  Quality Loss: {loss_dict['quality']:.4f}")
            print(f"  Recovery Loss: {loss_dict['recovery']:.4f}")
            print(f"  Decoder Loss: {loss_dict.get('decoder', 0):.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            if self.writer:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'Stage2/Loss_Epoch/{key}', value, epoch)
            
            # Evaluate
            if val_dataloader and (epoch + 1) % self.config.STAGE2_EVAL_INTERVAL == 0:
                metrics = self.evaluate(val_dataloader)
                print(f"\nValidation Metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")
                
                if self.visualizer:
                    self.visualizer.log_metrics(metrics, epoch, stage=2)
                
                # Check if best
                is_best = metrics['PSNR'] > best_psnr
                if is_best:
                    best_psnr = metrics['PSNR']
                    print(f"  New best PSNR: {best_psnr:.2f} dB")
            else:
                metrics = None
                is_best = False
            
            # Save checkpoint
            if (epoch + 1) % self.config.STAGE2_SAVE_INTERVAL == 0:
                self.save_checkpoint(epoch, metrics, is_best=is_best)
        
        # Save final model
        self.save_checkpoint(self.config.STAGE2_EPOCHS - 1, metrics, is_best=False)
        
        print("\n" + "="*80)
        print("Stage II Training Completed!")
        print(f"Best PSNR: {best_psnr:.2f} dB")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"TensorBoard logs: {self.log_dir}")
        print("="*80)