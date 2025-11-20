"""
Latent-WOFA-Seal Configuration File
Contains all hyperparameters, paths, and training settings.

Modified to support WOFA Progressive Distortion Strategy:
- Adjusted for 200 epochs training
- Noise: Progressive in first 20% epochs (0.01 → 1.0)
- Geometry: Progressive in first 50% epochs (0.05 → 1.0)
- Masking: Three-stage progressive (adjusted for reasonable pace)
"""

import os
from datetime import datetime


class Config:
    """
    Main configuration class for Latent-WOFA-Seal project.
    """
    
    # ==================== Project Info ====================
    PROJECT_NAME = "Latent-WOFA-Seal"
    VERSION = "1.0.0"
    
    # ==================== Paths ====================
    # Base directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "dataset")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    # COCO dataset paths
    COCO_ROOT = os.path.join(DATA_DIR, "train")
    # COCO_ANN_FILE = os.path.join(DATA_DIR, "coco", "annotations", "captions_train2017.json")
    COCO_VAL_ROOT = os.path.join(DATA_DIR, "coco", "val2017")
    COCO_VAL_ANN_FILE = os.path.join(DATA_DIR, "coco", "annotations", "captions_val2017.json")
    
    # Create directories if not exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # ==================== Timestamp ====================
    @staticmethod
    def get_timestamp():
        """Generate timestamp for experiment tracking."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ==================== Device Settings ====================
    DEVICE = "cuda"  # "cuda" or "cpu"
    NUM_WORKERS = 0
    PIN_MEMORY = True
    
    # ==================== Model Architecture ====================
    # Pattern Encoder
    PATTERN_ENCODER_CONFIG = {
        'in_channels': 1,
        'latent_channels': 64,
        'num_res_blocks': 6,
        'num_heads': 8
    }
    
    # Pattern Decoder
    PATTERN_DECODER_CONFIG = {
        'in_channels': 1,
        'hidden_channels': 64
    }
    
    # Latent Embedder
    LATENT_EMBEDDER_CONFIG = {
        'image_channels': 4,
        'pattern_channels': 1,
        'hidden_channels': 64
    }
    
    # Latent Extractor
    LATENT_EXTRACTOR_CONFIG = {
        'in_channels': 4,
        'base_channels': 32
    }
    
    # ==================== Stage I Training (WOFA Pre-training) ====================
    # Training hyperparameters
    STAGE1_EPOCHS = 1000
    STAGE1_BATCH_SIZE = 4
    STAGE1_LEARNING_RATE = 1e-4
    STAGE1_WEIGHT_DECAY = 1e-5
    STAGE1_BETAS = (0.9, 0.999)
    
    # Learning rate scheduler
    STAGE1_LR_SCHEDULER = "cosine"  # "cosine", "step", or "none"
    STAGE1_LR_MIN = 1e-6
    
    # Loss weights
    STAGE1_LOSS_RECONSTRUCTION = 1.0
    STAGE1_LOSS_PERCEPTUAL = 0.0  # Can add perceptual loss if needed
    
    # ==================== WOFA Progressive Distortion Strategy ====================
    # Enable WOFA progressive curriculum learning
    STAGE1_PROGRESSIVE_DISTORTION = True
    
    # Distortion warmup ratios (as fraction of total epochs)
    # ADJUSTED FOR 200 EPOCHS TO AVOID TOO RAPID INCREASE
    
    # Noise: Progressive in first 20% epochs
    # Epoch 1-40: 0.01 → 1.0
    WOFA_NOISE_WARMUP_RATIO = 0.2
    
    # Geometry: Progressive in first 50% epochs
    # Epoch 1-100: 0.05 → 1.0
    WOFA_GEOMETRY_WARMUP_RATIO = 0.5
    
    # Masking Stage 1: Progressive in first 5% epochs
    # Epoch 1-10: 0.10 → 0.50 (keep at least 50%)
    # CHANGED FROM 0.01 to 0.05 to slow down masking progression
    WOFA_MASKING_STAGE1_RATIO = 0.05
    
    # Masking Stage 2: Progressive in first 25% epochs
    # Epoch 10-50: 0.50 → 0.75 (keep at least 25%)
    # CHANGED FROM 0.1 to 0.25 to slow down masking progression
    WOFA_MASKING_STAGE2_RATIO = 0.25
    
    # Masking Stage 3: After 25% epochs
    # Epoch 50+: 0.70 (cap at 70% masking)
    
    # Distortion schedule type: 'linear', 'cosine', or 'exponential'
    # 'linear': Uniform increase (recommended for clarity)
    # 'cosine': Smooth acceleration (good for smoother transitions)
    # 'exponential': Fast early increase (not recommended for beginners)
    WOFA_SCHEDULE_TYPE = 'linear'
    
    # Checkpointing
    STAGE1_SAVE_INTERVAL = 5   # Save every 5 epochs (changed from 1 for efficiency)
    STAGE1_EVAL_INTERVAL = 5   # Evaluate every 5 epochs
    
    # ==================== Stage II Training (Quality & Fusion) ====================
    # Training hyperparameters
    STAGE2_EPOCHS = 50
    STAGE2_BATCH_SIZE = 8
    STAGE2_LEARNING_RATE = 5e-5
    STAGE2_WEIGHT_DECAY = 1e-5
    STAGE2_BETAS = (0.9, 0.999)
    
    # Learning rate scheduler
    STAGE2_LR_SCHEDULER = "cosine"
    STAGE2_LR_MIN = 1e-6
    STAGE2_LR_WARMUP_EPOCHS = 3
    
    # Loss weights (decoupled design)
    STAGE2_LOSS_QUALITY = 1.0      # MSE(z_img, z_w)
    STAGE2_LOSS_RECOVERY = 10.0    # MSE(P, P_hat)
    STAGE2_LOSS_DECODER = 1.0      # MSE(V, V_hat) - optional
    
    # Distortion settings for Stage II
    STAGE2_DISTORTION_STRENGTH = 0.7  # Fixed or random
    STAGE2_RANDOM_DISTORTION = True   # Random strength in [0.3, 0.9]
    
    # Stable Diffusion VAE
    SD_VAE_MODEL = "stabilityai/sd-vae-ft-mse"
    
    # Checkpointing
    STAGE2_SAVE_INTERVAL = 5
    STAGE2_EVAL_INTERVAL = 1
    
    # ==================== Distortion Parameters ====================
    # Latent Masking (simulates partial theft/cropping)
    DISTORTION_MASKING_ENABLED = True
    DISTORTION_MASKING_MIN_RATIO = 0.1   # Minimum mask area ratio (not used in WOFA, kept for compatibility)
    DISTORTION_MASKING_MAX_RATIO = 0.7   # Maximum mask area ratio (70% at full strength)
    
    # Latent Geometry (simulates geometric transforms)
    DISTORTION_GEOMETRY_ENABLED = True
    DISTORTION_TRANSLATION_MAX = 0.2     # Max translation ratio (±20%)
    DISTORTION_ROTATION_MAX = 45.0       # Max rotation in degrees (±45°)
    DISTORTION_SCALE_MIN = 0.8           # Min scale factor
    DISTORTION_SCALE_MAX = 1.2           # Max scale factor
    
    # Noise
    DISTORTION_NOISE_ENABLED = True
    DISTORTION_NOISE_MAX_STD = 0.1       # Max Gaussian noise std (at full strength)
    
    # ==================== TensorBoard Settings ====================
    TENSORBOARD_ENABLED = True
    TENSORBOARD_LOG_INTERVAL = 50        # Log every N iterations
    TENSORBOARD_IMAGE_INTERVAL = 200     # Log images every N iterations
    TENSORBOARD_HISTOGRAM_INTERVAL = 500 # Log histograms every N iterations
    
    # ==================== Visualization Settings ====================
    VIS_BATCH_SIZE = 4  # Number of samples to visualize
    VIS_SAVE_FORMAT = "png"
    VIS_DPI = 150
    
    # ==================== BLIP Settings ====================
    USE_BLIP = True  # Set to False for faster testing without BLIP
    BLIP_MODEL = "/home/xiaohui/.cache/huggingface/hub/stabilityai/blip/"
    BLIP_MAX_LENGTH = 50
    
    # ==================== QR Code Settings ====================
    QR_VERSION = 5  # QR code version (1-40)
    QR_ERROR_CORRECTION = "H"  # L, M, Q, H (High = most robust)
    QR_BOX_SIZE = 10
    QR_BORDER = 2
    QR_SIZE = 256  # Output QR code size
    
    # Signature key (for MetaSeal)
    QR_PRIVATE_KEY = "latent_wofa_seal_2025_secret_key"
    
    # ==================== Evaluation Settings ====================
    EVAL_BATCH_SIZE = 16
    EVAL_QR_DECODE_ENABLED = True  # Try to decode QR codes during evaluation
    
    # Metrics
    EVAL_METRICS = ["PSNR", "SSIM", "MSE", "QR_Accuracy"]
    
    # ==================== Inference Settings ====================
    INFERENCE_DEVICE = "cuda"
    INFERENCE_BATCH_SIZE = 1
    
    # ==================== Experimental Settings ====================
    # Mixed precision training
    USE_AMP = True  # Automatic Mixed Precision
    
    # Gradient clipping
    GRAD_CLIP_NORM = 1.0
    
    # Reproducibility
    RANDOM_SEED = 42
    
    # Debug mode
    DEBUG_MODE = False
    DEBUG_ITERATIONS = 10  # Only train for N iterations in debug mode
    
    @classmethod
    def print_config(cls):
        """Print all configuration settings."""
        print("=" * 80)
        print(f"{cls.PROJECT_NAME} v{cls.VERSION} - Configuration")
        print("=" * 80)
        print("\n📊 WOFA Progressive Distortion Strategy (Adjusted for 200 epochs):")
        print(f"  Noise warmup:     epochs 1-{int(cls.STAGE1_EPOCHS * cls.WOFA_NOISE_WARMUP_RATIO)} ({cls.WOFA_NOISE_WARMUP_RATIO:.0%}) → 0.01 to 1.0")
        print(f"  Geometry warmup:  epochs 1-{int(cls.STAGE1_EPOCHS * cls.WOFA_GEOMETRY_WARMUP_RATIO)} ({cls.WOFA_GEOMETRY_WARMUP_RATIO:.0%}) → 0.05 to 1.0")
        print(f"  Masking stage 1:  epochs 1-{int(cls.STAGE1_EPOCHS * cls.WOFA_MASKING_STAGE1_RATIO)} ({cls.WOFA_MASKING_STAGE1_RATIO:.0%}) → 0.10 to 0.50")
        print(f"  Masking stage 2:  epochs {int(cls.STAGE1_EPOCHS * cls.WOFA_MASKING_STAGE1_RATIO)+1}-{int(cls.STAGE1_EPOCHS * cls.WOFA_MASKING_STAGE2_RATIO)} ({cls.WOFA_MASKING_STAGE2_RATIO:.0%}) → 0.50 to 0.75")
        print(f"  Masking stage 3:  epochs {int(cls.STAGE1_EPOCHS * cls.WOFA_MASKING_STAGE2_RATIO)+1}+ → 0.70 (capped)")
        print(f"  Schedule type:    {cls.WOFA_SCHEDULE_TYPE}")
        print("=" * 80)
        
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value) and key.isupper():
                print(f"{key:40s} = {value}")
        
        print("=" * 80)
    
    @classmethod
    def get_stage1_log_dir(cls, timestamp=None):
        """Get Stage I log directory with timestamp."""
        if timestamp is None:
            timestamp = cls.get_timestamp()
        return os.path.join(cls.LOG_DIR, f"stage1_{timestamp}")
    
    @classmethod
    def get_stage2_log_dir(cls, timestamp=None):
        """Get Stage II log directory with timestamp."""
        if timestamp is None:
            timestamp = cls.get_timestamp()
        return os.path.join(cls.LOG_DIR, f"stage2_{timestamp}")
    
    @classmethod
    def get_stage1_checkpoint_dir(cls, timestamp=None):
        """Get Stage I checkpoint directory with timestamp."""
        if timestamp is None:
            timestamp = cls.get_timestamp()
        return os.path.join(cls.CHECKPOINT_DIR, f"stage1_{timestamp}")
    
    @classmethod
    def get_stage2_checkpoint_dir(cls, timestamp=None):
        """Get Stage II checkpoint directory with timestamp."""
        if timestamp is None:
            timestamp = cls.get_timestamp()
        return os.path.join(cls.CHECKPOINT_DIR, f"stage2_{timestamp}")


# Export for easy import
if __name__ == "__main__":
    Config.print_config()