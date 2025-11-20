"""
Visualization Tools for Latent-WOFA-Seal

Provides functions for:
- Saving comparison grids
- Tensor to image conversion
- TensorBoard image logging
- Result visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from PIL import Image
import torchvision.utils as vutils


def normalize_tensor(tensor, min_val=None, max_val=None):
    """
    Normalize tensor to [0, 1] range for visualization.
    
    Args:
        tensor: Input tensor
        min_val: Minimum value (if None, use tensor min)
        max_val: Maximum value (if None, use tensor max)
    
    Returns:
        normalized: Tensor in [0, 1]
    """
    if min_val is None:
        min_val = tensor.min()
    if max_val is None:
        max_val = tensor.max()
    
    if max_val - min_val > 1e-6:
        normalized = (tensor - min_val) / (max_val - min_val)
    else:
        normalized = torch.zeros_like(tensor)
    
    return normalized.clamp(0, 1)


def tensor_to_image(tensor, denormalize=True):
    """
    Convert tensor to PIL Image.
    
    Args:
        tensor: [C, H, W] or [H, W] tensor
        denormalize: Whether to denormalize from [-1, 1] to [0, 1]
    
    Returns:
        image: PIL Image
    """
    # Convert to CPU and numpy
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    
    # Denormalize if needed
    if denormalize and tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    # Ensure [0, 1] range
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy
    if tensor.dim() == 3:  # [C, H, W]
        array = tensor.permute(1, 2, 0).numpy()
    else:  # [H, W]
        array = tensor.numpy()
    
    # Scale to [0, 255]
    array = (array * 255).astype(np.uint8)
    
    # Convert to PIL
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    
    return Image.fromarray(array)


def save_comparison_grid(images_dict, save_path, titles=None, figsize=(15, 10), dpi=150):
    """
    Save a grid of images for comparison.
    
    Args:
        images_dict: Dict of {name: tensor} where tensor is [B, C, H, W]
        save_path: Path to save figure
        titles: Optional list of titles for each image type
        figsize: Figure size
        dpi: DPI for saving
    """
    num_types = len(images_dict)
    batch_size = list(images_dict.values())[0].size(0)
    
    # Limit to reasonable number of samples
    max_samples = min(batch_size, 4)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(max_samples, num_types, figure=fig, hspace=0.3, wspace=0.1)
    
    # Get image names
    names = list(images_dict.keys())
    
    # Plot grid
    for row in range(max_samples):
        for col, name in enumerate(names):
            ax = fig.add_subplot(gs[row, col])
            
            # Get image
            img_tensor = images_dict[name][row]
            
            # Convert to displayable format
            if img_tensor.dim() == 3:  # [C, H, W]
                if img_tensor.size(0) == 1:  # Grayscale
                    img = img_tensor.squeeze(0).cpu().numpy()
                    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                else:  # RGB
                    img = img_tensor.permute(1, 2, 0).cpu().numpy()
                    img = np.clip(img, 0, 1)
                    ax.imshow(img)
            else:  # [H, W]
                img = img_tensor.cpu().numpy()
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            
            # Add title only on first row
            if row == 0:
                title = titles[col] if titles else name
                ax.set_title(title, fontsize=10)
            
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


class Visualizer:
    """
    Comprehensive visualization tool for training monitoring.
    """
    def __init__(self, save_dir, tensorboard_writer=None):
        """
        Args:
            save_dir: Directory to save visualizations
            tensorboard_writer: Optional TensorBoard SummaryWriter
        """
        self.save_dir = save_dir
        self.writer = tensorboard_writer
        os.makedirs(save_dir, exist_ok=True)
    
    def log_stage1_batch(self, 
                        qr_codes, 
                        latent_patterns,
                        distorted_patterns,
                        reconstructed_qrs,
                        global_step,
                        num_images=4):
        """
        Visualize Stage I training batch.
        
        Args:
            qr_codes: [B, 1, 256, 256]
            latent_patterns: [B, 1, 64, 64]
            distorted_patterns: [B, 1, 64, 64]
            reconstructed_qrs: [B, 1, 256, 256]
            global_step: Current training step
            num_images: Number of images to visualize
        """
        n = min(num_images, qr_codes.size(0))
        
        # Prepare images dict
        images_dict = {
            'Original_QR': qr_codes[:n],
            'Latent_Pattern': normalize_tensor(latent_patterns[:n]),
            'Distorted_Pattern': normalize_tensor(distorted_patterns[:n]),
            'Reconstructed_QR': reconstructed_qrs[:n],
            'Difference': torch.abs(qr_codes[:n] - reconstructed_qrs[:n])
        }
        
        # Save to file
        save_path = os.path.join(self.save_dir, f'stage1_step_{global_step:06d}.png')
        save_comparison_grid(images_dict, save_path)
        
        # Log to TensorBoard
        if self.writer:
            for name, imgs in images_dict.items():
                self.writer.add_images(f'Stage1/{name}', imgs, global_step, dataformats='NCHW')
    
    def log_stage2_batch(self,
                        original_images,
                        watermarked_images,
                        qr_codes,
                        pattern_latents,
                        extracted_patterns,
                        reconstructed_qrs,
                        global_step,
                        num_images=4):
        """
        Visualize Stage II training batch.
        
        Args:
            original_images: [B, 3, H, W] - Decoded from original latent
            watermarked_images: [B, 3, H, W] - Decoded from watermarked latent
            qr_codes: [B, 1, 256, 256]
            pattern_latents: [B, 1, 64, 64]
            extracted_patterns: [B, 1, 64, 64]
            reconstructed_qrs: [B, 1, 256, 256] - Decoded from extracted pattern
            global_step: Current training step
            num_images: Number of images to visualize
        """
        n = min(num_images, original_images.size(0))
        
        # Normalize images to [0, 1]
        original_imgs = normalize_tensor(original_images[:n])
        watermarked_imgs = normalize_tensor(watermarked_images[:n])
        
        # Prepare images dict
        images_dict = {
            'Original_Image': original_imgs,
            'Watermarked_Image': watermarked_imgs,
            'Image_Difference': torch.abs(original_imgs - watermarked_imgs) * 10,  # Amplify difference
            'Original_QR': qr_codes[:n],
            'Pattern_Latent': normalize_tensor(pattern_latents[:n]),
            'Extracted_Pattern': normalize_tensor(extracted_patterns[:n]),
            'Reconstructed_QR': reconstructed_qrs[:n],
            'QR_Difference': torch.abs(qr_codes[:n] - reconstructed_qrs[:n])
        }
        
        # Save to file
        save_path = os.path.join(self.save_dir, f'stage2_step_{global_step:06d}.png')
        save_comparison_grid(images_dict, save_path, figsize=(20, 10))
        
        # Log to TensorBoard
        if self.writer:
            # Images
            for name, imgs in images_dict.items():
                self.writer.add_images(f'Stage2/{name}', imgs, global_step, dataformats='NCHW')
            
            # Pattern comparison
            pattern_diff = F.mse_loss(pattern_latents[:n], extracted_patterns[:n])
            self.writer.add_scalar('Stage2/Pattern_MSE', pattern_diff.item(), global_step)
    
    def log_loss_curves(self, loss_dict, global_step, stage=1):
        """
        Log loss curves to TensorBoard.
        
        Args:
            loss_dict: Dict of losses
            global_step: Current step
            stage: Training stage (1 or 2)
        """
        if self.writer:
            prefix = f'Stage{stage}/Loss'
            for name, value in loss_dict.items():
                self.writer.add_scalar(f'{prefix}/{name}', value, global_step)
    
    def log_metrics(self, metrics_dict, epoch, stage=1):
        """
        Log evaluation metrics to TensorBoard.
        
        Args:
            metrics_dict: Dict of metrics
            epoch: Current epoch
            stage: Training stage
        """
        if self.writer:
            prefix = f'Stage{stage}/Metrics'
            for name, value in metrics_dict.items():
                self.writer.add_scalar(f'{prefix}/{name}', value, epoch)
    
    def close(self):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()


# Testing
if __name__ == "__main__":
    print("Testing visualizer...")
    
    # Create dummy data
    qr_codes = torch.rand(4, 1, 256, 256)
    latent_patterns = torch.randn(4, 1, 64, 64)
    distorted_patterns = torch.randn(4, 1, 64, 64)
    reconstructed_qrs = torch.rand(4, 1, 256, 256)
    
    # Test save_comparison_grid
    print("\n--- Testing save_comparison_grid ---")
    images_dict = {
        'Original': qr_codes,
        'Latent': normalize_tensor(latent_patterns),
        'Distorted': normalize_tensor(distorted_patterns),
        'Reconstructed': reconstructed_qrs
    }
    
    save_comparison_grid(images_dict, 'test_grid.png')
    print("Saved test_grid.png")
    
    # Test Visualizer
    print("\n--- Testing Visualizer ---")
    visualizer = Visualizer(save_dir='./test_vis')
    
    visualizer.log_stage1_batch(
        qr_codes, latent_patterns, distorted_patterns, reconstructed_qrs,
        global_step=100
    )
    print("Saved Stage I visualization")
    
    print("\n✅ Visualizer test passed!")
    
    # Clean up
    import shutil
    if os.path.exists('test_grid.png'):
        os.remove('test_grid.png')
    if os.path.exists('./test_vis'):
        shutil.rmtree('./test_vis')