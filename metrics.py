"""
Evaluation Metrics for Latent-WOFA-Seal

Includes:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MSE (Mean Squared Error)
- QR Code Accuracy (decoding success rate)
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
from pyzbar.pyzbar import decode as qr_decode
from PIL import Image
import io


def tensor_to_numpy(tensor):
    """
    Convert PyTorch tensor to NumPy array.
    
    Args:
        tensor: [B, C, H, W] or [C, H, W]
    
    Returns:
        array: NumPy array [H, W, C] or [H, W]
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
        
        # Remove batch dimension if single image
        if tensor.dim() == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        
        # Convert to numpy
        array = tensor.numpy()
        
        # Transpose from [C, H, W] to [H, W, C] if needed
        if array.ndim == 3:
            array = np.transpose(array, (1, 2, 0))
        
        return array
    else:
        return tensor


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR between two images.
    
    Args:
        img1: Tensor [B, C, H, W] or numpy array
        img2: Tensor [B, C, H, W] or numpy array
        max_val: Maximum possible pixel value (1.0 for normalized images)
    
    Returns:
        psnr_value: Average PSNR across batch
    """
    # Convert to numpy if needed
    if isinstance(img1, torch.Tensor):
        img1 = tensor_to_numpy(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor_to_numpy(img2)
    
    # Handle batch
    if img1.ndim == 4:  # Batch of images
        psnr_values = []
        for i in range(img1.shape[0]):
            psnr_val = psnr(img1[i], img2[i], data_range=max_val)
            psnr_values.append(psnr_val)
        return np.mean(psnr_values)
    else:
        return psnr(img1, img2, data_range=max_val)


def calculate_ssim(img1, img2, max_val=1.0, multichannel=True):
    """
    Calculate SSIM between two images.
    
    Args:
        img1: Tensor [B, C, H, W] or numpy array
        img2: Tensor [B, C, H, W] or numpy array
        max_val: Maximum possible pixel value
        multichannel: Whether to treat as multichannel image
    
    Returns:
        ssim_value: Average SSIM across batch
    """
    # Convert to numpy if needed
    if isinstance(img1, torch.Tensor):
        img1 = tensor_to_numpy(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor_to_numpy(img2)
    
    # Handle batch
    if img1.ndim == 4:  # Batch of images
        ssim_values = []
        for i in range(img1.shape[0]):
            # Determine if multichannel based on shape
            is_multichannel = img1[i].shape[-1] > 1 if img1[i].ndim == 3 else False
            ssim_val = ssim(
                img1[i], img2[i], 
                data_range=max_val,
                channel_axis=2 if is_multichannel else None
            )
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        is_multichannel = img1.shape[-1] > 1 if img1.ndim == 3 else False
        return ssim(
            img1, img2, 
            data_range=max_val,
            channel_axis=2 if is_multichannel else None
        )


def calculate_mse(img1, img2):
    """
    Calculate MSE between two images.
    
    Args:
        img1: Tensor [B, C, H, W]
        img2: Tensor [B, C, H, W]
    
    Returns:
        mse_value: Mean squared error
    """
    if isinstance(img1, torch.Tensor) and isinstance(img2, torch.Tensor):
        return F.mse_loss(img1, img2).item()
    else:
        # Convert to numpy
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        return np.mean((img1 - img2) ** 2)


def decode_qr_code(qr_image):
    """
    Try to decode QR code from image.
    
    Args:
        qr_image: Tensor [1, H, W] or [H, W] or PIL Image or numpy array
    
    Returns:
        decoded_data: String if successful, None otherwise
        success: Boolean indicating if decode was successful
    """
    try:
        # Convert to PIL Image
        if isinstance(qr_image, torch.Tensor):
            qr_array = tensor_to_numpy(qr_image)
            # Ensure grayscale
            if qr_array.ndim == 3:
                qr_array = qr_array[:, :, 0]
            # Scale to [0, 255]
            qr_array = (qr_array * 255).astype(np.uint8)
            qr_pil = Image.fromarray(qr_array, mode='L')
        elif isinstance(qr_image, np.ndarray):
            if qr_image.ndim == 3:
                qr_image = qr_image[:, :, 0]
            qr_array = (qr_image * 255).astype(np.uint8)
            qr_pil = Image.fromarray(qr_array, mode='L')
        elif isinstance(qr_image, Image.Image):
            qr_pil = qr_image.convert('L')
        else:
            return None, False
        
        # Try to decode
        decoded_objects = qr_decode(qr_pil)
        
        if len(decoded_objects) > 0:
            # Return first decoded data
            decoded_data = decoded_objects[0].data.decode('utf-8')
            return decoded_data, True
        else:
            return None, False
            
    except Exception as e:
        # print(f"QR decode error: {e}")
        return None, False


def calculate_qr_accuracy(original_qr_batch, reconstructed_qr_batch):
    """
    Calculate QR code reconstruction accuracy.
    
    Args:
        original_qr_batch: Tensor [B, 1, H, W] - Original QR codes
        reconstructed_qr_batch: Tensor [B, 1, H, W] - Reconstructed QR codes
    
    Returns:
        accuracy: Percentage of successfully decoded QR codes
        decode_rate: Percentage of reconstructed QRs that could be decoded
        match_rate: Percentage of decoded QRs that match original
    """
    batch_size = original_qr_batch.size(0)
    
    successful_decodes = 0
    successful_matches = 0
    
    for i in range(batch_size):
        original_qr = original_qr_batch[i]
        reconstructed_qr = reconstructed_qr_batch[i]
        
        # Try to decode original
        original_data, original_success = decode_qr_code(original_qr)
        
        # Try to decode reconstructed
        reconstructed_data, reconstructed_success = decode_qr_code(reconstructed_qr)
        
        if reconstructed_success:
            successful_decodes += 1
            
            # Check if matches original
            if original_success and original_data == reconstructed_data:
                successful_matches += 1
    
    decode_rate = (successful_decodes / batch_size) * 100
    
    if successful_decodes > 0:
        match_rate = (successful_matches / successful_decodes) * 100
    else:
        match_rate = 0.0
    
    # Overall accuracy
    accuracy = (successful_matches / batch_size) * 100
    
    return accuracy, decode_rate, match_rate


class MetricsTracker:
    """
    Track and compute metrics during training/evaluation.
    """
    def __init__(self, metrics=None):
        """
        Args:
            metrics: List of metric names to track
                    ['PSNR', 'SSIM', 'MSE', 'QR_Accuracy']
        """
        if metrics is None:
            metrics = ['PSNR', 'SSIM', 'MSE', 'QR_Accuracy']
        
        self.metrics = metrics
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.values = {metric: [] for metric in self.metrics}
        self.counts = {metric: 0 for metric in self.metrics}
    
    def update(self, metric_name, value, count=1):
        """
        Update a specific metric.
        
        Args:
            metric_name: Name of metric
            value: Metric value
            count: Number of samples (for averaging)
        """
        if metric_name in self.values:
            self.values[metric_name].append(value)
            self.counts[metric_name] += count
    
    def compute(self, metric_name):
        """
        Compute average of a specific metric.
        
        Args:
            metric_name: Name of metric
        
        Returns:
            average: Average value
        """
        if metric_name in self.values and len(self.values[metric_name]) > 0:
            return np.mean(self.values[metric_name])
        else:
            return 0.0
    
    def compute_all(self):
        """
        Compute all tracked metrics.
        
        Returns:
            results: Dict of metric averages
        """
        results = {}
        for metric in self.metrics:
            results[metric] = self.compute(metric)
        return results
    
    def get_summary(self):
        """Get formatted summary string."""
        results = self.compute_all()
        summary = " | ".join([f"{k}: {v:.4f}" for k, v in results.items()])
        return summary


# Testing
if __name__ == "__main__":
    print("Testing metrics...")
    
    # Test PSNR and SSIM
    print("\n--- Testing PSNR and SSIM ---")
    img1 = torch.randn(4, 3, 256, 256)
    img2 = img1 + 0.1 * torch.randn(4, 3, 256, 256)
    
    psnr_val = calculate_psnr(img1, img2)
    ssim_val = calculate_ssim(img1, img2)
    mse_val = calculate_mse(img1, img2)
    
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"MSE: {mse_val:.6f}")
    
    # Test MetricsTracker
    print("\n--- Testing MetricsTracker ---")
    tracker = MetricsTracker(metrics=['PSNR', 'SSIM', 'MSE'])
    
    for i in range(5):
        tracker.update('PSNR', 30.0 + i)
        tracker.update('SSIM', 0.9 + i * 0.01)
        tracker.update('MSE', 0.01 - i * 0.001)
    
    results = tracker.compute_all()
    print(f"Tracked metrics: {results}")
    print(f"Summary: {tracker.get_summary()}")
    
    print("\n✅ Metrics test passed!")