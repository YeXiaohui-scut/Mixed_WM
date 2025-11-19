import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CocoCaptions
from PIL import Image
import qrcode
import hashlib
import io
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

class COCOWithMetaSeal(data.Dataset):
    """
    COCO Dataset with MetaSeal watermark generation.
    Combines BLIP captioning + QR code generation for semantic watermarks.
    """
    def __init__(self, root, annFile, transform=None, use_blip=True, device='cuda'):
        """
        Args:
            root: Path to COCO images
            annFile: Path to COCO annotations
            transform: Image transforms
            use_blip: Whether to use BLIP for captioning (set False for testing)
            device: Device for BLIP model
        """
        self.coco_dataset = CocoCaptions(root=root, annFile=annFile)
        self.transform = transform if transform else self._default_transform()
        self.use_blip = use_blip
        self.device = device
        
        # Initialize BLIP model for semantic extraction
        if self.use_blip:
            print("Loading BLIP model for semantic extraction...")
            self.blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(device)
            self.blip_model.eval()
            print("BLIP model loaded successfully.")
        
        # QR code generator settings
        self.qr_size = 256
        
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def _generate_caption(self, image):
        """
        Generate semantic caption using BLIP model.
        
        Args:
            image: PIL Image
        Returns:
            caption: String description
        """
        if not self.use_blip:
            # Fallback for testing without BLIP
            return f"test_image_{torch.randint(0, 100000, (1,)).item()}"
        
        with torch.no_grad():
            # Prepare image for BLIP
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
        return caption
    
    def _sign_caption(self, caption, private_key="latent_wofa_seal_2025"):
        """
        Sign the caption with simulated private key (using SHA256).
        
        Args:
            caption: Text caption
            private_key: Simulated private key
        Returns:
            signature: Signed hash string
        """
        # Combine caption with private key
        message = f"{caption}|{private_key}"
        
        # Generate SHA256 hash as signature
        signature = hashlib.sha256(message.encode('utf-8')).hexdigest()
        
        # Create signed message
        signed_message = f"{caption}|SIG:{signature[:16]}"  # Use first 16 chars
        
        return signed_message
    
    def _generate_qr_code(self, data):
        """
        Generate QR code from signed data.
        
        Args:
            data: String data to encode
        Returns:
            qr_tensor: [1, 256, 256] binary tensor
        """
        # Create QR code
        qr = qrcode.QRCode(
            version=5,  # Controls size (5 = 37x37 modules)
            error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
            box_size=10,
            border=2,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        # Generate QR code image
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to grayscale PIL Image and resize
        qr_img = qr_img.convert('L').resize((self.qr_size, self.qr_size), Image.BILINEAR)
        
        # Convert to numpy array and normalize to [0, 1]
        qr_array = np.array(qr_img, dtype=np.float32) / 255.0
        
        # Binarize: threshold at 0.5
        qr_array = (qr_array > 0.5).astype(np.float32)
        
        # Convert to tensor [1, H, W]
        qr_tensor = torch.from_numpy(qr_array).unsqueeze(0)
        
        return qr_tensor
    
    def __len__(self):
        return len(self.coco_dataset)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor [3, 256, 256]
            watermark: QR code pattern tensor [1, 256, 256]
            caption: Original caption string
        """
        # Load image and annotations from COCO
        image, captions = self.coco_dataset[idx]
        
        # Generate semantic caption using BLIP
        caption = self._generate_caption(image)
        
        # Sign the caption (MetaSeal)
        signed_caption = self._sign_caption(caption)
        
        # Generate QR code watermark
        watermark = self._generate_qr_code(signed_caption)
        
        # Transform image
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'watermark': watermark,
            'caption': caption,
            'signed_caption': signed_caption
        }


def get_dataloader(root, annFile, batch_size=8, num_workers=4, use_blip=True, device='cuda'):
    """
    Create DataLoader for training.
    
    Args:
        root: Path to COCO images
        annFile: Path to COCO annotations
        batch_size: Batch size
        num_workers: Number of workers for data loading
        use_blip: Whether to use BLIP model
        device: Device for BLIP
    
    Returns:
        dataloader: DataLoader instance
    """
    dataset = COCOWithMetaSeal(
        root=root,
        annFile=annFile,
        use_blip=use_blip,
        device=device
    )
    
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# Testing function
if __name__ == "__main__":
    # Test without BLIP (for quick testing)
    print("Testing data provider...")
    
    # Create a dummy dataset for testing
    dataset = COCOWithMetaSeal(
        root="./data/coco/train2017",
        annFile="./data/coco/annotations/captions_train2017.json",
        use_blip=False  # Set to True to test with BLIP
    )
    
    # Test single sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Watermark shape: {sample['watermark'].shape}")
    print(f"Caption: {sample['caption']} ")
    print(f"Signed caption: {sample['signed_caption']}")
    print(f"Watermark range: [{sample['watermark'].min()}, {sample['watermark'].max()}]")
    
    print("\nData provider test passed!")