import json
from pathlib import Path
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms
import qrcode
import hashlib
import numpy as np


class CachedCOCOWithMetaSeal(data.Dataset):
    """
    Dataset that loads precomputed captions (JSON) and returns image + watermark.

    Expects JSON format: list of {"file_path": "/abs/path/to/img.jpg", "caption": "..."}
    """
    def __init__(self, captions_json, transform=None, qr_size=256):
        captions_json = Path(captions_json)
        if not captions_json.exists():
            raise FileNotFoundError(f"Captions JSON not found: {captions_json}")

        with open(captions_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # keep only entries whose files exist
        self.entries = [e for e in data if Path(e.get('file_path', '')).exists()]
        if not self.entries:
            raise RuntimeError(f"No valid image entries found in {captions_json}")

        self.transform = transform if transform is not None else self._default_transform()
        self.qr_size = qr_size

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def _sign_caption(self, caption, private_key="latent_wofa_seal_2025"):
        message = f"{caption}|{private_key}"
        signature = hashlib.sha256(message.encode('utf-8')).hexdigest()
        signed_message = f"{caption}|SIG:{signature[:16]}"
        return signed_message

    def _generate_qr_code(self, data_str: str):
        # Create QR code
        qr = qrcode.QRCode(
            version=5,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=2,
        )
        qr.add_data(data_str)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img = qr_img.convert('L').resize((self.qr_size, self.qr_size))
        qr_array = np.array(qr_img, dtype=np.float32) / 255.0
        qr_array = (qr_array > 0.5).astype(np.float32)
        qr_tensor = torch.from_numpy(qr_array).unsqueeze(0)
        return qr_tensor

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = Path(entry['file_path'])
        caption = entry.get('caption', '')

        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Sign and make QR
        signed_caption = self._sign_caption(caption)
        watermark = self._generate_qr_code(signed_caption)

        return {
            'image': image,
            'watermark': watermark,
            'caption': caption,
            'signed_caption': signed_caption
        }


def get_dataloader(captions_json, batch_size=8, num_workers=4, shuffle=True):
    dataset = CachedCOCOWithMetaSeal(captions_json)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=True)
    return loader


if __name__ == '__main__':
    # Quick smoke test (no BLIP, just check loading)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--captions', required=True)
    args = parser.parse_args()

    ds = CachedCOCOWithMetaSeal(args.captions)
    sample = ds[0]
    print('Sample keys:', list(sample.keys()))
    print('Image shape:', sample['image'].shape)
    print('Watermark shape:', sample['watermark'].shape)
