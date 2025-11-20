#!/usr/bin/env python3
"""
Precompute BLIP captions for a COCO-style image folder (or using annotations) and save to JSON.

Outputs a JSON list of objects: {"file_path": ..., "caption": ..., "timestamp": ...}

Usage:
  python precompute_captions.py --root /path/to/images --out captions.json --device cuda
  python precompute_captions.py \
    --root /home/xiaohui/2025_Diffusion_wm/Latent_WOFA/dataset/train  \
    --out data_json/captions.json

This script runs BLIP in the main process (num_workers=0) to avoid multiprocessing GPU issues.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from PIL import Image

import torch

def main():
    parser = argparse.ArgumentParser(description='Precompute BLIP captions for COCO image folder')
    parser.add_argument('--root', type=str, required=True, help='Root folder with images')
    parser.add_argument('--annFile', type=str, default=None, help='Optional COCO captions json to get filenames')
    parser.add_argument('--out', type=str, required=True, help='Output JSON path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch', type=int, default=1, help='Batch size for BLIP generation (default 1)')
    args = parser.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)

    # Load BLIP
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
    except Exception as e:
        print('Please install transformers and BLIP dependencies: pip install transformers')
        raise

    print('Loading BLIP model...')
    processor = BlipProcessor.from_pretrained("/home/xiaohui/.cache/huggingface/hub/stabilityai/blip/")
    model = BlipForConditionalGeneration.from_pretrained("/home/xiaohui/.cache/huggingface/hub/stabilityai/blip/")
    model.to(args.device)
    model.eval()

    # Build list of image file paths
    image_paths = []
    if args.annFile:
        # Read COCO annotations (captions json) to list image file names
        print(f'Loading annotations from {args.annFile}...')
        with open(args.annFile, 'r', encoding='utf-8') as f:
            ann = json.load(f)
        id2file = {img['id']: img['file_name'] for img in ann.get('images', [])}
        # Use unique file list preserving order
        files = [id2file[a['image_id']] for a in ann.get('annotations', []) if a['image_id'] in id2file]
        # remove duplicates while preserving order
        seen = set(); uniq = []
        for fn in files:
            if fn not in seen:
                seen.add(fn); uniq.append(fn)
        for fn in uniq:
            p = root / fn
            if p.exists():
                image_paths.append(p)
    else:
        exts = ('**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.bmp')
        for e in exts:
            image_paths.extend(sorted(root.glob(e)))

    print(f'Found {len(image_paths)} images')

    results = []
    device = torch.device(args.device)

    for p in tqdm(image_paths, desc='Generating captions'):
        try:
            pil = Image.open(p).convert('RGB')
        except Exception as e:
            print(f'Warning: could not open {p}: {e}'); continue

        # Run BLIP processor and model
        with torch.no_grad():
            inputs = processor(pil, return_tensors='pt')
            # move tensors to device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            out = model.generate(**inputs, max_length=50)
            caption = processor.decode(out[0], skip_special_tokens=True)

        results.append({
            'file_path': str(p.resolve()),
            'caption': caption,
            'timestamp': datetime.now().isoformat()
        })

    # Save JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f'Saved {len(results)} captions to {out_path}')

if __name__ == '__main__':
    main()
