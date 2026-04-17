"""Generate depth maps for training data using Depth Anything V2.

Reads the monolithic HDF5 file, generates depth for each frame,
and adds a 'depth' dataset to the file.
"""

import torch
import numpy as np
import h5py
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/data/lewm_data/aigen_train.h5')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--model-size', default='Small', choices=['Small', 'Base', 'Large'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Depth Anything V2
    print(f"Loading Depth Anything V2 {args.model_size}...")
    model = torch.hub.load('DepthAnything/Depth-Anything-V2', f'depth_anything_v2_vit{args.model_size[0].lower()}14',
                           pretrained=True, trust_repo=True)
    model = model.to(device).eval()

    # Process HDF5 file
    with h5py.File(args.input, 'r+') as f:
        n_frames = f['pixels'].shape[0]
        h, w = f['pixels'].shape[1], f['pixels'].shape[2]
        print(f"Processing {n_frames} frames ({h}x{w})")

        # Create depth dataset if not exists
        if 'depth' in f:
            print("Depth dataset already exists, overwriting")
            del f['depth']

        depth_ds = f.create_dataset(
            'depth', shape=(n_frames, h, w), dtype=np.uint8,
            chunks=(min(256, n_frames), h, w)
        )

        for i in range(0, n_frames, args.batch_size):
            end = min(i + args.batch_size, n_frames)
            batch = f['pixels'][i:end]  # (B, H, W, 3) uint8

            # Depth Anything V2 expects (B, 3, H, W) float [0,1] or raw images
            # Use the model's infer_image method per-frame
            for j in range(len(batch)):
                img = batch[j]  # (H, W, 3) uint8 numpy
                depth = model.infer_image(img)  # returns (H, W) float
                # Normalize to 0-255 uint8 (disparity: bright=close)
                depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
                depth_ds[i + j] = depth_norm.astype(np.uint8)

            if (i // args.batch_size) % 10 == 0:
                print(f"  {end}/{n_frames} frames processed")

    print(f"Done. Depth added to {args.input}")


if __name__ == '__main__':
    main()
