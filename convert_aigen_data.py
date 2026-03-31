"""Convert Aigen HDF5 episodes to LeWM's monolithic HDF5 format.

Aigen format: separate files per episode, keys: images (N,224,224,3), actions (N,2), proprio (N,5)
LeWM format: single .h5 with flat arrays + ep_len/ep_offset metadata, keys: pixels, action, proprio
"""

import h5py
import numpy as np
from pathlib import Path
import argparse


def convert(input_dir: str, output_path: str):
    input_dir = Path(input_dir)
    files = sorted(
        [f for f in input_dir.iterdir() if f.suffix in ('.hdf5', '.h5')]
    )
    print(f"Found {len(files)} episode files in {input_dir}")

    # First pass: compute total size and episode lengths
    ep_lengths = []
    total_frames = 0
    for f in files:
        with h5py.File(f, 'r') as hf:
            # Handle both naming conventions
            if 'images' in hf:
                n = hf['images'].shape[0]
            elif 'pixels' in hf:
                n = hf['pixels'].shape[0]
            else:
                raise KeyError(f"No 'images' or 'pixels' key in {f}")
            ep_lengths.append(n)
            total_frames += n

    ep_offsets = np.cumsum([0] + ep_lengths[:-1]).astype(np.int64)
    ep_lengths = np.array(ep_lengths, dtype=np.int64)

    print(f"Total frames: {total_frames}")
    print(f"Episode lengths: min={ep_lengths.min()}, max={ep_lengths.max()}, mean={ep_lengths.mean():.0f}")

    # Get shapes from first file
    with h5py.File(files[0], 'r') as hf:
        img_key = 'images' if 'images' in hf else 'pixels'
        img_shape = hf[img_key].shape[1:]  # (H, W, C)
        action_dim = hf['actions'].shape[1] if 'actions' in hf else hf['action'].shape[1]
        has_proprio = 'proprio' in hf
        proprio_dim = hf['proprio'].shape[1] if has_proprio else 0

    print(f"Image shape: {img_shape}, Action dim: {action_dim}, Proprio dim: {proprio_dim}")

    # Second pass: write monolithic file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as out:
        # Create datasets
        pixels_ds = out.create_dataset(
            'pixels', shape=(total_frames, *img_shape), dtype=np.uint8,
            chunks=(min(256, total_frames), *img_shape)
        )
        action_ds = out.create_dataset(
            'action', shape=(total_frames, action_dim), dtype=np.float32,
            chunks=(min(1024, total_frames), action_dim)
        )
        if has_proprio:
            proprio_ds = out.create_dataset(
                'proprio', shape=(total_frames, proprio_dim), dtype=np.float32,
                chunks=(min(1024, total_frames), proprio_dim)
            )

        # Metadata
        out.create_dataset('ep_len', data=ep_lengths)
        out.create_dataset('ep_offset', data=ep_offsets)

        # Copy data
        offset = 0
        for i, f in enumerate(files):
            with h5py.File(f, 'r') as hf:
                n = ep_lengths[i]
                img_key = 'images' if 'images' in hf else 'pixels'
                act_key = 'actions' if 'actions' in hf else 'action'

                pixels_ds[offset:offset + n] = hf[img_key][:]
                action_ds[offset:offset + n] = hf[act_key][:]
                if has_proprio:
                    proprio_ds[offset:offset + n] = hf['proprio'][:]

            offset += n
            if (i + 1) % 10 == 0 or i == len(files) - 1:
                print(f"  Processed {i + 1}/{len(files)} episodes ({offset}/{total_frames} frames)")

    print(f"\nWritten to {output_path}")
    print(f"  pixels: ({total_frames}, {img_shape})")
    print(f"  action: ({total_frames}, {action_dim})")
    if has_proprio:
        print(f"  proprio: ({total_frames}, {proprio_dim})")
    print(f"  ep_len: ({len(ep_lengths)},)")
    print(f"  ep_offset: ({len(ep_offsets)},)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/data/cosmos_policy_data/train')
    parser.add_argument('--output', default='/data/lewm_data/aigen_train.h5')
    args = parser.parse_args()
    convert(args.input, args.output)
