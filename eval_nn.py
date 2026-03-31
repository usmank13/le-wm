"""Nearest-neighbor retrieval evaluation for LeWM encoder.

Encodes val set frames, finds top-5 nearest neighbors for random queries,
and saves a grid visualization. Optionally compares to DINOv2-small baseline.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import h5py
import argparse


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_lewm_encoder(ckpt_path, device):
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.eval()
    encoder = model.encoder
    projector = model.projector
    return encoder, projector


def load_dinov2_small(device):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
    # Disable xformers to allow CPU fallback, force to device
    for block in model.blocks:
        block.attn.memory_efficient = False
    model = model.to(device).eval()
    return model


def load_frames(h5_path, max_frames=2000, stride=5):
    """Load a subset of frames from the HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        total = f['pixels'].shape[0]
        indices = np.arange(0, total, stride)[:max_frames]
        frames = f['pixels'][indices]  # (N, H, W, C) uint8
    return frames, indices


def preprocess(frames, device):
    """uint8 NHWC -> float NCHW normalized."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device)


@torch.no_grad()
def encode_lewm(encoder, projector, frames_tensor, batch_size=64):
    embs = []
    for i in range(0, len(frames_tensor), batch_size):
        batch = frames_tensor[i:i+batch_size]
        out = encoder(batch, interpolate_pos_encoding=True)
        cls_token = out.last_hidden_state[:, 0]
        emb = projector(cls_token)
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def encode_dinov2(model, frames_tensor, batch_size=64):
    embs = []
    for i in range(0, len(frames_tensor), batch_size):
        batch = frames_tensor[i:i+batch_size]
        emb = model(batch)
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0)


def find_nn(embs, query_indices, k=5):
    """Find k nearest neighbors for each query."""
    embs_norm = F.normalize(embs, dim=-1)
    queries = embs_norm[query_indices]
    sims = queries @ embs_norm.T  # (Q, N)
    # Zero out self-similarity
    for i, qi in enumerate(query_indices):
        sims[i, qi] = -1.0
    topk = sims.topk(k, dim=-1)
    return topk.indices, topk.values


def make_grid(frames, query_indices, nn_indices, nn_scores, out_path, title=""):
    """Save a grid: each row = query + top-5 neighbors with similarity scores."""
    n_queries = len(query_indices)
    k = nn_indices.shape[1]
    cols = 1 + k
    cell_h, cell_w = 120, 120
    pad = 2
    header = 25

    W = cols * (cell_w + pad) + pad
    H = n_queries * (cell_h + pad + header) + pad
    canvas = Image.new('RGB', (W, H), (255, 255, 255))

    for row, qi in enumerate(query_indices):
        y = row * (cell_h + pad + header) + pad
        # Query frame (blue border)
        img = Image.fromarray(frames[qi]).resize((cell_w, cell_h))
        canvas.paste(img, (pad, y + header))

        # Neighbors
        for col in range(k):
            ni = nn_indices[row, col].item()
            score = nn_scores[row, col].item()
            img = Image.fromarray(frames[ni]).resize((cell_w, cell_h))
            x = (col + 1) * (cell_w + pad) + pad
            canvas.paste(img, (x, y + header))

    canvas.save(out_path)
    print(f"Saved grid to {out_path} ({n_queries} queries × {k} neighbors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='/root/.stable_worldmodel/lewm_epoch_100_object.ckpt')
    parser.add_argument('--data', default='/data/lewm_data/aigen_val.h5')
    parser.add_argument('--max-frames', type=int, default=2000)
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--n-queries', type=int, default=8)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--dinov2', action='store_true', help='Also run DINOv2-small baseline')
    parser.add_argument('--out-dir', default='/tmp/lewm_eval')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load frames
    print(f"Loading frames from {args.data}...")
    frames, indices = load_frames(args.data, args.max_frames, args.stride)
    print(f"Loaded {len(frames)} frames")
    frames_tensor = preprocess(frames, device)

    # Random query indices
    rng = np.random.RandomState(args.seed)
    query_idx = rng.choice(len(frames), size=args.n_queries, replace=False)

    # LeWM evaluation
    print("Loading LeWM encoder...")
    encoder, projector = load_lewm_encoder(args.ckpt, device)
    print("Encoding with LeWM...")
    lewm_embs = encode_lewm(encoder, projector, frames_tensor)
    print(f"LeWM embeddings: {lewm_embs.shape}, mean={lewm_embs.mean():.3f}, std={lewm_embs.std():.3f}")

    nn_idx, nn_scores = find_nn(lewm_embs, query_idx, args.k)
    make_grid(frames, query_idx, nn_idx, nn_scores, out_dir / "lewm_nn.jpg", "LeWM")

    # DINOv2 baseline
    if args.dinov2:
        print("Loading DINOv2-small...")
        dino = load_dinov2_small(device)
        print("Encoding with DINOv2...")
        dino_embs = encode_dinov2(dino, frames_tensor)
        print(f"DINOv2 embeddings: {dino_embs.shape}, mean={dino_embs.mean():.3f}, std={dino_embs.std():.3f}")

        nn_idx_d, nn_scores_d = find_nn(dino_embs, query_idx, args.k)
        make_grid(frames, query_idx, nn_idx_d, nn_scores_d, out_dir / "dinov2_nn.jpg", "DINOv2")

    # Embedding stats
    print("\n=== Embedding Statistics ===")
    print(f"LeWM: dim={lewm_embs.shape[1]}, variance={lewm_embs.var(0).mean():.4f}")
    if args.dinov2:
        print(f"DINOv2: dim={dino_embs.shape[1]}, variance={dino_embs.var(0).mean():.4f}")


if __name__ == '__main__':
    main()
