"""Visual odometry probe on frozen LeWM vs DINOv2 encoders.

Trains a small MLP to predict ego-motion (dx, dy) from pairs of consecutive
frame embeddings. Compares LeWM, DINOv2, and random-init baselines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from pathlib import Path
import argparse

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VOProbe(nn.Module):
    """MLP probe: concat(emb_t, emb_t+1) -> (dx, dy)"""
    def __init__(self, emb_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, emb_t, emb_t1):
        return self.net(torch.cat([emb_t, emb_t1], dim=-1))


def load_lewm_encoder(ckpt_path, device):
    from model_loading import load_encoder_projector
    return load_encoder_projector(ckpt_path, device, freeze=True)


def load_dinov2_small(device):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
    return model.to(device).eval()


def load_random_vit(device):
    """Random init ViT-small as baseline."""
    from transformers import ViTConfig, ViTModel
    config = ViTConfig(hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
                       intermediate_size=1536, patch_size=14, image_size=224)
    model = ViTModel(config)
    return model.to(device).eval()


def preprocess(frames, device):
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device)


@torch.no_grad()
def encode_all(encoder_fn, frames_np, device, batch_size=32):
    """Encode frames in batches, keeping raw frames on CPU."""
    embs = []
    for i in range(0, len(frames_np), batch_size):
        batch_np = frames_np[i:i+batch_size]
        batch = preprocess(batch_np, device)
        emb = encoder_fn(batch)
        embs.append(emb.cpu())
        del batch
    torch.cuda.empty_cache()
    return torch.cat(embs, dim=0)


def build_pairs(h5_path, device):
    """Load consecutive frame pairs and their actions within episodes."""
    with h5py.File(h5_path, 'r') as f:
        pixels = f['pixels'][:]  # (N, H, W, C)
        actions = f['action'][:]  # (N, 2)
        ep_len = f['ep_len'][:]
        ep_offset = f['ep_offset'][:]

    # Build valid pair indices (within episodes, skip episode boundaries)
    pair_idx = []
    for i, (off, length) in enumerate(zip(ep_offset, ep_len)):
        for t in range(length - 1):
            pair_idx.append((off + t, off + t + 1))

    pair_idx = np.array(pair_idx)
    actions_tensor = torch.from_numpy(actions).float()

    return pixels, pair_idx, actions_tensor


def train_probe(embs, pair_idx, actions, emb_dim, n_epochs=50, lr=1e-3, val_split=0.1, device='cpu'):
    """Train VO probe and return val MSE."""
    n = len(pair_idx)
    perm = np.random.permutation(n)
    n_val = max(1, int(n * val_split))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    probe = VOProbe(emb_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    embs = embs.to(device)
    actions = actions.to(device)

    best_val = float('inf')
    for epoch in range(n_epochs):
        # Train
        probe.train()
        np.random.shuffle(train_idx)
        train_loss = 0
        bs = 512
        for i in range(0, len(train_idx), bs):
            batch = train_idx[i:i+bs]
            idx_t = pair_idx[batch, 0]
            idx_t1 = pair_idx[batch, 1]
            pred = probe(embs[idx_t], embs[idx_t1])
            target = actions[idx_t]  # action at t causes transition t->t+1
            loss = F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(train_idx)

        # Val
        probe.eval()
        with torch.no_grad():
            idx_t = pair_idx[val_idx, 0]
            idx_t1 = pair_idx[val_idx, 1]
            pred = probe(embs[idx_t], embs[idx_t1])
            target = actions[idx_t]
            val_loss = F.mse_loss(pred, target).item()

        if val_loss < best_val:
            best_val = val_loss

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: train={train_loss:.6f}, val={val_loss:.6f}")

    # Per-dimension breakdown
    probe.eval()
    with torch.no_grad():
        idx_t = pair_idx[val_idx, 0]
        idx_t1 = pair_idx[val_idx, 1]
        pred = probe(embs[idx_t], embs[idx_t1])
        target = actions[idx_t]
        mse_per_dim = (pred - target).pow(2).mean(0)

    return best_val, mse_per_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='/root/.stable_worldmodel/lewm_epoch_100_object.ckpt')
    parser.add_argument('--train-data', default='/data/lewm_data/aigen_train.h5')
    parser.add_argument('--val-data', default='/data/lewm_data/aigen_val.h5')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--dinov2', action='store_true')
    parser.add_argument('--random', action='store_true', help='Also test random init ViT')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("Loading frame pairs...")
    pixels, pair_idx, actions = build_pairs(args.train_data, device)
    print(f"Frames: {len(pixels)}, Pairs: {len(pair_idx)}")
    print(f"Action stats: mean={actions.mean(0).tolist()}, std={actions.std(0).tolist()}")

    # Baseline: predict mean action
    mean_action = actions[pair_idx[:, 0]].mean(0)
    baseline_mse = (actions[pair_idx[:, 0]] - mean_action).pow(2).mean().item()
    print(f"\nBaseline (predict mean): MSE = {baseline_mse:.6f}")

    results = {}

    # LeWM
    print("\n=== LeWM ===")
    encoder, projector = load_lewm_encoder(args.ckpt, device)
    def lewm_fn(batch):
        out = encoder(batch, interpolate_pos_encoding=True)
        return projector(out.last_hidden_state[:, 0])
    lewm_embs = encode_all(lewm_fn, pixels, device)
    print(f"Embeddings: {lewm_embs.shape}")
    best, per_dim = train_probe(lewm_embs, pair_idx, actions, lewm_embs.shape[1],
                                 n_epochs=args.epochs, device=device)
    results['LeWM'] = best
    print(f"Best val MSE: {best:.6f} (dx={per_dim[0]:.6f}, dy={per_dim[1]:.6f})")
    del encoder, projector
    torch.cuda.empty_cache()

    # DINOv2
    if args.dinov2:
        print("\n=== DINOv2-small ===")
        dino = load_dinov2_small(device)
        dino_embs = encode_all(lambda b: dino(b), pixels, device)
        print(f"Embeddings: {dino_embs.shape}")
        best, per_dim = train_probe(dino_embs, pair_idx, actions, dino_embs.shape[1],
                                     n_epochs=args.epochs, device=device)
        results['DINOv2'] = best
        print(f"Best val MSE: {best:.6f} (dx={per_dim[0]:.6f}, dy={per_dim[1]:.6f})")
        del dino
        torch.cuda.empty_cache()

    # Random ViT
    if args.random:
        print("\n=== Random ViT-small ===")
        rand_vit = load_random_vit(device)
        def rand_fn(batch):
            out = rand_vit(batch, interpolate_pos_encoding=True)
            return out.last_hidden_state[:, 0]
        rand_embs = encode_all(rand_fn, pixels, device)
        print(f"Embeddings: {rand_embs.shape}")
        best, per_dim = train_probe(rand_embs, pair_idx, actions, rand_embs.shape[1],
                                     n_epochs=args.epochs, device=device)
        results['Random'] = best
        print(f"Best val MSE: {best:.6f} (dx={per_dim[0]:.6f}, dy={per_dim[1]:.6f})")

    # Summary
    print("\n" + "="*50)
    print("VISUAL ODOMETRY PROBE RESULTS")
    print("="*50)
    print(f"{'Method':<15} {'Val MSE':>10} {'vs Baseline':>12}")
    print("-"*40)
    print(f"{'Mean baseline':<15} {baseline_mse:>10.6f} {'—':>12}")
    for name, mse in sorted(results.items(), key=lambda x: x[1]):
        ratio = mse / baseline_mse
        print(f"{name:<15} {mse:>10.6f} {ratio:>11.1%}")


if __name__ == '__main__':
    main()
