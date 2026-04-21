import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VOProbe(nn.Module):
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


def preprocess(frames, device):
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device)


def load_dinov2_small(device):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
    return model.to(device).eval()


def build_pairs(h5_path):
    with h5py.File(h5_path, 'r') as f:
        pixels = f['pixels'][:]
        actions = f['action'][:]
        ep_len = f['ep_len'][:]
        ep_offset = f['ep_offset'][:]

    pair_idx = []
    for off, length in zip(ep_offset, ep_len):
        for t in range(length - 1):
            pair_idx.append((off + t, off + t + 1))
    pair_idx = np.array(pair_idx)
    actions_tensor = torch.from_numpy(actions).float()
    return pixels, pair_idx, actions_tensor


@torch.no_grad()
def encode_all(encoder_fn, frames_np, device, batch_size=32):
    embs = []
    for i in range(0, len(frames_np), batch_size):
        batch_np = frames_np[i:i+batch_size]
        batch = preprocess(batch_np, device)
        emb = encoder_fn(batch)
        embs.append(emb.cpu())
        del batch
    torch.cuda.empty_cache()
    return torch.cat(embs, dim=0)


def train_probe(embs, pair_idx, actions, emb_dim, n_epochs=50, lr=1e-3, val_split=0.1, device='cpu'):
    n = len(pair_idx)
    perm = np.random.permutation(n)
    n_val = max(1, int(n * val_split))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    probe = VOProbe(emb_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    embs = embs.to(device)
    actions = actions.to(device)

    best_val = float('inf')
    best_per_dim = None
    for epoch in range(n_epochs):
        probe.train()
        np.random.shuffle(train_idx)
        bs = 512
        for i in range(0, len(train_idx), bs):
            batch = train_idx[i:i+bs]
            idx_t = pair_idx[batch, 0]
            idx_t1 = pair_idx[batch, 1]
            pred = probe(embs[idx_t], embs[idx_t1])
            target = actions[idx_t]
            loss = F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            idx_t = pair_idx[val_idx, 0]
            idx_t1 = pair_idx[val_idx, 1]
            pred = probe(embs[idx_t], embs[idx_t1])
            target = actions[idx_t]
            val_loss = F.mse_loss(pred, target).item()
            mse_per_dim = (pred - target).pow(2).mean(0)
        if val_loss < best_val:
            best_val = val_loss
            best_per_dim = mse_per_dim.clone()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: val={val_loss:.6f}")

    return best_val, best_per_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    pixels, pair_idx, actions = build_pairs(args.train_data)
    print(f"Frames: {len(pixels)}, Pairs: {len(pair_idx)}")

    mean_action = actions[pair_idx[:, 0]].mean(0)
    baseline_mse = (actions[pair_idx[:, 0]] - mean_action).pow(2).mean().item()
    print(f"Baseline (predict mean): MSE = {baseline_mse:.6f}")

    dino = load_dinov2_small(device)
    dino_embs = encode_all(lambda b: dino(b), pixels, device)
    print(f"Embeddings: {dino_embs.shape}")
    best, per_dim = train_probe(dino_embs, pair_idx, actions, dino_embs.shape[1], n_epochs=args.epochs, device=device)
    print(f"Best val MSE: {best:.6f} (dx={per_dim[0]:.6f}, dy={per_dim[1]:.6f})")


if __name__ == '__main__':
    main()
