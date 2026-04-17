"""Rollout prediction evaluation for LeWM.

Encodes a sequence of frames, then predicts future embeddings autoregressively
using real actions but no new images. Measures how prediction error grows over
the rollout horizon.
"""

import torch
import torch.nn.functional as F
import numpy as np
import h5py
import argparse
from pathlib import Path

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def preprocess(frames_np, device='cpu'):
    x = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device)


def load_model(ckpt_path, device):
    """Load JEPA model from Lightning checkpoint."""
    from train_dinov2 import DINOv2Encoder
    from module import ARPredictor, Embedder, MLP
    from jepa import JEPA

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    # Config from the DINOv2 training: hidden=384, embed=192, action_dim=2, frameskip=2
    hidden_dim = 384
    embed_dim = 192
    effective_act_dim = 2 * 2  # action_dim * frameskip

    encoder = DINOv2Encoder(freeze=True)
    enc_sd = {k.replace('model.encoder.', ''): v for k, v in sd.items() if k.startswith('model.encoder.')}
    encoder.load_state_dict(enc_sd, strict=True)

    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    proj_sd = {k.replace('model.projector.', ''): v for k, v in sd.items() if k.startswith('model.projector.')}
    projector.load_state_dict(proj_sd, strict=True)

    predictor = ARPredictor(
        num_frames=3, input_dim=embed_dim, hidden_dim=hidden_dim,
        output_dim=hidden_dim, depth=6, heads=16, mlp_dim=2048,
        dim_head=64, dropout=0.1, emb_dropout=0.0,
    )
    pred_sd = {k.replace('model.predictor.', ''): v for k, v in sd.items() if k.startswith('model.predictor.')}
    predictor.load_state_dict(pred_sd, strict=True)

    pred_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pp_sd = {k.replace('model.pred_proj.', ''): v for k, v in sd.items() if k.startswith('model.pred_proj.')}
    pred_proj.load_state_dict(pp_sd, strict=True)

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    ae_sd = {k.replace('model.action_encoder.', ''): v for k, v in sd.items() if k.startswith('model.action_encoder.')}
    action_encoder.load_state_dict(ae_sd, strict=True)

    model = JEPA(encoder, predictor, action_encoder, projector, pred_proj)
    return model.to(device).eval()


def load_episodes(h5_path):
    """Load episodes as list of (pixels, actions) tuples."""
    episodes = []
    with h5py.File(h5_path, 'r') as f:
        ep_len = f['ep_len'][:]
        ep_offset = f['ep_offset'][:]
        pixels = f['pixels']
        actions = f['action']

        for i, (off, length) in enumerate(zip(ep_offset, ep_len)):
            ep_pixels = pixels[off:off+length]
            ep_actions = actions[off:off+length]
            episodes.append((np.array(ep_pixels), np.array(ep_actions)))

    return episodes


@torch.no_grad()
def encode_frames(model, frames_np, device, batch_size=32):
    """Encode frames through encoder + projector."""
    embs = []
    for i in range(0, len(frames_np), batch_size):
        batch = preprocess(frames_np[i:i+batch_size], device)
        out = model.encoder(batch, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]
        emb = model.projector(cls)
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def rollout_episode(model, ep_pixels, ep_actions, history_size=3,
                    max_horizon=20, frameskip=2, device='cpu'):
    """Run rollout prediction on one episode.

    Returns: dict with per-step prediction errors.
    """
    n_frames = len(ep_pixels)
    if n_frames < history_size + max_horizon + 1:
        return None

    # Encode all frames (ground truth)
    all_embs = encode_frames(model, ep_pixels, device)  # (T, D)

    # Pick a start point in the middle of the episode
    start = n_frames // 4

    # Context: frames [start, start+history_size)
    ctx_embs = all_embs[start:start+history_size].unsqueeze(0)  # (1, H, D)

    # Actions: need frameskip-concatenated actions
    # With frameskip=2, each "action" fed to model is concat of 2 consecutive raw actions
    raw_actions = torch.from_numpy(ep_actions).float()

    def get_action_block(t):
        """Get frameskip-concatenated action at timestep t."""
        acts = []
        for fs in range(frameskip):
            idx = t * frameskip + fs if t * frameskip + fs < len(raw_actions) else len(raw_actions) - 1
            acts.append(raw_actions[idx])
        return torch.cat(acts, dim=-1)

    # Build context actions
    ctx_actions = torch.stack([get_action_block(start + t) for t in range(history_size)]).unsqueeze(0)  # (1, H, A)

    # Encode context actions
    act_emb = model.action_encoder(ctx_actions.to(device)).cpu()

    # Autoregressive rollout
    errors_cosine = []
    errors_mse = []
    pred_emb = ctx_embs.clone()  # (1, H, D)
    all_act_emb = act_emb.clone()  # (1, H, A_emb)

    for step in range(max_horizon):
        t = start + history_size + step
        if t >= n_frames:
            break

        # Predict next embedding from last `history_size` embeddings + actions
        hs = min(history_size, pred_emb.size(1))
        pred_input = pred_emb[:, -hs:]
        act_input = all_act_emb[:, -hs:]

        next_pred = model.predict(pred_input.to(device), act_input.to(device)).cpu()
        next_pred = next_pred[:, -1:]  # (1, 1, D)

        # Ground truth
        gt_emb = all_embs[t:t+1].unsqueeze(0)  # (1, 1, D)

        # Compute errors
        cosine_sim = F.cosine_similarity(next_pred.squeeze(), gt_emb.squeeze(), dim=0).item()
        mse = (next_pred.squeeze() - gt_emb.squeeze()).pow(2).mean().item()
        errors_cosine.append(cosine_sim)
        errors_mse.append(mse)

        # Append predicted embedding and next action
        pred_emb = torch.cat([pred_emb, next_pred], dim=1)
        next_act = get_action_block(t).unsqueeze(0).unsqueeze(0)
        next_act_emb = model.action_encoder(next_act.to(device)).cpu()
        all_act_emb = torch.cat([all_act_emb, next_act_emb], dim=1)

    return {
        'cosine': errors_cosine,
        'mse': errors_mse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='/root/.stable_worldmodel/lewm_epoch_100_object.ckpt')
    parser.add_argument('--data', default='/data/lewm_data/aigen_val.h5')
    parser.add_argument('--max-horizon', type=int, default=20)
    parser.add_argument('--num-episodes', type=int, default=5)
    parser.add_argument('--frameskip', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading model...")
    model = load_model(args.ckpt, device)

    print(f"Loading episodes from {args.data}...")
    episodes = load_episodes(args.data)
    print(f"Loaded {len(episodes)} episodes")

    # Select episodes
    rng = np.random.RandomState(args.seed)
    ep_indices = rng.choice(len(episodes), size=min(args.num_episodes, len(episodes)), replace=False)

    all_cosine = []
    all_mse = []

    for i, ep_idx in enumerate(ep_indices):
        ep_pixels, ep_actions = episodes[ep_idx]
        print(f"\nEpisode {ep_idx} ({len(ep_pixels)} frames)...")

        result = rollout_episode(
            model, ep_pixels, ep_actions,
            history_size=3, max_horizon=args.max_horizon,
            frameskip=args.frameskip, device=device,
        )

        if result is None:
            print(f"  Skipped (too short)")
            continue

        all_cosine.append(result['cosine'])
        all_mse.append(result['mse'])

        print(f"  Steps: {len(result['cosine'])}")
        print(f"  Cosine sim: step1={result['cosine'][0]:.4f}, step5={result['cosine'][min(4,len(result['cosine'])-1)]:.4f}, step10={result['cosine'][min(9,len(result['cosine'])-1)]:.4f}")
        print(f"  MSE:        step1={result['mse'][0]:.6f}, step5={result['mse'][min(4,len(result['mse'])-1)]:.6f}, step10={result['mse'][min(9,len(result['mse'])-1)]:.6f}")

    # Aggregate
    if not all_cosine:
        print("No valid episodes!")
        return

    # Pad to same length and average
    max_len = max(len(c) for c in all_cosine)
    cosine_matrix = np.full((len(all_cosine), max_len), np.nan)
    mse_matrix = np.full((len(all_mse), max_len), np.nan)
    for i, (c, m) in enumerate(zip(all_cosine, all_mse)):
        cosine_matrix[i, :len(c)] = c
        mse_matrix[i, :len(m)] = m

    mean_cosine = np.nanmean(cosine_matrix, axis=0)
    mean_mse = np.nanmean(mse_matrix, axis=0)

    print("\n" + "=" * 60)
    print("ROLLOUT PREDICTION RESULTS (averaged)")
    print("=" * 60)
    print(f"{'Step':>5} {'Cosine Sim':>12} {'MSE':>12}")
    print("-" * 32)
    for t in range(len(mean_cosine)):
        print(f"{t+1:>5} {mean_cosine[t]:>12.4f} {mean_mse[t]:>12.6f}")


if __name__ == '__main__':
    main()
