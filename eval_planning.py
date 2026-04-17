"""Offline goal-conditioned planning evaluation for LeWM.

Tests whether the model's latent space supports goal-directed action optimization.
Same principle as the paper's MPC eval (Section 5.1) but on recorded trajectories:

1. Pick start context and goal frame from a trajectory (configurable gap)
2. Use CEM to optimize action sequence in latent space toward goal embedding
3. Compare optimized actions to ground-truth actions

Usage:
    python eval_planning.py \
        --ckpt /data/lewm_checkpoints/tiny_depth_epoch100_7dlgiyuf.ckpt \
        --model-type tiny \
        --val-data /data/tartanground_lewm/tartanground.h5 \
        --goal-distance 20 \
        --output planning_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from run_surprise_eval_predictor import load_full_model, IMAGENET_MEAN, IMAGENET_STD


@torch.no_grad()
def encode_frames(model, pixels, device, batch_size=64):
    """Encode pixel frames to latent embeddings. Returns (N, D) tensor."""
    embs = []
    for i in range(0, len(pixels), batch_size):
        batch = torch.from_numpy(pixels[i:i + batch_size]).permute(0, 3, 1, 2).float() / 255.0
        batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
        batch = batch.to(device)
        out = model.encoder(batch, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]
        emb = model.projector(cls)
        embs.append(emb)
    return torch.cat(embs, dim=0)


@torch.no_grad()
def latent_rollout(model, start_embs, action_seqs, history_size=3):
    """Roll out action sequences through the predictor in latent space.

    Args:
        model: JEPA model
        start_embs: (H, D) context embeddings (H = history_size)
        action_seqs: (S, T, act_dim) candidate action sequences
        history_size: context window for predictor

    Returns:
        final_embs: (S, D) final predicted embedding per sequence
        all_embs: (S, T, D) full predicted trajectory per sequence
    """
    S, T, act_dim = action_seqs.shape
    D = start_embs.shape[-1]
    device = start_embs.device

    # Expand start context for all samples: (S, H, D)
    emb = start_embs.unsqueeze(0).expand(S, -1, -1).clone()

    predicted = []
    for t in range(T):
        # Truncate to last history_size
        ctx_emb = emb[:, -history_size:]  # (S, H, D)
        ctx_act = action_seqs[:, max(0, t - history_size + 1):t + 1]  # (S, <=H, act_dim)

        # Pad actions if we don't have enough history yet
        if ctx_act.shape[1] < ctx_emb.shape[1]:
            pad = torch.zeros(S, ctx_emb.shape[1] - ctx_act.shape[1], act_dim, device=device)
            ctx_act = torch.cat([pad, ctx_act], dim=1)

        act_emb = model.action_encoder(ctx_act)  # (S, H, D)
        pred = model.predict(ctx_emb, act_emb)  # (S, H, D)
        next_emb = pred[:, -1:]  # (S, 1, D)

        predicted.append(next_emb)
        emb = torch.cat([emb, next_emb], dim=1)

    all_embs = torch.cat(predicted, dim=1)  # (S, T, D)
    final_embs = all_embs[:, -1]  # (S, D)
    return final_embs, all_embs


def cem_optimize(model, start_embs, goal_emb, action_dim, horizon,
                 history_size=3, num_samples=300, n_iterations=30,
                 elite_k=30, device='cpu'):
    """CEM optimization to find action sequence reaching goal in latent space.

    Returns:
        best_actions: (horizon, action_dim) best action sequence found
        best_trajectory: (horizon, D) predicted latent trajectory
        best_score: float, cosine similarity to goal
    """
    # Initialize action distribution
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones(horizon, action_dim, device=device)

    best_score = -float('inf')
    best_actions = None
    best_trajectory = None

    for iteration in range(n_iterations):
        # Sample action candidates
        noise = torch.randn(num_samples, horizon, action_dim, device=device)
        candidates = mean.unsqueeze(0) + noise * std.unsqueeze(0)  # (S, T, act_dim)

        # Rollout each candidate
        final_embs, all_embs = latent_rollout(
            model, start_embs, candidates, history_size)

        # Score by cosine similarity to goal
        scores = F.cosine_similarity(
            final_embs, goal_emb.unsqueeze(0).expand(num_samples, -1), dim=-1)  # (S,)

        # Track global best
        iter_best_idx = scores.argmax()
        if scores[iter_best_idx] > best_score:
            best_score = scores[iter_best_idx].item()
            best_actions = candidates[iter_best_idx].clone()
            best_trajectory = all_embs[iter_best_idx].clone()

        # Elite selection and refit
        elite_idx = scores.topk(elite_k).indices
        elite_actions = candidates[elite_idx]  # (K, T, act_dim)
        mean = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0).clamp(min=1e-4)

    return best_actions, best_trajectory, best_score


def prepare_gt_actions(raw_actions, start_idx, horizon, frameskip=2):
    """Prepare ground-truth actions with frameskip concatenation."""
    T = len(raw_actions)
    act_list = []
    for t in range(horizon):
        idx = start_idx + t
        parts = []
        for fs in range(frameskip):
            i = idx * frameskip + fs if idx * frameskip + fs < T else T - 1
            parts.append(raw_actions[i])
        act_list.append(np.concatenate(parts))
    return np.array(act_list, dtype=np.float32)


def run_trial(model, ep_pixels, ep_actions, start_idx, goal_distance,
              history_size, device, cem_kwargs):
    """Run one planning trial. Returns dict of metrics."""
    goal_idx = start_idx + goal_distance

    # Encode start context and goal
    ctx_pixels = ep_pixels[start_idx:start_idx + history_size]
    goal_pixel = ep_pixels[goal_idx:goal_idx + 1]
    # Also encode intermediate GT frames for trajectory comparison
    gt_pixels = ep_pixels[start_idx + history_size:goal_idx + 1]

    all_needed = np.concatenate([ctx_pixels, gt_pixels, goal_pixel], axis=0)
    all_embs = encode_frames(model, all_needed, device)

    start_embs = all_embs[:history_size]  # (H, D)
    gt_embs = all_embs[history_size:-1]   # (goal_distance - history_size, D)
    goal_emb = all_embs[-1]               # (D,)

    # Prepare GT actions
    act_dim = cem_kwargs.get('action_dim', 4)
    frameskip = 2
    gt_actions = prepare_gt_actions(ep_actions, start_idx + history_size - 1,
                                    goal_distance, frameskip)
    gt_actions_t = torch.from_numpy(gt_actions).float().to(device)

    # CEM optimization
    best_actions, best_trajectory, goal_score = cem_optimize(
        model, start_embs, goal_emb,
        action_dim=act_dim, horizon=goal_distance,
        history_size=history_size, device=device,
        **{k: v for k, v in cem_kwargs.items() if k != 'action_dim'})

    # Metrics
    # 1. Goal cosine sim (already computed)

    # 2. Action similarity to GT
    min_len = min(len(best_actions), len(gt_actions_t))
    action_cosine = F.cosine_similarity(
        best_actions[:min_len].reshape(1, -1),
        gt_actions_t[:min_len].reshape(1, -1)).item()
    action_mse = F.mse_loss(best_actions[:min_len], gt_actions_t[:min_len]).item()

    # 3. Intermediate trajectory similarity
    traj_len = min(best_trajectory.shape[0], gt_embs.shape[0])
    if traj_len > 0:
        traj_cosine = F.cosine_similarity(
            best_trajectory[:traj_len], gt_embs[:traj_len], dim=-1).mean().item()
    else:
        traj_cosine = 0.0

    return {
        'goal_cosine_sim': goal_score,
        'action_cosine_sim': action_cosine,
        'action_mse': action_mse,
        'trajectory_cosine_sim': traj_cosine,
    }


def main():
    parser = argparse.ArgumentParser(description='Offline goal-conditioned planning eval')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--model-type', required=True, choices=['tiny', 'small', 'dinov2'])
    parser.add_argument('--val-data', required=True)
    parser.add_argument('--goal-distance', type=int, default=20,
                        help='Steps between start and goal frame')
    parser.add_argument('--trials-per-ep', type=int, default=5)
    parser.add_argument('--cem-samples', type=int, default=300)
    parser.add_argument('--cem-iterations', type=int, default=30)
    parser.add_argument('--cem-elite-k', type=int, default=30)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model ({args.model_type})...")
    model, embed_dim, hist_size, act_dim = load_full_model(args.ckpt, args.model_type, device)
    print(f"  embed_dim={embed_dim}, history={hist_size}, act_dim={act_dim}")

    # Load data
    f = h5py.File(args.val_data, 'r')
    all_pixels = f['pixels'][:]
    all_actions = f['action'][:]
    ep_lens = f['ep_len'][:]
    ep_offsets = f['ep_offset'][:]
    f.close()
    print(f"Loaded {len(ep_lens)} episodes, {len(all_pixels)} total frames")

    cem_kwargs = {
        'action_dim': act_dim,
        'num_samples': args.cem_samples,
        'n_iterations': args.cem_iterations,
        'elite_k': args.cem_elite_k,
    }

    # Compute random baseline: average cosine sim between random frame pairs
    # This measures latent space compactness — higher = more compressed space
    print("Computing random baseline cosine similarity...")
    n_baseline_pairs = 500
    rand_idx = np.random.choice(len(all_pixels), size=n_baseline_pairs * 2, replace=True)
    baseline_embs = encode_frames(model, all_pixels[rand_idx], device)
    embs_a = baseline_embs[:n_baseline_pairs]
    embs_b = baseline_embs[n_baseline_pairs:]
    random_cosine_sim = F.cosine_similarity(embs_a, embs_b, dim=-1).mean().item()
    print(f"  Random pair cosine sim: {random_cosine_sim:.4f}")

    # Run trials
    all_metrics = defaultdict(list)
    total_trials = 0

    for ei in range(len(ep_lens)):
        offset = ep_offsets[ei]
        length = ep_lens[ei]
        ep_pixels = all_pixels[offset:offset + length]
        ep_actions = all_actions[offset:offset + length]

        min_len_needed = hist_size + args.goal_distance + 1
        if length < min_len_needed:
            print(f"  Episode {ei}: skipped (too short: {length} < {min_len_needed})")
            continue

        max_start = length - args.goal_distance - hist_size
        starts = np.random.choice(
            range(0, max_start, max(1, max_start // 20)),
            size=min(args.trials_per_ep, max(1, max_start // 5)),
            replace=False)

        for start in starts:
            metrics = run_trial(
                model, ep_pixels, ep_actions, start, args.goal_distance,
                hist_size, device, cem_kwargs)

            for k, v in metrics.items():
                all_metrics[k].append(v)
            total_trials += 1

        print(f"  Episode {ei}: {len(starts)} trials "
              f"(goal_cos={np.mean(all_metrics['goal_cosine_sim'][-len(starts):]):.4f})")

    # Summary
    print(f"\n{'='*60}")
    print(f"OFFLINE PLANNING EVAL (goal_distance={args.goal_distance})")
    print(f"{'='*60}")

    summary = {}
    for k in sorted(all_metrics.keys()):
        vals = np.array(all_metrics[k])
        print(f"  {k:30s}: mean={vals.mean():.4f}  std={vals.std():.4f}")
        summary[k] = {'mean': float(vals.mean()), 'std': float(vals.std())}

    print(f"\n  Total trials: {total_trials}")

    output = {
        'model': {'checkpoint': str(args.ckpt), 'type': args.model_type},
        'random_baseline_cosine_sim': random_cosine_sim,
        'config': {
            'goal_distance': args.goal_distance,
            'history_size': hist_size,
            'act_dim': act_dim,
            'cem_samples': args.cem_samples,
            'cem_iterations': args.cem_iterations,
            'cem_elite_k': args.cem_elite_k,
            'val_data': args.val_data,
            'seed': args.seed,
        },
        'summary': summary,
        'total_trials': total_trials,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
