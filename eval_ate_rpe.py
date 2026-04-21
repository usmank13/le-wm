"""ATE/RPE evaluation via latent motion probe trajectory reconstruction.

Pipeline:
1. Encode consecutive frames with a frozen representation.
2. Train a probe to predict relative motion from embedding pairs.
3. Integrate predicted local-frame motion into 2D trajectories using GT heading.
4. Compare reconstructed trajectories against GT using ATE and RPE.

This is a pragmatic representation-centric metric, not a predictor-native rollout metric.
The script assumes `action[:, :2]` is the robot-frame planar displacement that is
consistent with `proprio[:, :3] = (x, y, heading)`. This is true for the current
TartanGround export and should be verified before using other datasets.
"""

import argparse
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from eval_vo import IMAGENET_MEAN, IMAGENET_STD, VOProbe
from eval_vo import load_lewm_encoder, load_dinov2_small
from eval_common import resolve_model_spec, EvalMetadata, make_output, write_json, assert_local_action_semantics


@dataclass
class EpisodeSlice:
    pixels: np.ndarray
    proprio: np.ndarray
    action: np.ndarray


def preprocess(frames, device):
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device)


@torch.no_grad()
def encode_all(encoder_fn, frames_np, device, batch_size=32):
    embs = []
    for i in range(0, len(frames_np), batch_size):
        batch_np = frames_np[i:i + batch_size]
        batch = preprocess(batch_np, device)
        emb = encoder_fn(batch)
        embs.append(emb.cpu())
        del batch
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return torch.cat(embs, dim=0)


def load_h5_episodes(path):
    episodes = []
    with h5py.File(path, 'r') as f:
        pixels = f['pixels'][:]
        proprio = f['proprio'][:]
        action = f['action'][:]
        ep_len = f['ep_len'][:]
        ep_offset = f['ep_offset'][:]
    for off, length in zip(ep_offset, ep_len):
        episodes.append(EpisodeSlice(
            pixels=pixels[off:off + length],
            proprio=proprio[off:off + length],
            action=action[off:off + length],
        ))
    return episodes


def build_training_pairs(episodes):
    pair_pixels = []
    pair_targets = []
    for ep in episodes:
        for t in range(len(ep.pixels) - 1):
            pair_pixels.append((ep.pixels[t], ep.pixels[t + 1]))
            pair_targets.append(ep.action[t])
    return pair_pixels, np.asarray(pair_targets, dtype=np.float32)


def encode_pair_dataset(pair_pixels, encoder_fn, device, batch_size=32):
    first = np.stack([p[0] for p in pair_pixels], axis=0)
    second = np.stack([p[1] for p in pair_pixels], axis=0)
    emb_a = encode_all(encoder_fn, first, device, batch_size=batch_size)
    emb_b = encode_all(encoder_fn, second, device, batch_size=batch_size)
    return emb_a, emb_b


def train_probe(emb_a, emb_b, targets, n_epochs=50, lr=1e-3, val_split=0.1, device='cpu', seed=42):
    rng = np.random.default_rng(seed)
    n = len(targets)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_split))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    probe = VOProbe(emb_a.shape[1]).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    emb_a = emb_a.to(device)
    emb_b = emb_b.to(device)
    targets = torch.from_numpy(targets).float().to(device)

    best_val = float('inf')
    best_state = None
    for epoch in range(n_epochs):
        probe.train()
        rng.shuffle(train_idx)
        bs = 512
        for i in range(0, len(train_idx), bs):
            batch = train_idx[i:i + bs]
            pred = probe(emb_a[batch], emb_b[batch])
            loss = F.mse_loss(pred, targets[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            pred = probe(emb_a[val_idx], emb_b[val_idx])
            val_loss = F.mse_loss(pred, targets[val_idx]).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    return probe, best_val


@torch.no_grad()
def predict_episode_actions(probe, encoder_fn, episode, device):
    embs = encode_all(encoder_fn, episode.pixels, device)
    pred = probe(embs[:-1].to(device), embs[1:].to(device)).cpu().numpy()
    return pred


def local_to_world(delta_local, heading):
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return rot @ delta_local.astype(np.float32)


def integrate_actions(start_xy, headings, rel_actions):
    traj = [np.asarray(start_xy, dtype=np.float32)]
    curr = traj[0].copy()
    for delta_local, heading in zip(rel_actions, headings):
        curr = curr + local_to_world(delta_local, heading)
        traj.append(curr.copy())
    return np.stack(traj, axis=0)


def compute_ate(gt_xy, pred_xy):
    err = np.linalg.norm(pred_xy - gt_xy, axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def compute_rpe(gt_xy, pred_xy, delta=1):
    if len(gt_xy) <= delta:
        return None
    gt_rel = gt_xy[delta:] - gt_xy[:-delta]
    pred_rel = pred_xy[delta:] - pred_xy[:-delta]
    err = np.linalg.norm(pred_rel - gt_rel, axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def evaluate_episodes(probe, encoder_fn, episodes, device, rpe_delta=1):
    metrics = []
    for ep in episodes:
        pred_actions = predict_episode_actions(probe, encoder_fn, ep, device)
        gt_xy = ep.proprio[:, :2].astype(np.float32)
        gt_heading = ep.proprio[:-1, 2].astype(np.float32)
        pred_xy = integrate_actions(gt_xy[0], gt_heading[:len(pred_actions)], pred_actions)
        gt_xy = gt_xy[:len(pred_xy)]
        metrics.append({
            'ate': compute_ate(gt_xy, pred_xy),
            'rpe': compute_rpe(gt_xy, pred_xy, delta=rpe_delta),
            'steps': int(len(pred_actions)),
        })
    return metrics


def summarize(metrics):
    ate = np.array([m['ate'] for m in metrics], dtype=np.float32)
    rpe = np.array([m['rpe'] for m in metrics if m['rpe'] is not None], dtype=np.float32)
    steps = np.array([m['steps'] for m in metrics], dtype=np.int32)
    return {
        'ate': {'mean': float(ate.mean()), 'std': float(ate.std())},
        'rpe': {'mean': float(rpe.mean()), 'std': float(rpe.std())},
        'steps': {'mean': float(steps.mean()), 'std': float(steps.std())},
        'n_episodes': int(len(metrics)),
    }


def build_encoder(model_type, ckpt_path, device):
    if model_type == 'dinov2':
        model = load_dinov2_small(device)
        return lambda batch: model(batch)
    if model_type == 'lewm':
        encoder, projector = load_lewm_encoder(ckpt_path, device)

        def fn(batch):
            out = encoder(batch, interpolate_pos_encoding=True)
            cls = out.last_hidden_state[:, 0]
            return projector(cls)

        return fn
    raise ValueError(model_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--eval-data', required=True)
    parser.add_argument('--model-label', default=None)
    parser.add_argument('--model-type', choices=['lewm', 'dinov2'], required=True)
    parser.add_argument('--assume-local-action', action='store_true',
                        help='Require action[:,:2] to match local-frame motion from proprio')
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--rpe-delta', type=int, default=1)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    spec = resolve_model_spec(args.model_label, args.model_type, args.ckpt)
    train_episodes = load_h5_episodes(args.train_data)
    eval_episodes = load_h5_episodes(args.eval_data)
    encoder_fn = build_encoder(args.model_type, spec.checkpoint, device)

    mismatch = None
    if args.assume_local_action:
        mismatch = assert_local_action_semantics(args.eval_data)

    pair_pixels, pair_targets = build_training_pairs(train_episodes)
    emb_a, emb_b = encode_pair_dataset(pair_pixels, encoder_fn, device)
    probe, best_val = train_probe(emb_a, emb_b, pair_targets, n_epochs=args.epochs, device=device, seed=args.seed)

    metrics = evaluate_episodes(probe, encoder_fn, eval_episodes, device, rpe_delta=args.rpe_delta)
    summary = summarize(metrics)

    metadata = EvalMetadata(
        eval_name='probe_ate_rpe',
        model_label=spec.label,
        model_type=spec.model_type,
        checkpoint=spec.checkpoint,
        train_data=args.train_data,
        eval_data=args.eval_data,
        seed=args.seed,
        extra={'rpe_delta': args.rpe_delta, 'assume_local_action': args.assume_local_action},
    )
    output = make_output(metadata, summary, per_item=metrics, probe_val_mse=best_val, local_action_mismatch=mismatch)

    write_json(args.output, output)
    print(output['summary'])
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
