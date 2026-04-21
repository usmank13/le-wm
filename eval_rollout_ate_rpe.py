"""Predictor-rollout trajectory metric on datasets with local-frame action semantics.

This metric evaluates whether the JEPA predictor can roll out latent trajectories that
remain consistent enough with observed future embeddings to support motion decoding.
Pipeline:
1. Encode an episode with a frozen encoder/projector.
2. Roll out the predictor autoregressively from an initial latent context using GT actions.
3. Decode local planar motion from predicted-vs-predicted consecutive latent pairs using a
   motion probe trained on real consecutive pairs from the training set.
4. Integrate decoded local motion into world XY using GT heading and compute ATE/RPE.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from eval_ate_rpe import (
    EpisodeSlice,
    VOProbe,
    compute_ate,
    compute_rpe,
    integrate_actions,
    load_h5_episodes,
    summarize,
    train_probe,
)
from run_surprise_eval_predictor import load_full_model, IMAGENET_MEAN, IMAGENET_STD


def encode_pixels(model, pixels, device, batch_size=64):
    embs = []
    for i in range(0, len(pixels), batch_size):
        batch = torch.from_numpy(pixels[i:i + batch_size]).permute(0, 3, 1, 2).float() / 255.0
        batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
        batch = batch.to(device)
        out = model.encoder(batch, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]
        embs.append(model.projector(cls).cpu())
    return torch.cat(embs, dim=0)


def build_probe_training_data(model, episodes, device):
    emb_a, emb_b, targets = [], [], []
    for ep in episodes:
        embs = encode_pixels(model, ep.pixels, device)
        emb_a.append(embs[:-1])
        emb_b.append(embs[1:])
        targets.append(torch.from_numpy(ep.action[:-1]).float())
    return torch.cat(emb_a, dim=0), torch.cat(emb_b, dim=0), torch.cat(targets, dim=0).numpy()


@torch.no_grad()
def rollout_episode_embeddings(model, episode, device, history_size):
    embs = encode_pixels(model, episode.pixels, device).to(device)
    actions = torch.from_numpy(episode.action[:-1]).float().to(device)

    context = embs[:history_size].clone().unsqueeze(0)
    predicted = [context[0]]

    for t in range(history_size, len(embs)):
        act_hist = actions[t - history_size:t].unsqueeze(0)
        act_emb = model.action_encoder(act_hist)
        pred = model.predict(context[:, -history_size:], act_emb)
        next_emb = pred[:, -1:]
        context = torch.cat([context, next_emb], dim=1)
        predicted.append(next_emb[0])

    predicted = torch.cat(predicted, dim=0)
    return embs.cpu(), predicted.cpu()


@torch.no_grad()
def decode_rollout_actions(probe, predicted_embs, device):
    pred = probe(predicted_embs[:-1].to(device), predicted_embs[1:].to(device))
    return pred.cpu().numpy()


def evaluate_episodes(model, probe, episodes, device, history_size, rpe_delta=1):
    metrics = []
    for ep in episodes:
        _, pred_embs = rollout_episode_embeddings(model, ep, device, history_size)
        pred_actions = decode_rollout_actions(probe, pred_embs, device)
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


def validate_local_action_semantics(episodes):
    check_ep = episodes[0]
    gt_xy = check_ep.proprio[:, :2]
    gt_heading = check_ep.proprio[:-1, 2]
    gt_world = gt_xy[1:] - gt_xy[:-1]
    recon_local = []
    for d, th in zip(gt_world, gt_heading):
        c, s = np.cos(th), np.sin(th)
        rot_inv = np.array([[c, s], [-s, c]], dtype=np.float32)
        recon_local.append(rot_inv @ d)
    recon_local = np.stack(recon_local)
    mismatch = np.linalg.norm(recon_local - check_ep.action[:-1], axis=1).mean()
    if mismatch > 1e-3:
        raise ValueError(f'action/proprio mismatch too large for rollout trajectory metric: mean l2={mismatch:.6f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--eval-data', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--model-type', choices=['tiny', 'small', 'dinov2'], required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--rpe-delta', type=int, default=1)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_episodes = load_h5_episodes(args.train_data)
    eval_episodes = load_h5_episodes(args.eval_data)
    validate_local_action_semantics(eval_episodes)

    model, _, history_size, _ = load_full_model(args.ckpt, args.model_type, device)
    emb_a, emb_b, targets = build_probe_training_data(model, train_episodes, device)
    probe, best_val = train_probe(emb_a, emb_b, targets, n_epochs=args.epochs, device=device, seed=args.seed)
    metrics = evaluate_episodes(model, probe, eval_episodes, device, history_size, rpe_delta=args.rpe_delta)
    summary = summarize(metrics)

    output = {
        'model_type': args.model_type,
        'ckpt': args.ckpt,
        'train_data': args.train_data,
        'eval_data': args.eval_data,
        'probe_val_mse': best_val,
        'history_size': history_size,
        'rpe_delta': args.rpe_delta,
        'summary': summary,
        'per_episode': metrics,
    }

    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
