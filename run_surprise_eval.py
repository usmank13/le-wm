"""
Run surprise score evaluation using manifest from create_surprise_pairs.py.

Loads each model, computes surprise scores for all plausible/implausible pairs,
reports separation per perturbation type.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_encoder(ckpt_path, model_type, device):
    """Load encoder + projector."""
    from module import MLP
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    if model_type == 'dinov2':
        from train_dinov2 import DINOv2Encoder
        encoder = DINOv2Encoder(freeze=True)
        hidden_dim = 384
    elif model_type == 'tiny':
        import stable_pretraining as spt
        encoder = spt.backbone.utils.vit_hf(
            'tiny', patch_size=14, image_size=224, pretrained=False, use_mask_token=False)
        hidden_dim = 192
    elif model_type == 'small':
        import stable_pretraining as spt
        encoder = spt.backbone.utils.vit_hf(
            'small', patch_size=14, image_size=224, pretrained=False, use_mask_token=False)
        hidden_dim = 384

    embed_dim = 192
    enc_sd = {k.replace('model.encoder.', ''): v for k, v in sd.items() if k.startswith('model.encoder.')}
    encoder.load_state_dict(enc_sd, strict=True)

    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    proj_sd = {k.replace('model.projector.', ''): v for k, v in sd.items() if k.startswith('model.projector.')}
    projector.load_state_dict(proj_sd, strict=True)

    encoder.to(device).eval()
    projector.to(device).eval()
    return encoder, projector


def extract_frames(video_path, target_size=224):
    """Extract all frames from video."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size))
        frames.append(frame)
    cap.release()
    return frames


@torch.no_grad()
def compute_surprise(encoder, projector, frames, device, context_len=3):
    """Compute per-step surprise as cosine distance between consecutive embeddings.
    
    Surprise = how much the embedding changes between consecutive frames.
    A world model that learned physical priors should produce embeddings where
    plausible transitions have low surprise and implausible ones have high surprise.
    
    Returns dict with surprise statistics.
    """
    if len(frames) < context_len + 1:
        return None

    # Encode all frames
    batch = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
    batch = batch.to(device)

    embs = []
    for i in range(0, len(batch), 32):
        out = encoder(batch[i:i+32], interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]
        emb = projector(cls)
        embs.append(emb)
    embs = torch.cat(embs, dim=0)  # (T, D)

    # Compute consecutive cosine distances (surprise per step)
    cos_sims = F.cosine_similarity(embs[:-1], embs[1:], dim=-1)
    cos_dists = 1.0 - cos_sims  # higher = more surprising

    # Also compute MSE between consecutive embeddings
    mse_dists = ((embs[:-1] - embs[1:]) ** 2).mean(dim=-1)

    return {
        'cos_dist_mean': cos_dists.mean().item(),
        'cos_dist_std': cos_dists.std().item(),
        'cos_dist_max': cos_dists.max().item(),
        'mse_mean': mse_dists.mean().item(),
        'mse_std': mse_dists.std().item(),
        'mse_max': mse_dists.max().item(),
        'n_frames': len(frames),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--model-type', required=True, choices=['tiny', 'small', 'dinov2'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-pairs', type=int, default=0, help='Limit pairs (0=all)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    with open(args.manifest) as f:
        manifest = json.load(f)

    pairs = manifest['pairs']
    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]

    print(f"Loading {args.model_type} from {args.ckpt}...")
    encoder, projector = load_encoder(args.ckpt, args.model_type, device)

    # Group by perturbation type
    by_perturbation = defaultdict(list)
    for pair in pairs:
        by_perturbation[pair['perturbation']].append(pair)

    results_by_perturb = {}
    all_plaus_surprise = []
    all_implaus_surprise = []

    for pname, ppairs in sorted(by_perturbation.items()):
        plaus_scores = []
        implaus_scores = []

        for pi, pair in enumerate(ppairs):
            plaus_frames = extract_frames(pair['plausible'])
            implaus_frames = extract_frames(pair['implausible'])

            if not plaus_frames or not implaus_frames:
                continue

            plaus_s = compute_surprise(encoder, projector, plaus_frames, device)
            implaus_s = compute_surprise(encoder, projector, implaus_frames, device)

            if plaus_s and implaus_s:
                plaus_scores.append(plaus_s['cos_dist_mean'])
                implaus_scores.append(implaus_s['cos_dist_mean'])
                all_plaus_surprise.append(plaus_s['cos_dist_mean'])
                all_implaus_surprise.append(implaus_s['cos_dist_mean'])

        if not plaus_scores:
            continue

        plaus_arr = np.array(plaus_scores)
        implaus_arr = np.array(implaus_scores)
        separation = implaus_arr.mean() - plaus_arr.mean()
        # How often implausible > plausible (paired)
        correct = sum(i > p for i, p in zip(implaus_scores, plaus_scores))
        accuracy = correct / len(plaus_scores)

        results_by_perturb[pname] = {
            'n_pairs': len(plaus_scores),
            'plausible_mean': float(plaus_arr.mean()),
            'plausible_std': float(plaus_arr.std()),
            'implausible_mean': float(implaus_arr.mean()),
            'implausible_std': float(implaus_arr.std()),
            'separation': float(separation),
            'accuracy': float(accuracy),
        }

        direction = '✓' if separation > 0 else '✗'
        print(f"  {pname:25s}: plaus={plaus_arr.mean():.4f} implaus={implaus_arr.mean():.4f} "
              f"sep={separation:+.4f} acc={accuracy:.1%} {direction}")

    # Overall
    p_all = np.array(all_plaus_surprise)
    i_all = np.array(all_implaus_surprise)
    overall_sep = i_all.mean() - p_all.mean()
    overall_acc = sum(i > p for i, p in zip(all_implaus_surprise, all_plaus_surprise)) / len(all_plaus_surprise)

    print(f"\n  {'OVERALL':25s}: plaus={p_all.mean():.4f} implaus={i_all.mean():.4f} "
          f"sep={overall_sep:+.4f} acc={overall_acc:.1%}")

    output = {
        'model': {'checkpoint': str(args.ckpt), 'type': args.model_type},
        'per_perturbation': results_by_perturb,
        'overall': {
            'n_pairs': len(all_plaus_surprise),
            'plausible_mean': float(p_all.mean()),
            'implausible_mean': float(i_all.mean()),
            'separation': float(overall_sep),
            'accuracy': float(overall_acc),
        }
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
