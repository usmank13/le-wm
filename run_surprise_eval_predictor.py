"""
Predictor-based Surprise Score Evaluation using HDF5 validation data.

Matches the paper's VoE methodology:
1. Encode frames with encoder → z_t
2. Predict next embedding with predictor: ẑ_{t+1} = pred(z_t, a_t)  
3. Surprise = ||ẑ_{t+1} - z_{t+1}||² (or cosine distance)
4. Compare surprise on plausible vs perturbed trajectories

Perturbations applied to pixel frames while keeping actions unchanged:
- brightness_jump: abrupt 2x brightness at midpoint
- color_swap: swap R/G channels at midpoint  
- teleportation: replace frames with frames from a different episode (physical violation)
- temporal_reversal: reverse frame order in a region from midpoint
- freeze: freeze frames from midpoint onward
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

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_full_model(ckpt_path, model_type, device):
    """Load full JEPA model from checkpoint, inferring architecture from weights."""
    from eval_wind_probe_predictor import infer_predictor_config
    from jepa import JEPA
    from module import MLP, ARPredictor, Embedder

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
    pcfg = infer_predictor_config(sd)

    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    predictor = ARPredictor(
        num_frames=pcfg['num_frames'], depth=pcfg['depth'],
        heads=pcfg['heads'], mlp_dim=pcfg['mlp_dim'],
        input_dim=pcfg['input_dim'], hidden_dim=pcfg['hidden_dim'],
        dim_head=pcfg['dim_head'])
    pred_proj_input = pcfg.get('output_dim', pcfg['hidden_dim'])
    pred_proj = MLP(input_dim=pred_proj_input, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)

    act_w = sd.get('model.action_encoder.patch_embed.weight')
    act_dim = act_w.shape[1] if act_w is not None else 4
    action_encoder = Embedder(input_dim=act_dim, emb_dim=embed_dim)

    model = JEPA(encoder=encoder, predictor=predictor,
                 action_encoder=action_encoder, projector=projector,
                 pred_proj=pred_proj)

    model_sd = {k.replace('model.', '', 1): v for k, v in sd.items() if k.startswith('model.')}
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing:
        print(f"  Warning: missing keys: {missing[:3]}")

    model.to(device).eval()
    return model, embed_dim, pcfg['num_frames'], act_dim


# === Perturbation functions (operate on numpy pixel arrays) ===

def perturb_brightness_jump(pixels, midpoint):
    """Abrupt 2x brightness at midpoint."""
    result = pixels.copy()
    result[midpoint:] = np.clip(result[midpoint:].astype(np.float32) * 2.0, 0, 255).astype(np.uint8)
    return result


def perturb_color_swap(pixels, midpoint):
    """Swap R and G channels at midpoint."""
    result = pixels.copy()
    result[midpoint:, :, :, 0], result[midpoint:, :, :, 1] = \
        pixels[midpoint:, :, :, 1].copy(), pixels[midpoint:, :, :, 0].copy()
    return result


def perturb_teleportation(pixels, midpoint, other_pixels):
    """Replace frames from midpoint with frames from a different episode."""
    result = pixels.copy()
    n_replace = len(pixels) - midpoint
    # Use frames from other episode starting at a random offset
    start = np.random.randint(0, max(1, len(other_pixels) - n_replace))
    end = min(start + n_replace, len(other_pixels))
    actual = end - start
    result[midpoint:midpoint + actual] = other_pixels[start:end]
    return result


def perturb_temporal_reversal(pixels, midpoint):
    """Reverse frame order from midpoint onward."""
    result = pixels.copy()
    result[midpoint:] = result[midpoint:][::-1]
    return result


def perturb_freeze(pixels, midpoint):
    """Freeze at midpoint frame."""
    result = pixels.copy()
    result[midpoint:] = result[midpoint:midpoint + 1]
    return result


@torch.no_grad()
def compute_predictor_surprise(model, pixels, actions, device, history_size=3,
                                frameskip=2, batch_encode=64):
    """Compute per-step predictor surprise for a trajectory.
    
    Surprise_t = ||pred(z_{t-1}, a_{t-1}) - z_t||² 
    
    Returns array of surprise values, one per timestep (starting from history_size).
    """
    T = len(pixels)

    # Encode all frames
    embs = []
    for i in range(0, T, batch_encode):
        batch = torch.from_numpy(pixels[i:i + batch_encode]).permute(0, 3, 1, 2).float() / 255.0
        batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
        batch = batch.to(device)
        out = model.encoder(batch, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]
        emb = model.projector(cls)
        embs.append(emb)
    embs = torch.cat(embs, dim=0)  # (T, D)

    # Prepare actions with frameskip concatenation
    # Training uses frameskip=2: action_t = concat(raw_action_t, raw_action_{t+1})
    act_tensor = torch.from_numpy(actions).float().to(device)  # (T, 2)
    if frameskip > 1:
        # Concatenate frameskip consecutive actions
        padded = F.pad(act_tensor, (0, 0, 0, frameskip - 1))  # pad end
        act_cat = torch.cat([padded[i:i + T] for i in range(frameskip)], dim=-1)  # (T, 2*frameskip)
    else:
        act_cat = act_tensor

    # Compute surprise at each step using predictor
    surprises_mse = []
    surprises_cos = []

    for t in range(history_size, T):
        # Context: last history_size embeddings up to t-1
        ctx_start = max(0, t - history_size)
        ctx_emb = embs[ctx_start:t].unsqueeze(0)  # (1, H, D)
        ctx_act = act_cat[ctx_start:t].unsqueeze(0)  # (1, H, act_dim)

        # Encode actions
        act_emb = model.action_encoder(ctx_act)  # (1, H, D)

        # Predict next embedding
        pred = model.predict(ctx_emb, act_emb)  # (1, H, D)
        pred_next = pred[0, -1]  # predicted z_t

        # Actual embedding at t
        actual = embs[t]

        # Surprise metrics
        mse = F.mse_loss(pred_next, actual).item()
        cos_dist = (1.0 - F.cosine_similarity(pred_next.unsqueeze(0),
                                               actual.unsqueeze(0))).item()
        surprises_mse.append(mse)
        surprises_cos.append(cos_dist)

    return np.array(surprises_mse), np.array(surprises_cos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--model-type', required=True, choices=['tiny', 'small', 'dinov2'])
    parser.add_argument('--val-data', default='/data/lewm_data/aigen_val.h5')
    parser.add_argument('--clip-len', type=int, default=30, help='Frames per clip')
    parser.add_argument('--clips-per-ep', type=int, default=5)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Loading model ({args.model_type})...")
    model, embed_dim, hist_size, act_dim = load_full_model(args.ckpt, args.model_type, device)
    print(f"  embed_dim={embed_dim}, history={hist_size}, act_dim={act_dim}")

    # Load validation data
    f = h5py.File(args.val_data, 'r')
    all_pixels = f['pixels'][:]
    all_actions = f['action'][:]
    ep_lens = f['ep_len'][:]
    ep_offsets = f['ep_offset'][:]
    f.close()
    print(f"Loaded {len(ep_lens)} episodes, {len(all_pixels)} total frames")

    perturbations = {
        'brightness_jump': perturb_brightness_jump,
        'color_swap': perturb_color_swap,
        'temporal_reversal': perturb_temporal_reversal,
        'teleportation': None,  # needs other_pixels, handled separately
    }

    results_by_perturb = defaultdict(lambda: {'plausible': [], 'implausible': []})
    np.random.seed(42)

    for ei in range(len(ep_lens)):
        offset = ep_offsets[ei]
        length = ep_lens[ei]
        ep_pixels = all_pixels[offset:offset + length]
        ep_actions = all_actions[offset:offset + length]

        # Pick clip start points
        max_start = length - args.clip_len - 1
        if max_start < 1:
            continue
        starts = np.random.choice(range(0, max_start, 5),
                                   size=min(args.clips_per_ep, max_start // 5),
                                   replace=False)

        for ci, start in enumerate(starts):
            clip_pixels = ep_pixels[start:start + args.clip_len]
            clip_actions = ep_actions[start:start + args.clip_len]
            midpoint = args.clip_len // 2

            # Plausible surprise
            plaus_mse, plaus_cos = compute_predictor_surprise(
                model, clip_pixels, clip_actions, device, hist_size)

            for pname, pfunc in perturbations.items():
                if pname == 'teleportation':
                    # Use frames from a different episode
                    other_ei = (ei + 1) % len(ep_lens)
                    other_offset = ep_offsets[other_ei]
                    other_len = ep_lens[other_ei]
                    other_pixels = all_pixels[other_offset:other_offset + other_len]
                    perturbed = perturb_teleportation(clip_pixels, midpoint, other_pixels)
                else:
                    perturbed = pfunc(clip_pixels, midpoint)

                # Implausible surprise (same actions, perturbed pixels)
                imp_mse, imp_cos = compute_predictor_surprise(
                    model, perturbed, clip_actions, device, hist_size)

                # Compare post-perturbation surprise (after midpoint)
                post_start = midpoint - hist_size  # account for warmup
                post_start = max(0, post_start)

                plaus_post = plaus_mse[post_start:].mean()
                imp_post = imp_mse[post_start:].mean()

                results_by_perturb[pname]['plausible'].append(float(plaus_post))
                results_by_perturb[pname]['implausible'].append(float(imp_post))

        print(f"  Episode {ei}: {len(starts)} clips processed")

    # Summarize
    print(f"\n{'='*60}")
    print(f"PREDICTOR-BASED SURPRISE EVAL (MSE)")
    print(f"{'='*60}")

    all_results = {}
    all_plaus = []
    all_implaus = []

    for pname in sorted(results_by_perturb.keys()):
        data = results_by_perturb[pname]
        p = np.array(data['plausible'])
        im = np.array(data['implausible'])
        sep = im.mean() - p.mean()
        correct = sum(i > pl for i, pl in zip(data['implausible'], data['plausible']))
        acc = correct / len(p) if len(p) > 0 else 0

        direction = '✓' if sep > 0 else '✗'
        print(f"  {pname:25s}: plaus={p.mean():.4f} implaus={im.mean():.4f} "
              f"sep={sep:+.4f} acc={acc:.1%} {direction}")

        all_results[pname] = {
            'n_pairs': len(p),
            'plausible_mean': float(p.mean()),
            'implausible_mean': float(im.mean()),
            'separation': float(sep),
            'accuracy': float(acc),
        }
        all_plaus.extend(data['plausible'])
        all_implaus.extend(data['implausible'])

    p_all = np.array(all_plaus)
    i_all = np.array(all_implaus)
    overall_acc = sum(i > p for i, p in zip(all_implaus, all_plaus)) / len(all_plaus)
    print(f"\n  {'OVERALL':25s}: plaus={p_all.mean():.4f} implaus={i_all.mean():.4f} "
          f"sep={i_all.mean() - p_all.mean():+.4f} acc={overall_acc:.1%}")

    output = {
        'model': {'checkpoint': str(args.ckpt), 'type': args.model_type},
        'config': {'clip_len': args.clip_len, 'history_size': hist_size,
                   'act_dim': act_dim, 'val_data': args.val_data},
        'per_perturbation': all_results,
        'overall': {
            'n_pairs': len(all_plaus),
            'plausible_mean': float(p_all.mean()),
            'implausible_mean': float(i_all.mean()),
            'separation': float(i_all.mean() - p_all.mean()),
            'accuracy': float(overall_acc),
        }
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
