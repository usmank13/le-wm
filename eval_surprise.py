"""
Surprise Score Evaluation for LeWM

Methodology (following LeWM paper):
1. Load a video (sequence of frames) — either plausible or implausible
2. Encode each frame through the LeWM encoder → embeddings
3. For each timestep t, use the predictor to predict emb_{t+1} from context (emb_{t-2:t} + actions)
4. Surprise score = prediction error between predicted and actual embedding at t+1
5. Higher surprise = the model finds the video more unexpected

For our evaluation:
- Plausible videos: physically consistent dynamics (wind, lighting)
- Implausible videos: same scene with physics violations (reversed wind region, contradictory lighting)
- Both share same generative artifacts, so any surprise difference reflects physics understanding

Usage:
    python eval_surprise.py \
        --ckpt <checkpoint_path> \
        --video-dir <dir_with_plausible_and_implausible_videos> \
        --model-type tiny|small|dinov2 \
        [--frameskip 2] \
        [--context-len 3] \
        [--output results.json]

Video directory structure:
    video_dir/
        pair_001/
            plausible.mp4
            implausible.mp4
        pair_002/
            plausible.mp4
            implausible.mp4
        ...

Each pair should be the same scene with plausible vs implausible dynamics.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model(ckpt_path, model_type, device):
    """Load a LeWM model from checkpoint.
    
    Args:
        ckpt_path: Path to Lightning checkpoint
        model_type: 'tiny', 'small', or 'dinov2'
        device: torch device
    
    Returns:
        model: JEPA model in eval mode
    """
    from module import ARPredictor, Embedder, MLP
    from jepa import JEPA

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    if model_type == 'dinov2':
        from train_dinov2 import DINOv2Encoder
        encoder = DINOv2Encoder(freeze=True)
        hidden_dim = 384
        embed_dim = 192
    elif model_type == 'tiny':
        import stable_pretraining as spt
        encoder = spt.backbone.utils.vit_hf(
            'tiny', patch_size=14, image_size=224,
            pretrained=False, use_mask_token=False
        )
        hidden_dim = 192
        embed_dim = 192
    elif model_type == 'small':
        import stable_pretraining as spt
        encoder = spt.backbone.utils.vit_hf(
            'small', patch_size=14, image_size=224,
            pretrained=False, use_mask_token=False
        )
        hidden_dim = 384
        embed_dim = 192
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load encoder weights
    enc_sd = {k.replace('model.encoder.', ''): v
              for k, v in sd.items() if k.startswith('model.encoder.')}
    encoder.load_state_dict(enc_sd, strict=True)

    # Build and load remaining components
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    proj_sd = {k.replace('model.projector.', ''): v
               for k, v in sd.items() if k.startswith('model.projector.')}
    projector.load_state_dict(proj_sd, strict=True)

    predictor = ARPredictor(
        num_frames=3, input_dim=embed_dim, hidden_dim=hidden_dim,
        output_dim=hidden_dim, depth=6, heads=16, mlp_dim=2048,
        dim_head=64, dropout=0.1, emb_dropout=0.0,
    )
    pred_sd = {k.replace('model.predictor.', ''): v
               for k, v in sd.items() if k.startswith('model.predictor.')}
    predictor.load_state_dict(pred_sd, strict=True)

    pred_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pp_sd = {k.replace('model.pred_proj.', ''): v
             for k, v in sd.items() if k.startswith('model.pred_proj.')}
    pred_proj.load_state_dict(pp_sd, strict=True)

    # For surprise eval we don't use actions — use zero actions
    # Action dim = 2 * frameskip = 4
    action_encoder = Embedder(input_dim=4, emb_dim=embed_dim)
    ae_sd = {k.replace('model.action_encoder.', ''): v
             for k, v in sd.items() if k.startswith('model.action_encoder.')}
    action_encoder.load_state_dict(ae_sd, strict=True)

    model = JEPA(encoder, predictor, action_encoder, projector, pred_proj)
    return model.to(device).eval()


def extract_frames(video_path, target_size=224, max_frames=None):
    """Extract frames from a video file.
    
    Returns:
        frames: numpy array of shape (T, H, W, 3) uint8 RGB
        fps: video frame rate
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size))
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return np.stack(frames, axis=0), fps


def preprocess_frames(frames_np, device):
    """Convert uint8 numpy frames to normalized tensor.
    
    Args:
        frames_np: (T, H, W, 3) uint8
        device: torch device
    
    Returns:
        tensor: (T, 3, H, W) float normalized with ImageNet stats
    """
    x = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device)


@torch.no_grad()
def encode_frames(model, frames_np, device, batch_size=32):
    """Encode all frames through encoder + projector.
    
    Returns:
        embeddings: (T, D) tensor
    """
    embs = []
    for i in range(0, len(frames_np), batch_size):
        batch = preprocess_frames(frames_np[i:i + batch_size], device)
        out = model.encoder(batch, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]
        emb = model.projector(cls)
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def compute_surprise_scores(model, embeddings, device, context_len=3, frameskip=2):
    """Compute per-timestep surprise scores.
    
    For each timestep t >= context_len, predict emb_t from context
    emb_{t-context_len:t} and measure prediction error.
    
    Since these are videos without robot actions, we use zero actions.
    The surprise comes from the visual dynamics being unexpected.
    
    Args:
        model: JEPA model
        embeddings: (T, D) tensor of frame embeddings
        device: torch device
        context_len: number of context frames
        frameskip: action frameskip (for zero-action dimension)
    
    Returns:
        surprise_scores: dict with per-step cosine distance and MSE
    """
    T, D = embeddings.shape
    
    # Zero actions (no robot actions in these videos)
    zero_action = torch.zeros(1, context_len, 2 * frameskip, device=device)
    action_emb = model.action_encoder(zero_action)  # (1, ctx, D)
    
    cosine_distances = []
    mse_scores = []
    
    for t in range(context_len, T):
        # Context: embeddings from t-context_len to t-1
        ctx = embeddings[t - context_len:t].unsqueeze(0).to(device)  # (1, ctx, D)
        
        # Predict next embedding
        pred = model.predict(ctx, action_emb)  # (1, 1, D) or (1, ctx, D)
        pred_emb = pred[:, -1, :]  # Take last prediction
        
        # Ground truth
        gt_emb = embeddings[t:t + 1].to(device)
        
        # Surprise = prediction error
        cos_dist = 1.0 - F.cosine_similarity(pred_emb, gt_emb, dim=-1).item()
        mse = F.mse_loss(pred_emb, gt_emb).item()
        
        cosine_distances.append(cos_dist)
        mse_scores.append(mse)
    
    return {
        'cosine_distances': cosine_distances,
        'mse_scores': mse_scores,
        'mean_cosine_distance': float(np.mean(cosine_distances)),
        'mean_mse': float(np.mean(mse_scores)),
        'max_cosine_distance': float(np.max(cosine_distances)),
        'max_mse': float(np.max(mse_scores)),
        'std_cosine_distance': float(np.std(cosine_distances)),
        'std_mse': float(np.std(mse_scores)),
    }


def evaluate_video_pair(model, plausible_path, implausible_path, device,
                        context_len=3, frameskip=2, target_size=224):
    """Evaluate a plausible/implausible video pair.
    
    Returns:
        result dict with surprise scores for both videos and separation metrics.
    """
    # Extract and encode frames
    plaus_frames, plaus_fps = extract_frames(plausible_path, target_size)
    implaus_frames, implaus_fps = extract_frames(implausible_path, target_size)
    
    print(f"  Plausible: {len(plaus_frames)} frames @ {plaus_fps:.1f}fps")
    print(f"  Implausible: {len(implaus_frames)} frames @ {implaus_fps:.1f}fps")
    
    plaus_embs = encode_frames(model, plaus_frames, device)
    implaus_embs = encode_frames(model, implaus_frames, device)
    
    # Compute surprise scores
    plaus_surprise = compute_surprise_scores(
        model, plaus_embs, device, context_len, frameskip)
    implaus_surprise = compute_surprise_scores(
        model, implaus_embs, device, context_len, frameskip)
    
    # Separation metrics
    # A good world model should show higher surprise for implausible videos
    cos_separation = implaus_surprise['mean_cosine_distance'] - plaus_surprise['mean_cosine_distance']
    mse_separation = implaus_surprise['mean_mse'] - plaus_surprise['mean_mse']
    
    # Relative separation (normalized by plausible surprise)
    cos_relative = cos_separation / (plaus_surprise['mean_cosine_distance'] + 1e-8)
    mse_relative = mse_separation / (plaus_surprise['mean_mse'] + 1e-8)
    
    return {
        'plausible': plaus_surprise,
        'implausible': implaus_surprise,
        'separation': {
            'cosine_distance_diff': float(cos_separation),
            'mse_diff': float(mse_separation),
            'cosine_relative_separation': float(cos_relative),
            'mse_relative_separation': float(mse_relative),
        },
        'num_frames': {
            'plausible': len(plaus_frames),
            'implausible': len(implaus_frames),
        }
    }


def main():
    parser = argparse.ArgumentParser(description='LeWM Surprise Score Evaluation')
    parser.add_argument('--ckpt', required=True, help='Path to model checkpoint')
    parser.add_argument('--model-type', required=True, choices=['tiny', 'small', 'dinov2'],
                        help='Model architecture type')
    parser.add_argument('--video-dir', required=True,
                        help='Directory containing pair_*/plausible.mp4 and pair_*/implausible.mp4')
    parser.add_argument('--context-len', type=int, default=3,
                        help='Number of context frames for prediction')
    parser.add_argument('--frameskip', type=int, default=2,
                        help='Frameskip (for action dimension)')
    parser.add_argument('--target-size', type=int, default=224,
                        help='Frame resize dimension')
    parser.add_argument('--output', default='surprise_results.json',
                        help='Output JSON file')
    parser.add_argument('--device', default='cuda',
                        help='Device (cuda or cpu)')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model ({args.model_type}) from {args.ckpt}...")
    model = load_model(args.ckpt, args.model_type, device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Find video pairs
    video_dir = Path(args.video_dir)
    pairs = sorted([d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith('pair_')])
    
    if not pairs:
        # Also support flat directory with plausible_*.mp4 / implausible_*.mp4
        plausible_vids = sorted(video_dir.glob('plausible*.mp4'))
        implausible_vids = sorted(video_dir.glob('implausible*.mp4'))
        if plausible_vids and implausible_vids:
            pairs = list(zip(plausible_vids, implausible_vids))
        else:
            print(f"No video pairs found in {video_dir}")
            print("Expected: pair_*/plausible.mp4 + pair_*/implausible.mp4")
            print("  or: plausible_001.mp4 + implausible_001.mp4")
            return
    
    results = {
        'model': {
            'checkpoint': str(args.ckpt),
            'type': args.model_type,
            'context_len': args.context_len,
            'frameskip': args.frameskip,
        },
        'pairs': [],
        'summary': {},
    }
    
    all_cos_sep = []
    all_mse_sep = []
    all_cos_rel = []
    all_mse_rel = []
    
    for i, pair in enumerate(pairs):
        if isinstance(pair, tuple):
            plaus_path, implaus_path = pair
            pair_name = plaus_path.stem
        else:
            pair_name = pair.name
            plaus_path = pair / 'plausible.mp4'
            implaus_path = pair / 'implausible.mp4'
            
            # Also check for .avi, .mov
            if not plaus_path.exists():
                for ext in ['.avi', '.mov', '.webm', '.mkv']:
                    alt = pair / f'plausible{ext}'
                    if alt.exists():
                        plaus_path = alt
                        break
            if not implaus_path.exists():
                for ext in ['.avi', '.mov', '.webm', '.mkv']:
                    alt = pair / f'implausible{ext}'
                    if alt.exists():
                        implaus_path = alt
                        break
        
        if not plaus_path.exists() or not implaus_path.exists():
            print(f"Skipping {pair_name}: missing video files")
            continue
        
        print(f"\n[{i + 1}/{len(pairs)}] {pair_name}")
        print(f"  Plausible: {plaus_path}")
        print(f"  Implausible: {implaus_path}")
        
        pair_result = evaluate_video_pair(
            model, plaus_path, implaus_path, device,
            args.context_len, args.frameskip, args.target_size
        )
        pair_result['name'] = pair_name
        results['pairs'].append(pair_result)
        
        sep = pair_result['separation']
        all_cos_sep.append(sep['cosine_distance_diff'])
        all_mse_sep.append(sep['mse_diff'])
        all_cos_rel.append(sep['cosine_relative_separation'])
        all_mse_rel.append(sep['mse_relative_separation'])
        
        print(f"  Plausible mean surprise (cos): {pair_result['plausible']['mean_cosine_distance']:.6f}")
        print(f"  Implausible mean surprise (cos): {pair_result['implausible']['mean_cosine_distance']:.6f}")
        print(f"  Separation (cos): {sep['cosine_distance_diff']:.6f} "
              f"(relative: {sep['cosine_relative_separation']:.1%})")
    
    if all_cos_sep:
        results['summary'] = {
            'num_pairs': len(all_cos_sep),
            'mean_cosine_separation': float(np.mean(all_cos_sep)),
            'mean_mse_separation': float(np.mean(all_mse_sep)),
            'mean_relative_cosine_separation': float(np.mean(all_cos_rel)),
            'mean_relative_mse_separation': float(np.mean(all_mse_rel)),
            'positive_separation_rate': float(np.mean([s > 0 for s in all_cos_sep])),
        }
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Pairs evaluated: {results['summary']['num_pairs']}")
        print(f"Mean cosine separation: {results['summary']['mean_cosine_separation']:.6f}")
        print(f"Mean relative separation: {results['summary']['mean_relative_cosine_separation']:.1%}")
        print(f"Positive separation rate: {results['summary']['positive_separation_rate']:.0%}")
        print(f"  (% of pairs where implausible > plausible surprise)")
    
    # Save results
    output_path = Path(args.output)
    # Strip per-frame scores for compact output (keep summary stats per pair)
    for pair in results['pairs']:
        for key in ['plausible', 'implausible']:
            pair[key].pop('cosine_distances', None)
            pair[key].pop('mse_scores', None)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
