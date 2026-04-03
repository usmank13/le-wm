"""
Wind Speed Probe for LeWM

Tests whether LeWM embeddings encode wind speed information.
If the depth-regularized model better encodes physical structure,
it should also better predict wind conditions from video embeddings
(wind causes visible motion in foliage/crops).

Pipeline:
1. Extract sessions from captured tars
2. Get wind speed data per session via Open-Meteo API
3. Extract video frames from nav_front camera
4. Encode frames through LeWM encoder
5. Average embeddings per session → session-level representation
6. Train a linear probe: embedding → wind_speed
7. Report R² and MSE; compare baseline vs depth-reg

Usage:
    python eval_wind_probe.py \
        --ckpt <checkpoint_path> \
        --model-type tiny \
        --sessions-dir /data/wind_captures/extracted/ \
        --wind-data /data/wind_captures/wind_dataset.json \
        [--output wind_probe_results.json]
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

sys.path.insert(0, str(Path(__file__).parent))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model(ckpt_path, model_type, device):
    """Load LeWM encoder + projector."""
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
            'tiny', patch_size=14, image_size=224,
            pretrained=False, use_mask_token=False)
        hidden_dim = 192
    elif model_type == 'small':
        import stable_pretraining as spt
        encoder = spt.backbone.utils.vit_hf(
            'small', patch_size=14, image_size=224,
            pretrained=False, use_mask_token=False)
        hidden_dim = 384
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    enc_sd = {k.replace('model.encoder.', ''): v
              for k, v in sd.items() if k.startswith('model.encoder.')}
    encoder.load_state_dict(enc_sd, strict=True)

    embed_dim = 192
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    proj_sd = {k.replace('model.projector.', ''): v
               for k, v in sd.items() if k.startswith('model.projector.')}
    projector.load_state_dict(proj_sd, strict=True)

    encoder = encoder.to(device).eval()
    projector = projector.to(device).eval()
    return encoder, projector, embed_dim


def extract_nav_front_frames(session_dir, target_size=224, max_frames=100, frameskip=3):
    """Extract frames from nav_front color video in a session.
    
    Args:
        session_dir: Path to extracted session directory
        target_size: Resize frames to this size
        max_frames: Maximum frames to extract (evenly sampled)
        frameskip: Only take every Nth frame
    
    Returns:
        frames: numpy array (T, H, W, 3) uint8 RGB, or None if no video
    """
    video_dir = Path(session_dir) / 'video'
    if not video_dir.exists():
        return None
    
    # Find nav_front color video
    nav_front = None
    for f in sorted(video_dir.iterdir()):
        if 'nav_front' in f.name and 'color' in f.name and f.suffix == '.mp4':
            nav_front = f
            break
    
    if nav_front is None:
        # Try any capture_color_nav_front pattern
        for f in sorted(video_dir.iterdir()):
            if 'color' in f.name and 'nav' in f.name and f.suffix == '.mp4':
                nav_front = f
                break
    
    if nav_front is None:
        return None
    
    cap = cv2.VideoCapture(str(nav_front))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly
    if total <= 0:
        cap.release()
        return None
    
    indices = list(range(0, total, frameskip))
    if len(indices) > max_frames:
        step = len(indices) / max_frames
        indices = [indices[int(i * step)] for i in range(max_frames)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size))
        frames.append(frame)
    
    cap.release()
    
    if len(frames) < 5:
        return None
    
    return np.stack(frames, axis=0)


@torch.no_grad()
def encode_session(encoder, projector, frames_np, device, batch_size=32):
    """Encode frames and return mean + std pooled embedding.
    
    Returns both mean and std to capture temporal dynamics
    (windy sessions should have higher embedding variance).
    """
    embs = []
    for i in range(0, len(frames_np), batch_size):
        batch = torch.from_numpy(frames_np[i:i + batch_size])
        batch = batch.permute(0, 3, 1, 2).float() / 255.0
        batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
        batch = batch.to(device)
        
        out = encoder(batch, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]
        emb = projector(cls)
        embs.append(emb.cpu())
    
    all_embs = torch.cat(embs, dim=0)  # (T, D)
    
    mean_emb = all_embs.mean(dim=0)
    std_emb = all_embs.std(dim=0)
    
    # Also compute temporal difference features (captures dynamics)
    if len(all_embs) > 1:
        diffs = all_embs[1:] - all_embs[:-1]
        diff_mean = diffs.mean(dim=0)
        diff_std = diffs.std(dim=0)
    else:
        diff_mean = torch.zeros_like(mean_emb)
        diff_std = torch.zeros_like(std_emb)
    
    # Concatenate: [mean, std, diff_mean, diff_std] → 4*D features
    feature = torch.cat([mean_emb, std_emb, diff_mean, diff_std])
    return feature.numpy()


def run_wind_probe(features, wind_speeds, feature_dim_label=""):
    """Train a ridge regression probe and report results.
    
    Uses leave-one-out or 5-fold CV depending on sample size.
    """
    X = np.stack(features)
    y = np.array(wind_speeds)
    
    n = len(y)
    if n < 10:
        print(f"  Warning: only {n} samples, results may be unreliable")
    
    # Standardize features
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X - X_mean) / X_std
    
    # Standardize target
    y_mean = y.mean()
    y_std = y.std() + 1e-8
    y_norm = (y - y_mean) / y_std
    
    # Try multiple alphas
    best_r2 = -float('inf')
    best_alpha = 1.0
    
    n_folds = min(5, n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X_norm, y_norm, cv=kf, scoring='r2')
        mean_r2 = scores.mean()
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_alpha = alpha
    
    # Final model with best alpha
    model = Ridge(alpha=best_alpha)
    cv_r2 = cross_val_score(model, X_norm, y_norm, cv=kf, scoring='r2')
    cv_mse = -cross_val_score(model, X_norm, y, cv=kf, scoring='neg_mean_squared_error')
    
    # Fit on all data for inspection
    model.fit(X_norm, y)
    y_pred = model.predict(X_norm)
    train_r2 = r2_score(y, y_pred)
    
    return {
        'cv_r2_mean': float(cv_r2.mean()),
        'cv_r2_std': float(cv_r2.std()),
        'cv_mse_mean': float(cv_mse.mean()),
        'cv_mse_std': float(cv_mse.std()),
        'train_r2': float(train_r2),
        'best_alpha': float(best_alpha),
        'n_samples': n,
        'n_folds': n_folds,
        'wind_speed_range': [float(y.min()), float(y.max())],
        'wind_speed_mean': float(y.mean()),
        'wind_speed_std': float(y.std()),
    }


def main():
    parser = argparse.ArgumentParser(description='Wind Speed Probe for LeWM')
    parser.add_argument('--ckpt', required=True, help='Model checkpoint')
    parser.add_argument('--model-type', required=True, choices=['tiny', 'small', 'dinov2'])
    parser.add_argument('--sessions-dir', required=True,
                        help='Directory with extracted sessions')
    parser.add_argument('--wind-data', required=True,
                        help='Wind dataset JSON from build_wind_dataset.py')
    parser.add_argument('--max-frames', type=int, default=100,
                        help='Max frames per session')
    parser.add_argument('--output', default='wind_probe_results.json')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load wind data
    with open(args.wind_data) as f:
        wind_dataset = json.load(f)
    
    wind_by_session = {}
    for entry in wind_dataset['sessions']:
        # Key by session directory name
        session_name = Path(entry['session_path']).name
        wind_by_session[session_name] = entry
    
    print(f"Wind data for {len(wind_by_session)} sessions")
    
    # Load model
    print(f"Loading model ({args.model_type})...")
    encoder, projector, embed_dim = load_model(args.ckpt, args.model_type, device)
    
    # Process sessions
    sessions_dir = Path(args.sessions_dir)
    session_dirs = sorted([d for d in sessions_dir.iterdir() if d.is_dir()])
    
    features = []
    wind_speeds = []
    wind_gusts = []
    session_names = []
    
    for session_dir in session_dirs:
        session_name = session_dir.name
        
        if session_name not in wind_by_session:
            print(f"  {session_name}: no wind data, skipping")
            continue
        
        wind_info = wind_by_session[session_name]
        wind_speed = wind_info['wind_speed_10m_ms']
        
        if wind_speed is None:
            print(f"  {session_name}: null wind speed, skipping")
            continue
        
        print(f"  {session_name}: wind={wind_speed:.1f} m/s ...", end=" ", flush=True)
        
        frames = extract_nav_front_frames(
            session_dir, max_frames=args.max_frames)
        
        if frames is None:
            print("no video found")
            continue
        
        feature = encode_session(encoder, projector, frames, device)
        features.append(feature)
        wind_speeds.append(wind_speed)
        wind_gusts.append(wind_info.get('wind_gusts_10m_ms', wind_speed))
        session_names.append(session_name)
        print(f"{len(frames)} frames → {feature.shape[0]}D feature")
    
    if len(features) < 5:
        print(f"\nOnly {len(features)} valid sessions — need at least 5 for probe")
        return
    
    print(f"\n{'='*60}")
    print(f"WIND SPEED PROBE")
    print(f"{'='*60}")
    print(f"Sessions: {len(features)}")
    print(f"Feature dim: {features[0].shape[0]} (4 × {embed_dim})")
    print(f"Wind speed range: {min(wind_speeds):.1f} - {max(wind_speeds):.1f} m/s")
    print(f"Wind speed mean: {np.mean(wind_speeds):.1f} ± {np.std(wind_speeds):.1f} m/s")
    
    # Run probe on wind speed
    print(f"\n--- Wind Speed Probe ---")
    speed_results = run_wind_probe(features, wind_speeds)
    print(f"  CV R²: {speed_results['cv_r2_mean']:.4f} ± {speed_results['cv_r2_std']:.4f}")
    print(f"  CV MSE: {speed_results['cv_mse_mean']:.4f} ± {speed_results['cv_mse_std']:.4f}")
    print(f"  Train R²: {speed_results['train_r2']:.4f}")
    print(f"  Best alpha: {speed_results['best_alpha']}")
    
    # Run probe on wind gusts
    print(f"\n--- Wind Gust Probe ---")
    gust_results = run_wind_probe(features, wind_gusts)
    print(f"  CV R²: {gust_results['cv_r2_mean']:.4f} ± {gust_results['cv_r2_std']:.4f}")
    print(f"  CV MSE: {gust_results['cv_mse_mean']:.4f} ± {gust_results['cv_mse_std']:.4f}")
    
    # Random baseline
    print(f"\n--- Random Baseline ---")
    np.random.seed(42)
    random_features = [np.random.randn(features[0].shape[0]) for _ in features]
    random_results = run_wind_probe(random_features, wind_speeds)
    print(f"  CV R²: {random_results['cv_r2_mean']:.4f} ± {random_results['cv_r2_std']:.4f}")
    
    # Save results
    results = {
        'model': {
            'checkpoint': str(args.ckpt),
            'type': args.model_type,
            'embed_dim': embed_dim,
            'feature_dim': features[0].shape[0],
        },
        'data': {
            'n_sessions': len(features),
            'sessions': session_names,
            'wind_speeds': wind_speeds,
            'wind_gusts': wind_gusts,
        },
        'wind_speed_probe': speed_results,
        'wind_gust_probe': gust_results,
        'random_baseline': random_results,
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
