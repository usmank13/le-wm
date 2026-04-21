"""
Wind Speed Probe using Predictor Rollout Features.

Instead of probing static encoder embeddings, this uses the world model's
predictor to rollout future states and measures prediction error patterns.
A model that understands wind dynamics should have different prediction
error signatures for windy vs calm sessions.

Features per session:
- Mean/std of prediction error (cosine distance) across timesteps
- Mean/std of prediction error growth rate (how fast errors accumulate)
- Mean/std of embedding temporal difference magnitude

Usage:
    python eval_wind_probe_predictor.py \
        --ckpt <checkpoint> --model-type tiny \
        --wind-data /data/wind_captures/wind_all.json \
        --output wind_probe_predictor_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def infer_predictor_config(sd):
    """Infer ARPredictor config from checkpoint weight shapes."""
    # pos_embedding: (1, num_frames, input_dim)
    pos = sd.get('model.predictor.pos_embedding')
    num_frames = pos.shape[1] if pos is not None else 3
    input_dim = pos.shape[2] if pos is not None else 192

    # Count layers
    depth = 0
    while f'model.predictor.transformer.layers.{depth}.attn.to_qkv.weight' in sd:
        depth += 1

    # qkv weight: (3 * heads * dim_head, input_dim) → infer heads
    qkv = sd.get('model.predictor.transformer.layers.0.attn.to_qkv.weight')
    # mlp weight: (mlp_dim, input_dim)
    mlp_w = sd.get('model.predictor.transformer.layers.0.mlp.net.1.weight')
    mlp_dim = mlp_w.shape[0] if mlp_w is not None else 2048

    # Transformer output_dim from norm weight
    norm_w = sd.get('model.predictor.transformer.norm.weight')
    hidden_dim = norm_w.shape[0] if norm_w is not None else input_dim

    # heads: adaLN outputs 6*input_dim for ConditionalBlock
    # to_qkv = 3 * heads * dim_head
    # For this model: qkv=3072, input=192 → 3072/3=1024 total head dim
    # Default dim_head=64 → heads=1024/64=16
    total_qkv = qkv.shape[0] if qkv is not None else 3072
    dim_head = 64
    heads = total_qkv // (3 * dim_head)

    # Determine output_dim: if output_proj exists, its output shape; else hidden_dim
    out_proj_w = sd.get('model.predictor.transformer.output_proj.weight')
    if out_proj_w is not None and out_proj_w.dim() == 2:
        output_dim = out_proj_w.shape[0]
    else:
        output_dim = hidden_dim  # no output_proj or Identity

    return dict(num_frames=num_frames, input_dim=input_dim, hidden_dim=hidden_dim,
                output_dim=output_dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim)


def load_full_model(ckpt_path, model_type, device):
    """Load entire JEPA model (encoder + predictor + action_encoder)."""
    from model_loading import load_full_jepa

    model, embed_dim, hist_size, act_dim, meta = load_full_jepa(ckpt_path, device, freeze_encoder=True)
    pcfg = meta['predictor_cfg']
    print(f"  Predictor config: frames={pcfg['num_frames']}, depth={pcfg['depth']}, "
          f"heads={pcfg['heads']}, input={pcfg['input_dim']}, hidden={pcfg['hidden_dim']}")
    missing = meta.get('missing') or []
    unexpected = meta.get('unexpected') or []
    if missing:
        print(f"  Warning: missing keys: {missing}")
    if unexpected:
        print(f"  Warning: unexpected keys: {unexpected[:5]}")

    return model, embed_dim, hist_size


def extract_video_frames(session_dir, target_size=224, max_frames=200, frameskip=2):
    """Extract sequential frames from nav_front video."""
    video_dir = Path(session_dir) / 'video'
    if not video_dir.exists():
        return None

    nav_front = None
    for f in sorted(video_dir.iterdir()):
        if 'nav_front' in f.name and 'color' in f.name and f.suffix == '.mp4':
            nav_front = f
            break

    if nav_front is None:
        return None

    cap = cv2.VideoCapture(str(nav_front))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    frames = []
    for idx in range(0, min(total, max_frames * frameskip), frameskip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size))
        frames.append(frame)

    cap.release()
    if len(frames) < 10:
        return None
    return np.stack(frames, axis=0)


@torch.no_grad()
def compute_predictor_features(model, frames_np, device, window=5, stride=3, batch_size=16, act_dim=4):
    """Compute predictor-based features for a session.

    For each window of `window` frames:
    1. Encode all frames
    2. Use first frame embedding + zero actions to predict next frames
    3. Compute prediction errors (cosine distance) at each step

    Returns feature vector summarizing prediction error patterns.
    """
    # Encode all frames first
    all_embs = []
    for i in range(0, len(frames_np), batch_size):
        batch = torch.from_numpy(frames_np[i:i + batch_size])
        batch = batch.permute(0, 3, 1, 2).float() / 255.0
        batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
        batch = batch.to(device)

        out = model.encoder(batch, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]
        emb = model.projector(cls)
        all_embs.append(emb.cpu())

    all_embs = torch.cat(all_embs, dim=0)  # (T, D)
    T, D = all_embs.shape

    # Collect prediction errors across sliding windows
    step_errors = {i: [] for i in range(1, window)}  # step -> list of cosine distances
    prediction_residuals = []

    for start in range(0, T - window, stride):
        ctx = all_embs[start:start + window].unsqueeze(0).to(device)  # (1, W, D)
        targets = all_embs[start + 1:start + window].unsqueeze(0).to(device)  # (1, W-1, D)

        # Zero actions (robot action doesn't matter for wind — we want prediction residual)
        zero_acts = torch.zeros(1, window, act_dim, device=device)
        act_emb = model.action_encoder(zero_acts)

        # Predict from context
        pred = model.predict(ctx, act_emb)  # (1, W, D)

        # Compare predicted[t] with actual[t+1] for t=0..W-2
        for step in range(1, window):
            if start + step < T:
                pred_emb = pred[0, step - 1]  # predicted embedding for step
                actual_emb = targets[0, step - 1] if step - 1 < targets.shape[1] else None
                if actual_emb is not None:
                    cos_dist = 1.0 - F.cosine_similarity(
                        pred_emb.unsqueeze(0), actual_emb.unsqueeze(0)).item()
                    mse = F.mse_loss(pred_emb, actual_emb).item()
                    step_errors[step].append(cos_dist)

                    # Store residual vector for this step
                    if step == window - 1:
                        residual = (pred_emb - actual_emb).cpu().numpy()
                        prediction_residuals.append(residual)

    if not step_errors[1]:
        return None

    # Build feature vector from prediction error statistics
    features = []

    # Per-step error statistics
    for step in range(1, window):
        errs = step_errors[step]
        if errs:
            features.extend([np.mean(errs), np.std(errs), np.max(errs)])
        else:
            features.extend([0.0, 0.0, 0.0])

    # Error growth rate (how fast prediction degrades)
    step_means = [np.mean(step_errors[s]) for s in range(1, window) if step_errors[s]]
    if len(step_means) > 1:
        growth = np.diff(step_means)
        features.extend([np.mean(growth), np.std(growth)])
    else:
        features.extend([0.0, 0.0])

    # Prediction residual statistics (PCA of residuals would be ideal but keep simple)
    if prediction_residuals:
        residuals = np.stack(prediction_residuals)
        res_norms = np.linalg.norm(residuals, axis=1)
        features.extend([np.mean(res_norms), np.std(res_norms)])
        # Directional consistency of residuals (wind should push residuals in consistent direction)
        if len(residuals) > 1:
            mean_dir = residuals.mean(axis=0)
            mean_dir_norm = np.linalg.norm(mean_dir)
            avg_norm = np.mean(np.linalg.norm(residuals, axis=1))
            consistency = mean_dir_norm / (avg_norm + 1e-8)
            features.append(consistency)
        else:
            features.append(0.0)
    else:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def run_probe(features, targets, label, pca_dims=None):
    """Ridge regression probe with CV."""
    X = np.stack(features)
    y = np.array(targets)
    n = len(y)

    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X - X_mean) / X_std

    if pca_dims and pca_dims < X_norm.shape[1]:
        actual = min(pca_dims, n - 1, X_norm.shape[1])
        pca = PCA(n_components=actual, random_state=42)
        X_norm = pca.fit_transform(X_norm)
        print(f"  PCA: {X.shape[1]}D → {actual}D ({pca.explained_variance_ratio_.sum():.1%} var)")

    kf = KFold(n_splits=min(5, n), shuffle=True, random_state=42)

    best_r2, best_alpha = -float('inf'), 1.0
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        scores = cross_val_score(Ridge(alpha=alpha), X_norm, y, cv=kf, scoring='r2')
        if scores.mean() > best_r2:
            best_r2 = scores.mean()
            best_alpha = alpha

    model = Ridge(alpha=best_alpha)
    cv_r2 = cross_val_score(model, X_norm, y, cv=kf, scoring='r2')
    cv_mse = -cross_val_score(model, X_norm, y, cv=kf, scoring='neg_mean_squared_error')

    model.fit(X_norm, y)
    train_r2 = r2_score(y, model.predict(X_norm))

    print(f"  {label}: CV R²={cv_r2.mean():.3f}±{cv_r2.std():.3f}, train R²={train_r2:.3f}, alpha={best_alpha}")
    return {
        'cv_r2_mean': float(cv_r2.mean()),
        'cv_r2_std': float(cv_r2.std()),
        'cv_mse_mean': float(cv_mse.mean()),
        'train_r2': float(train_r2),
        'best_alpha': float(best_alpha),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--model-type', required=True, choices=['tiny', 'small', 'dinov2'])
    parser.add_argument('--wind-data', required=True)
    parser.add_argument('--window', type=int, default=5, help='Prediction window size')
    parser.add_argument('--pca-dims', type=int, default=10)
    parser.add_argument('--output', default='wind_probe_predictor_results.json')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    with open(args.wind_data) as f:
        wind_dataset = json.load(f)

    print(f"Loading full JEPA model ({args.model_type})...")
    model, embed_dim, hist_size = load_full_model(args.ckpt, args.model_type, device)

    # Infer action dim from model
    act_w = list(model.action_encoder.parameters())[0]
    act_dim = act_w.shape[1] if len(act_w.shape) >= 2 else 4
    print(f"  Action dim: {act_dim}, History size: {hist_size}")

    features = []
    wind_speeds = []
    session_names = []
    session_splits = []

    for entry in sorted(wind_dataset['sessions'], key=lambda x: x['session_path']):
        session_dir = Path(entry['session_path'])
        wind_speed = entry.get('wind_speed_10m_ms')
        if wind_speed is None or not session_dir.exists():
            continue

        print(f"  {session_dir.name}: wind={wind_speed:.1f} m/s ...", end=" ", flush=True)

        frames = extract_video_frames(session_dir, max_frames=200)
        if frames is None:
            print("no video")
            continue

        feat = compute_predictor_features(model, frames, device, window=args.window, act_dim=act_dim)
        if feat is None:
            print("no features")
            continue

        features.append(feat)
        wind_speeds.append(wind_speed)
        session_names.append(session_dir.name)
        session_splits.append(entry.get('split', 'unknown'))
        print(f"{len(frames)} frames → {feat.shape[0]}D feature")

    if len(features) < 5:
        print(f"Only {len(features)} sessions — need at least 5")
        return

    print(f"\n{'='*60}")
    print(f"PREDICTOR-BASED WIND PROBE")
    print(f"{'='*60}")
    print(f"Sessions: {len(features)}, Feature dim: {features[0].shape[0]}")
    print(f"Wind: {np.min(wind_speeds):.1f}-{np.max(wind_speeds):.1f} m/s (mean {np.mean(wind_speeds):.1f}±{np.std(wind_speeds):.1f})")

    pca = args.pca_dims if args.pca_dims > 0 else None

    print(f"\n--- Wind Speed (predictor features) ---")
    speed_res = run_probe(features, wind_speeds, "speed", pca_dims=pca)

    print(f"\n--- Random Baseline ---")
    np.random.seed(42)
    rand_feats = [np.random.randn(features[0].shape[0]) for _ in features]
    rand_res = run_probe(rand_feats, wind_speeds, "random", pca_dims=pca)

    results = {
        'model': {'checkpoint': str(args.ckpt), 'type': args.model_type},
        'config': {'window': args.window, 'pca_dims': args.pca_dims},
        'data': {'n_sessions': len(features), 'sessions': session_names,
                 'splits': session_splits, 'wind_speeds': wind_speeds},
        'wind_speed_probe': speed_res,
        'random_baseline': rand_res,
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
