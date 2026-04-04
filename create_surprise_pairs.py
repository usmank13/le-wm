"""
Create plausible/implausible video pairs from capture sessions for surprise score eval.

Simple perturbations that don't need diffusion models:
1. Lighting contradiction: left brightens while right darkens (impossible)
2. Temporal reversal: play video backwards in a region (implausible physics)
3. Color channel swap: swap R/G channels mid-video (impossible lighting shift)
4. Sudden freeze: freeze half the frame while other half continues (impossible)
5. Time-of-day jump: abrupt brightness shift mid-video (impossible transition)

Each perturbation has a corresponding plausible version (original or mild augmentation).

Usage:
    python create_surprise_pairs.py \
        --sessions-dir /data/wind_captures/extracted_109 \
        --output-dir /data/surprise_pairs \
        --max-sessions 10 \
        --frames-per-clip 30
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np


def extract_clip(video_path, start_frame=0, num_frames=30, resize=(224, 224)):
    """Extract a clip of sequential frames."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame + num_frames > total:
        start_frame = max(0, total - num_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()
    return frames, fps


def save_clip(frames, output_path, fps=15.0):
    """Save frames as mp4."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


# === Perturbation functions ===

def lighting_contradiction(frames, strength=0.5):
    """Left brightens while right darkens over time — physically impossible."""
    T = len(frames)
    H, W = frames[0].shape[:2]
    result = []
    for t in range(T):
        progress = t / max(T - 1, 1)
        frame = frames[t].astype(np.float32)
        gradient = np.linspace(1.0, 0.0, W).reshape(1, W, 1)
        adj = strength * progress
        factor = 1.0 + adj * (2.0 * gradient - 1.0)
        modified = np.clip(frame * factor, 0, 255).astype(np.uint8)
        result.append(modified)
    return result


def color_channel_swap(frames, swap_at=0.5):
    """Swap R and G channels mid-video — impossible color shift."""
    T = len(frames)
    swap_frame = int(T * swap_at)
    result = []
    for t in range(T):
        if t < swap_frame:
            result.append(frames[t].copy())
        else:
            # Blend into swapped channels over 5 frames
            blend = min(1.0, (t - swap_frame) / 5.0)
            f = frames[t].copy()
            swapped = f.copy()
            swapped[:, :, 0] = f[:, :, 1]  # B channel = G
            swapped[:, :, 1] = f[:, :, 0]  # G channel = B
            blended = (f * (1 - blend) + swapped * blend).astype(np.uint8)
            result.append(blended)
    return result


def sudden_freeze(frames, freeze_side='left'):
    """Freeze one half of the frame while the other continues — impossible."""
    T = len(frames)
    H, W = frames[0].shape[:2]
    freeze_frame = T // 3  # Freeze starts 1/3 through
    frozen = frames[freeze_frame].copy()

    result = []
    for t in range(T):
        f = frames[t].copy()
        if t > freeze_frame:
            if freeze_side == 'left':
                f[:, :W // 2] = frozen[:, :W // 2]
            else:
                f[:, W // 2:] = frozen[:, W // 2:]
        result.append(f)
    return result


def temporal_reversal_region(frames, region='center'):
    """Reverse time in a region while rest plays forward — impossible."""
    T = len(frames)
    H, W = frames[0].shape[:2]

    # Create soft mask for region
    mask = np.zeros((H, W), dtype=np.float32)
    if region == 'center':
        cy, cx = H // 2, W // 2
        rh, rw = H // 3, W // 3
        mask[cy - rh:cy + rh, cx - rw:cx + rw] = 1.0
    elif region == 'bottom':
        mask[H // 2:, :] = 1.0

    mask = cv2.GaussianBlur(mask, (21, 21), 8)
    mask_3c = np.stack([mask] * 3, axis=-1)

    reversed_frames = frames[::-1]
    result = []
    for t in range(T):
        blended = (frames[t].astype(np.float32) * (1 - mask_3c) +
                   reversed_frames[t].astype(np.float32) * mask_3c)
        result.append(blended.astype(np.uint8))
    return result


def brightness_jump(frames, jump_at=0.5, factor=2.0):
    """Abrupt brightness change mid-video — impossible instant lighting shift."""
    T = len(frames)
    jump_frame = int(T * jump_at)
    result = []
    for t in range(T):
        if t < jump_frame:
            result.append(frames[t].copy())
        else:
            brightened = np.clip(frames[t].astype(np.float32) * factor, 0, 255).astype(np.uint8)
            result.append(brightened)
    return result


def plausible_augment(frames, seed=42):
    """Mild plausible augmentation: slight brightness jitter, consistent across frame."""
    rng = np.random.RandomState(seed)
    result = []
    base_factor = 0.9 + rng.rand() * 0.2  # 0.9-1.1
    for f in frames:
        jitter = base_factor + rng.randn() * 0.01  # tiny per-frame noise
        result.append(np.clip(f.astype(np.float32) * jitter, 0, 255).astype(np.uint8))
    return result


PERTURBATIONS = {
    'lighting_contradiction': lighting_contradiction,
    'color_swap': color_channel_swap,
    'freeze_left': lambda f: sudden_freeze(f, 'left'),
    'temporal_reversal': lambda f: temporal_reversal_region(f, 'center'),
    'brightness_jump': brightness_jump,
}


def find_nav_front(session_dir):
    """Find nav_front color video in session."""
    vdir = Path(session_dir) / 'video'
    if not vdir.exists():
        return None
    for f in sorted(vdir.iterdir()):
        if 'nav_front' in f.name and 'color' in f.name and f.suffix == '.mp4':
            return f
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions-dir', nargs='+', required=True,
                        help='One or more directories containing extracted sessions')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-sessions', type=int, default=20)
    parser.add_argument('--frames-per-clip', type=int, default=30)
    parser.add_argument('--clips-per-session', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all sessions
    sessions = []
    for sdir in args.sessions_dir:
        for d in sorted(Path(sdir).iterdir()):
            if d.is_dir() and find_nav_front(d):
                sessions.append(d)

    if len(sessions) > args.max_sessions:
        sessions = random.sample(sessions, args.max_sessions)
    sessions.sort()

    print(f"Processing {len(sessions)} sessions, {len(PERTURBATIONS)} perturbation types")

    manifest = []

    for si, session in enumerate(sessions):
        video = find_nav_front(session)
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        cap.release()

        if total < args.frames_per_clip + 10:
            print(f"  {session.name}: too short ({total} frames), skipping")
            continue

        # Pick random start points for clips
        max_start = total - args.frames_per_clip - 1
        starts = random.sample(range(0, max_start, 10),
                               min(args.clips_per_session, max_start // 10))

        for ci, start in enumerate(starts):
            frames, _ = extract_clip(video, start, args.frames_per_clip)
            if len(frames) < args.frames_per_clip:
                continue

            clip_id = f"{session.name}_clip{ci}"

            # Save plausible (original)
            plaus_dir = os.path.join(args.output_dir, 'plausible')
            os.makedirs(plaus_dir, exist_ok=True)
            plaus_path = os.path.join(plaus_dir, f"{clip_id}.mp4")
            save_clip(frames, plaus_path, fps)

            # Save plausible augmented
            aug_frames = plausible_augment(frames)
            aug_path = os.path.join(plaus_dir, f"{clip_id}_aug.mp4")
            save_clip(aug_frames, aug_path, fps)

            # Save each implausible perturbation
            for pname, pfunc in PERTURBATIONS.items():
                imp_dir = os.path.join(args.output_dir, 'implausible', pname)
                os.makedirs(imp_dir, exist_ok=True)
                imp_frames = pfunc(frames)
                imp_path = os.path.join(imp_dir, f"{clip_id}.mp4")
                save_clip(imp_frames, imp_path, fps)

                manifest.append({
                    'clip_id': clip_id,
                    'session': session.name,
                    'start_frame': start,
                    'plausible': plaus_path,
                    'implausible': imp_path,
                    'perturbation': pname,
                })

        print(f"  [{si+1}/{len(sessions)}] {session.name}: {len(starts)} clips × {len(PERTURBATIONS)} perturbations")

    # Save manifest
    import json
    manifest_path = os.path.join(args.output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump({'pairs': manifest, 'perturbations': list(PERTURBATIONS.keys())}, f, indent=2)

    print(f"\nDone: {len(manifest)} pairs saved to {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
