"""Convert TartanGround trajectories to LeWM's monolithic HDF5 format.

TartanGround format (per trajectory):
    image_lcam_front/*.png   — 640x640 RGB
    depth_lcam_front/*.npy   — 640x640 metric depth
    pose_lcam_front.txt      — tx ty tz qx qy qz qw per line

LeWM format:
    pixels:    (N, 224, 224, 3) uint8
    action:    (N, 2) float32  — (dx, dy) displacement in local frame
    proprio:   (N, 5) float32  — (x, y, heading, v_linear, v_angular)
    depth:     (N, 224, 224) uint8 (optional)
    ep_len:    (num_episodes,) int64
    ep_offset: (num_episodes,) int64

Usage:
    python convert_tartanground.py --input /data/tartanground --output /data/lewm_data/tartanground.h5
    python convert_tartanground.py --input /data/tartanground --output /data/lewm_data/tartanground.h5 --with-depth
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation


IMG_SIZE = 224


def quat_to_yaw(qx, qy, qz, qw):
    """Extract yaw (heading) from quaternion."""
    r = Rotation.from_quat([qx, qy, qz, qw])
    return r.as_euler('xyz')[2]  # yaw


def load_poses(pose_file):
    """Load poses from TartanGround text file. Returns (N, 7) array: tx ty tz qx qy qz qw."""
    return np.loadtxt(pose_file)


def poses_to_actions_and_proprio(poses):
    """Compute local-frame displacements and proprioception from global poses.

    Actions: (dx, dy) displacement in the robot's local frame at each step.
    Proprio: (x, y, heading, v_linear, v_angular) — global x/y, heading from quat,
             and approximate velocities (displacement magnitude, heading change).
    """
    n = len(poses)
    positions = poses[:, :3]  # tx, ty, tz
    quats = poses[:, 3:]      # qx, qy, qz, qw

    headings = np.array([quat_to_yaw(*q) for q in quats])

    # Global displacements
    dx_global = np.diff(positions[:, 0])
    dy_global = np.diff(positions[:, 1])

    # Rotate into local frame using current heading
    cos_h = np.cos(-headings[:-1])
    sin_h = np.sin(-headings[:-1])
    dx_local = dx_global * cos_h - dy_global * sin_h
    dy_local = dx_global * sin_h + dy_global * cos_h

    # Actions: (N, 2) — pad last frame with zeros
    actions = np.zeros((n, 2), dtype=np.float32)
    actions[:-1, 0] = dx_local
    actions[:-1, 1] = dy_local

    # Proprio: (x, y, heading, v_linear, v_angular)
    v_linear = np.zeros(n, dtype=np.float32)
    v_linear[:-1] = np.sqrt(dx_global**2 + dy_global**2)

    v_angular = np.zeros(n, dtype=np.float32)
    dheading = np.diff(headings)
    # Wrap to [-pi, pi]
    dheading = (dheading + np.pi) % (2 * np.pi) - np.pi
    v_angular[:-1] = dheading

    proprio = np.stack([
        positions[:, 0],
        positions[:, 1],
        headings,
        v_linear,
        v_angular,
    ], axis=-1).astype(np.float32)

    return actions, proprio


def find_trajectories(root_dir):
    """Find all trajectory directories that have images + poses."""
    root = Path(root_dir)
    trajectories = []

    for img_dir in sorted(root.rglob('image_lcam_front')):
        traj_dir = img_dir.parent
        # Look for pose file (may be named differently)
        pose_file = None
        for name in ['pose_lcam_front.txt', 'pose_lcam.txt', 'pose_left.txt']:
            candidate = traj_dir / name
            if candidate.exists():
                pose_file = candidate
                break

        if pose_file is None:
            print(f"  Skipping {traj_dir} — no pose file found")
            continue

        pngs = sorted(img_dir.glob('*.png'))
        if len(pngs) < 10:
            print(f"  Skipping {traj_dir} — only {len(pngs)} frames")
            continue

        trajectories.append({
            'dir': traj_dir,
            'img_dir': img_dir,
            'pose_file': pose_file,
            'depth_dir': traj_dir / 'depth_lcam_front',
            'n_frames': len(pngs),
        })

    return trajectories


def load_trajectory(traj, with_depth=False):
    """Load images, depth, poses for one trajectory."""
    img_files = sorted(traj['img_dir'].glob('*.png'))
    poses = load_poses(traj['pose_file'])

    n = min(len(img_files), len(poses))
    img_files = img_files[:n]
    poses = poses[:n]

    # Load and resize images
    pixels = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i, f in enumerate(img_files):
        img = cv2.imread(str(f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        pixels[i] = img

    # Depth (optional)
    depth = None
    if with_depth and traj['depth_dir'].exists():
        depth_files = sorted(traj['depth_dir'].glob('*.png'))
        if not depth_files:
            depth_files = sorted(traj['depth_dir'].glob('*.npy'))
        if len(depth_files) >= n:
            depth = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            for i in range(n):
                f = depth_files[i]
                if f.suffix == '.npy':
                    d = np.load(str(f))
                else:
                    # TartanGround: float32 metric depth encoded as 4-byte RGBA PNG
                    raw = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                    d = raw.view(np.float32)[:, :, 0]
                # Quantize metric depth to 8-bit (clip at 50m, scale to 0-255)
                d = np.clip(d, 0, 50.0)
                d = (d / 50.0 * 255).astype(np.uint8)
                d = cv2.resize(d, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                depth[i] = d

    actions, proprio = poses_to_actions_and_proprio(poses)

    return pixels, actions, proprio, depth


def convert(input_dir, output_path, with_depth=False, max_trajectories=None):
    trajectories = find_trajectories(input_dir)
    if not trajectories:
        print(f"No trajectories found under {input_dir}")
        return

    if max_trajectories:
        trajectories = trajectories[:max_trajectories]

    print(f"Found {len(trajectories)} trajectories")

    # First pass: load all trajectories, collect metadata
    all_data = []
    total_frames = 0
    for i, traj in enumerate(trajectories):
        pixels, actions, proprio, depth = load_trajectory(traj, with_depth)
        n = len(pixels)
        all_data.append((pixels, actions, proprio, depth))
        total_frames += n
        print(f"  [{i+1}/{len(trajectories)}] {traj['dir'].name}: {n} frames")

    ep_lengths = np.array([len(d[0]) for d in all_data], dtype=np.int64)
    ep_offsets = np.cumsum([0] + ep_lengths[:-1].tolist()).astype(np.int64)

    print(f"\nTotal: {total_frames} frames, {len(all_data)} episodes")
    print(f"Episode lengths: min={ep_lengths.min()}, max={ep_lengths.max()}, mean={ep_lengths.mean():.0f}")

    # Write HDF5
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    has_depth = with_depth and all_data[0][3] is not None

    with h5py.File(output_path, 'w') as out:
        pixels_ds = out.create_dataset(
            'pixels', shape=(total_frames, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8,
            chunks=(min(256, total_frames), IMG_SIZE, IMG_SIZE, 3))
        action_ds = out.create_dataset(
            'action', shape=(total_frames, 2), dtype=np.float32,
            chunks=(min(1024, total_frames), 2))
        proprio_ds = out.create_dataset(
            'proprio', shape=(total_frames, 5), dtype=np.float32,
            chunks=(min(1024, total_frames), 5))
        if has_depth:
            depth_ds = out.create_dataset(
                'depth', shape=(total_frames, IMG_SIZE, IMG_SIZE), dtype=np.uint8,
                chunks=(min(256, total_frames), IMG_SIZE, IMG_SIZE))

        out.create_dataset('ep_len', data=ep_lengths)
        out.create_dataset('ep_offset', data=ep_offsets)
        out.attrs['source'] = 'tartanground'
        out.attrs['img_size'] = IMG_SIZE

        offset = 0
        for i, (pixels, actions, proprio, depth) in enumerate(all_data):
            n = len(pixels)
            pixels_ds[offset:offset + n] = pixels
            action_ds[offset:offset + n] = actions
            proprio_ds[offset:offset + n] = proprio
            if has_depth and depth is not None:
                depth_ds[offset:offset + n] = depth
            offset += n

    print(f"\nWritten to {output_path}")
    print(f"  pixels:  ({total_frames}, {IMG_SIZE}, {IMG_SIZE}, 3)")
    print(f"  action:  ({total_frames}, 2)")
    print(f"  proprio: ({total_frames}, 5)")
    if has_depth:
        print(f"  depth:   ({total_frames}, {IMG_SIZE}, {IMG_SIZE})")
    print(f"  episodes: {len(ep_lengths)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert TartanGround to LeWM HDF5')
    parser.add_argument('--input', required=True, help='Root dir of TartanGround data')
    parser.add_argument('--output', required=True, help='Output .h5 path')
    parser.add_argument('--with-depth', action='store_true', help='Include depth maps')
    parser.add_argument('--max-trajectories', type=int, default=None,
                        help='Limit number of trajectories (for testing)')
    args = parser.parse_args()
    convert(args.input, args.output, args.with_depth, args.max_trajectories)
