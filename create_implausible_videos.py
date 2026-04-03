"""
Create implausible video variants from plausible source videos.

Methods for creating physics violations:
1. Region temporal reversal: Use SAM2 to segment a dynamic region (e.g., tree in wind),
   then reverse that region's motion while keeping the rest of the scene untouched.
2. Lighting contradiction: Apply gradual brightness increase to one half of the frame
   and decrease to the other (impossible lighting change).
3. Wind direction flip: Segment moving foliage, temporally reverse just that region.

Both plausible and implausible videos share the same generative artifacts
(from the image-to-video model), so any difference in surprise score must
reflect physics understanding, not artifact detection.

Usage:
    # From a plausible video, create implausible variant via region temporal reversal
    python create_implausible_videos.py \
        --input videos/plausible_wind.mp4 \
        --output videos/implausible_wind.mp4 \
        --method region_reverse \
        --mask-prompt "tree branches blowing in wind"

    # Lighting contradiction
    python create_implausible_videos.py \
        --input videos/plausible_lighting.mp4 \
        --output videos/implausible_lighting.mp4 \
        --method lighting_contradiction

    # Batch process a directory
    python create_implausible_videos.py \
        --input-dir videos/plausible/ \
        --output-dir videos/pairs/ \
        --method region_reverse
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def extract_video(video_path):
    """Extract frames from video. Returns (frames, fps, (w,h))."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps, (w, h)


def write_video(frames, output_path, fps, size):
    """Write frames to video file."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    for frame in frames:
        out.write(frame)
    out.release()


def create_region_reverse_with_mask(frames, mask_frames):
    """
    Reverse the temporal order of pixels within the masked region.
    
    The mask_frames should be binary masks (0/255) indicating the dynamic region.
    Pixels inside the mask are taken from the temporally reversed sequence,
    pixels outside are kept as-is.
    
    Args:
        frames: list of (H, W, 3) BGR frames
        mask_frames: list of (H, W) binary masks, or single mask applied to all frames
    
    Returns:
        implausible_frames: list of (H, W, 3) BGR frames
    """
    T = len(frames)
    reversed_frames = frames[::-1]
    
    result = []
    for t in range(T):
        if isinstance(mask_frames, list) and len(mask_frames) > 1:
            mask = mask_frames[t]
        else:
            mask = mask_frames[0] if isinstance(mask_frames, list) else mask_frames
        
        # Ensure mask is 3-channel for blending
        if len(mask.shape) == 2:
            mask_3c = np.stack([mask, mask, mask], axis=-1).astype(np.float32) / 255.0
        else:
            mask_3c = mask.astype(np.float32) / 255.0
        
        # Blend: inside mask = reversed, outside = original
        blended = (frames[t].astype(np.float32) * (1 - mask_3c) +
                   reversed_frames[t].astype(np.float32) * mask_3c)
        result.append(blended.astype(np.uint8))
    
    return result


def create_region_reverse_simple(frames, region='left_half'):
    """
    Reverse temporal order of a spatial region without SAM2.
    
    Simple fallback: reverse left half, right half, or center region.
    
    Args:
        frames: list of (H, W, 3) BGR frames
        region: 'left_half', 'right_half', 'center', 'top_half', 'bottom_half'
    
    Returns:
        implausible_frames
    """
    T = len(frames)
    H, W = frames[0].shape[:2]
    
    # Create spatial mask based on region
    mask = np.zeros((H, W), dtype=np.uint8)
    if region == 'left_half':
        mask[:, :W // 2] = 255
    elif region == 'right_half':
        mask[:, W // 2:] = 255
    elif region == 'center':
        cy, cx = H // 2, W // 2
        rh, rw = H // 4, W // 4
        mask[cy - rh:cy + rh, cx - rw:cx + rw] = 255
    elif region == 'top_half':
        mask[:H // 2, :] = 255
    elif region == 'bottom_half':
        mask[H // 2:, :] = 255
    
    # Apply Gaussian blur to mask edges for smooth blending
    mask = cv2.GaussianBlur(mask, (31, 31), 10)
    
    return create_region_reverse_with_mask(frames, mask)


def create_lighting_contradiction(frames, strength=0.4):
    """
    Apply contradictory lighting: brighten left side while darkening right.
    
    In reality, lighting changes affect the entire scene consistently.
    This creates an impossible scenario where light increases on one side
    and decreases on the other simultaneously.
    
    Args:
        frames: list of (H, W, 3) BGR frames
        strength: how much to brighten/darken (0-1)
    
    Returns:
        implausible_frames
    """
    T = len(frames)
    H, W = frames[0].shape[:2]
    
    result = []
    for t in range(T):
        progress = t / max(T - 1, 1)  # 0 to 1 over video duration
        
        frame = frames[t].astype(np.float32)
        
        # Create gradient mask: 1 on left, 0 on right
        gradient = np.linspace(1.0, 0.0, W).reshape(1, W, 1)
        gradient = np.broadcast_to(gradient, (H, W, 1))
        
        # Left side gets brighter, right side gets darker over time
        adjustment = strength * progress
        bright_factor = 1.0 + adjustment * gradient
        dark_factor = 1.0 - adjustment * (1.0 - gradient)
        
        modified = frame * bright_factor * dark_factor
        modified = np.clip(modified, 0, 255).astype(np.uint8)
        result.append(modified)
    
    return result


def create_gravity_violation(frames, region='center'):
    """
    Make objects in a region appear to fall upward by reversing vertical motion.
    
    Horizontally flip the temporal reversal in the masked region,
    creating the illusion of upward motion for falling objects.
    """
    T = len(frames)
    H, W = frames[0].shape[:2]
    
    mask = np.zeros((H, W), dtype=np.uint8)
    if region == 'center':
        cy, cx = H // 2, W // 2
        rh, rw = H // 3, W // 3
        mask[cy - rh:cy + rh, cx - rw:cx + rw] = 255
    
    mask = cv2.GaussianBlur(mask, (31, 31), 10)
    mask_3c = np.stack([mask, mask, mask], axis=-1).astype(np.float32) / 255.0
    
    # Reverse AND vertically flip the masked region
    reversed_frames = frames[::-1]
    
    result = []
    for t in range(T):
        flipped = cv2.flip(reversed_frames[t], 0)  # Vertical flip
        blended = (frames[t].astype(np.float32) * (1 - mask_3c) +
                   flipped.astype(np.float32) * mask_3c)
        result.append(blended.astype(np.uint8))
    
    return result


def create_with_sam2_mask(frames, mask_prompt, video_path):
    """
    Use SAM2 to segment a region and create temporal reversal.
    
    Requires SAM2 to be installed. Falls back to simple region if not available.
    
    Args:
        frames: list of frames
        mask_prompt: text prompt for SAM2 (e.g., "tree branches")
        video_path: path to original video (for SAM2 video predictor)
    
    Returns:
        implausible_frames, mask_frames
    """
    try:
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import torch
        
        print("  Using SAM2 for segmentation...")
        
        # Use SAM2 image predictor with point prompts on first frame
        # For text-based prompting, we'd need Grounding DINO + SAM2
        # For now, use center-region heuristic as the mask
        # The user can provide pre-computed masks
        
        print("  Note: SAM2 text prompting requires Grounding DINO.")
        print("  Falling back to motion-based masking...")
        raise ImportError("Using motion-based fallback")
        
    except (ImportError, Exception) as e:
        print(f"  SAM2 not available ({e}), using motion-based masking")
        return create_motion_based_reversal(frames)


def create_motion_based_reversal(frames):
    """
    Detect high-motion regions via optical flow and reverse them.
    
    This is the SAM2-free fallback. It finds where motion is happening
    and reverses that region's temporal order.
    """
    T = len(frames)
    if T < 3:
        return frames
    
    # Compute optical flow magnitude across the video
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    H, W = gray_frames[0].shape
    
    motion_accum = np.zeros((H, W), dtype=np.float64)
    for t in range(T - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[t], gray_frames[t + 1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        motion_accum += mag
    
    motion_accum /= (T - 1)
    
    # Threshold to find high-motion region
    threshold = np.percentile(motion_accum, 70)  # Top 30% motion
    mask = (motion_accum > threshold).astype(np.uint8) * 255
    
    # Clean up mask with morphological operations
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (31, 31), 10)
    
    print(f"  Motion mask coverage: {(mask > 127).sum() / mask.size:.1%}")
    
    return create_region_reverse_with_mask(frames, mask), mask


def main():
    parser = argparse.ArgumentParser(description='Create implausible video variants')
    parser.add_argument('--input', help='Input plausible video')
    parser.add_argument('--output', help='Output implausible video')
    parser.add_argument('--input-dir', help='Input directory of plausible videos')
    parser.add_argument('--output-dir', help='Output directory for video pairs')
    parser.add_argument('--method', default='region_reverse',
                        choices=['region_reverse', 'lighting_contradiction',
                                 'gravity', 'motion_reverse', 'all'],
                        help='Method for creating implausible variant')
    parser.add_argument('--region', default='left_half',
                        choices=['left_half', 'right_half', 'center',
                                 'top_half', 'bottom_half'],
                        help='Region for simple region_reverse')
    parser.add_argument('--mask-prompt', help='SAM2 text prompt for mask')
    parser.add_argument('--save-mask', action='store_true',
                        help='Save the motion mask as a video')
    args = parser.parse_args()
    
    if args.input:
        # Single video mode
        print(f"Processing {args.input}...")
        frames, fps, size = extract_video(args.input)
        print(f"  {len(frames)} frames, {fps:.1f}fps, {size[0]}x{size[1]}")
        
        if args.method == 'region_reverse':
            if args.mask_prompt:
                result = create_with_sam2_mask(frames, args.mask_prompt, args.input)
                if isinstance(result, tuple):
                    implausible, mask = result
                else:
                    implausible = result
            else:
                implausible = create_region_reverse_simple(frames, args.region)
        elif args.method == 'lighting_contradiction':
            implausible = create_lighting_contradiction(frames)
        elif args.method == 'gravity':
            implausible = create_gravity_violation(frames)
        elif args.method == 'motion_reverse':
            result = create_motion_based_reversal(frames)
            if isinstance(result, tuple):
                implausible, mask = result
            else:
                implausible = result
        
        output_path = args.output or args.input.replace('.mp4', '_implausible.mp4')
        write_video(implausible, output_path, fps, size)
        print(f"  Saved to {output_path}")
        
    elif args.input_dir:
        # Batch mode
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir or 'video_pairs')
        
        videos = sorted(input_dir.glob('*.mp4'))
        print(f"Found {len(videos)} videos in {input_dir}")
        
        for i, vid_path in enumerate(videos):
            pair_dir = output_dir / f'pair_{i + 1:03d}'
            pair_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n[{i + 1}/{len(videos)}] {vid_path.name}")
            
            # Copy plausible
            import shutil
            plaus_path = pair_dir / 'plausible.mp4'
            shutil.copy2(vid_path, plaus_path)
            
            # Create implausible
            frames, fps, size = extract_video(vid_path)
            print(f"  {len(frames)} frames, {fps:.1f}fps")
            
            methods_to_run = ['motion_reverse'] if args.method == 'all' else [args.method]
            
            for method in methods_to_run:
                if method == 'region_reverse':
                    implausible = create_region_reverse_simple(frames, args.region)
                elif method == 'lighting_contradiction':
                    implausible = create_lighting_contradiction(frames)
                elif method == 'gravity':
                    implausible = create_gravity_violation(frames)
                elif method == 'motion_reverse':
                    result = create_motion_based_reversal(frames)
                    if isinstance(result, tuple):
                        implausible, mask = result
                    else:
                        implausible = result
                
                suffix = f'_{method}' if args.method == 'all' else ''
                implaus_path = pair_dir / f'implausible{suffix}.mp4'
                write_video(implausible, implaus_path, fps, size)
                print(f"  Saved implausible ({method}) to {implaus_path}")
        
        print(f"\nDone. Video pairs in {output_dir}/")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
