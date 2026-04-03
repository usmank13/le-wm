"""
Compare surprise scores between baseline and depth-regularized LeWM.

Runs eval_surprise.py on both checkpoints and produces a comparison report.

Usage:
    python eval_surprise_compare.py \
        --video-dir <dir_with_video_pairs> \
        [--baseline-ckpt <path>] \
        [--depth-ckpt <path>] \
        [--output-dir results/surprise/]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


# Default checkpoint paths (in /data/lewm_checkpoints/)
DEFAULT_CHECKPOINTS = {
    'tiny_vanilla': {
        'ckpt': '/data/lewm_checkpoints/tiny_vanilla_epoch100_kw0zx2ub.ckpt',
        'model_type': 'tiny',
        'label': 'Tiny (baseline)',
    },
    'tiny_depth': {
        'ckpt': '/data/lewm_checkpoints/tiny_depth_epoch100_7dlgiyuf.ckpt',
        'model_type': 'tiny',
        'label': 'Tiny + depth reg',
    },
    'small_vanilla': {
        'ckpt': '/data/lewm_checkpoints/small_vanilla_epoch100_vqd54yo7.ckpt',
        'model_type': 'small',
        'label': 'Small (baseline)',
    },
    'dinov2': {
        'ckpt': '/data/lewm_checkpoints/dinov2_small_epoch100_vsrb0dj7.ckpt',
        'model_type': 'dinov2',
        'label': 'DINOv2 (frozen)',
    },
}


def run_surprise_eval(ckpt, model_type, video_dir, output_path, device='cuda'):
    """Run eval_surprise.py as a subprocess."""
    cmd = [
        sys.executable, str(Path(__file__).parent / 'eval_surprise.py'),
        '--ckpt', ckpt,
        '--model-type', model_type,
        '--video-dir', str(video_dir),
        '--output', str(output_path),
        '--device', device,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED:\n{result.stderr}")
        return None
    print(result.stdout)
    with open(output_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Compare surprise scores across models')
    parser.add_argument('--video-dir', required=True, help='Directory with video pairs')
    parser.add_argument('--models', nargs='+', default=['tiny_vanilla', 'tiny_depth'],
                        choices=list(DEFAULT_CHECKPOINTS.keys()),
                        help='Models to compare')
    parser.add_argument('--output-dir', default='results/surprise/',
                        help='Output directory')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_name in args.models:
        model_cfg = DEFAULT_CHECKPOINTS[model_name]
        print(f"\n{'=' * 60}")
        print(f"  {model_cfg['label']} ({model_name})")
        print(f"{'=' * 60}")
        
        output_path = output_dir / f"surprise_{model_name}.json"
        result = run_surprise_eval(
            model_cfg['ckpt'], model_cfg['model_type'],
            args.video_dir, output_path, args.device
        )
        if result:
            all_results[model_name] = result
    
    if len(all_results) < 2:
        print("Need at least 2 model results to compare")
        return
    
    # Build comparison table
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    # Header
    model_labels = [DEFAULT_CHECKPOINTS[m]['label'] for m in args.models if m in all_results]
    print(f"\n{'Metric':<35} " + " ".join(f"{l:>20}" for l in model_labels))
    print("-" * (35 + 21 * len(model_labels)))
    
    # Summary metrics
    for metric_key, metric_label in [
        ('mean_cosine_separation', 'Mean cos separation'),
        ('mean_mse_separation', 'Mean MSE separation'),
        ('mean_relative_cosine_separation', 'Relative cos separation'),
        ('mean_relative_mse_separation', 'Relative MSE separation'),
        ('positive_separation_rate', 'Positive sep rate'),
    ]:
        values = []
        for m in args.models:
            if m in all_results and 'summary' in all_results[m]:
                v = all_results[m]['summary'].get(metric_key, float('nan'))
                values.append(v)
            else:
                values.append(float('nan'))
        
        if 'relative' in metric_key or 'rate' in metric_key:
            fmt = lambda v: f"{v:.1%}"
        else:
            fmt = lambda v: f"{v:.6f}"
        
        print(f"{metric_label:<35} " + " ".join(f"{fmt(v):>20}" for v in values))
    
    # Per-pair breakdown
    print(f"\n{'Per-pair cosine separation':}")
    print(f"{'Pair':<25} " + " ".join(f"{l:>20}" for l in model_labels))
    print("-" * (25 + 21 * len(model_labels)))
    
    # Get pair names from first result
    first_model = [m for m in args.models if m in all_results][0]
    pair_names = [p['name'] for p in all_results[first_model]['pairs']]
    
    for i, name in enumerate(pair_names):
        values = []
        for m in args.models:
            if m in all_results and i < len(all_results[m]['pairs']):
                v = all_results[m]['pairs'][i]['separation']['cosine_distance_diff']
                values.append(v)
            else:
                values.append(float('nan'))
        print(f"{name:<25} " + " ".join(f"{v:>20.6f}" for v in values))
    
    # Save combined results
    combined = {
        'models': {m: DEFAULT_CHECKPOINTS[m] for m in args.models if m in all_results},
        'results': all_results,
        'comparison': {
            m: all_results[m].get('summary', {})
            for m in args.models if m in all_results
        }
    }
    combined_path = output_dir / 'surprise_comparison.json'
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")


if __name__ == '__main__':
    main()
