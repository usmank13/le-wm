"""Simple batch runner for LeWM eval scripts across model variants."""

import argparse
import subprocess
import sys
from pathlib import Path

from eval_common import MODEL_REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', required=True, help='Eval script to run')
    parser.add_argument('--models', nargs='+', default=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--extra-args', nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    script = Path(args.script)
    if not script.exists():
        raise FileNotFoundError(script)

    for label in args.models:
        if label not in MODEL_REGISTRY:
            raise KeyError(f'Unknown model label: {label}')
        spec = MODEL_REGISTRY[label]
        output = f'/tmp/{script.stem}_{label}.json'
        cmd = [sys.executable, str(script), '--model-label', label, '--model-type', spec.model_type, '--ckpt', spec.checkpoint, '--output', output]
        cmd.extend(args.extra_args)
        print('\n== Running', label, '==')
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
