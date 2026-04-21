"""Shared helpers for LeWM evaluation scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_type: str
    checkpoint: str | None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    'tiny_depth': ModelSpec('tiny_depth', 'tiny', '/data/lewm_checkpoints/tiny_depth_epoch100_7dlgiyuf.ckpt'),
    'tiny_vanilla': ModelSpec('tiny_vanilla', 'tiny', '/data/lewm_checkpoints/tiny_vanilla_epoch100_kw0zx2ub.ckpt'),
    'small_vanilla': ModelSpec('small_vanilla', 'small', '/data/lewm_checkpoints/small_vanilla_epoch100_vqd54yo7.ckpt'),
    'dino': ModelSpec('dino', 'dinov2', '/data/lewm_checkpoints/dinov2_small_epoch100_vsrb0dj7.ckpt'),
}


def resolve_model_spec(label: str | None = None, model_type: str | None = None, checkpoint: str | None = None) -> ModelSpec:
    if label:
        if label not in MODEL_REGISTRY:
            raise KeyError(f'Unknown model label: {label}')
        spec = MODEL_REGISTRY[label]
        if checkpoint and spec.checkpoint != checkpoint:
            return ModelSpec(label=label, model_type=spec.model_type, checkpoint=checkpoint)
        return spec
    if model_type is None:
        raise ValueError('Need either model label or model_type')
    return ModelSpec(label=label or Path(checkpoint or model_type).stem, model_type=model_type, checkpoint=checkpoint)


@dataclass
class EvalMetadata:
    eval_name: str
    model_label: str
    model_type: str
    checkpoint: str | None
    train_data: str | None
    eval_data: str | None
    seed: int | None
    extra: dict[str, Any]


def make_output(metadata: EvalMetadata, summary: dict[str, Any], per_item: list[dict[str, Any]] | None = None, **extra):
    out = {
        'metadata': {
            'eval_name': metadata.eval_name,
            'model_label': metadata.model_label,
            'model_type': metadata.model_type,
            'checkpoint': metadata.checkpoint,
            'train_data': metadata.train_data,
            'eval_data': metadata.eval_data,
            'seed': metadata.seed,
            **metadata.extra,
        },
        'summary': summary,
    }
    if per_item is not None:
        out['per_item'] = per_item
    out.update(extra)
    return out


def write_json(path: str | Path, payload: dict[str, Any]):
    Path(path).write_text(json.dumps(payload, indent=2))


def compute_local_action_mismatch(h5_path: str | Path) -> float:
    with h5py.File(h5_path, 'r') as f:
        proprio = f['proprio'][:]
        action = f['action'][:]
        ep_len = f['ep_len'][:]
        ep_offset = f['ep_offset'][:]
    off, ln = ep_offset[0], ep_len[0]
    gt_xy = proprio[off:off + ln, :2]
    gt_heading = proprio[off:off + ln - 1, 2]
    gt_world = gt_xy[1:] - gt_xy[:-1]
    recon_local = []
    for d, th in zip(gt_world, gt_heading):
        c, s = np.cos(th), np.sin(th)
        rot_inv = np.array([[c, s], [-s, c]], dtype=np.float32)
        recon_local.append(rot_inv @ d)
    recon_local = np.stack(recon_local)
    return float(np.linalg.norm(recon_local - action[off:off + ln - 1], axis=1).mean())


def assert_local_action_semantics(h5_path: str | Path, tol: float = 1e-3):
    mismatch = compute_local_action_mismatch(h5_path)
    if mismatch > tol:
        raise ValueError(f'action/proprio mismatch too large for local-action trajectory eval: mean l2={mismatch:.6f}')
    return mismatch
