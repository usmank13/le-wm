"""Shared checkpoint loading helpers for LeWM eval scripts."""

from pathlib import Path

import torch
from transformers import ViTConfig, ViTModel

from dinov2_utils import DINOv2Encoder
from eval_wind_probe_predictor import infer_predictor_config
from jepa import JEPA
from module import MLP, ARPredictor, Embedder


def load_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return ckpt['state_dict'] if 'state_dict' in ckpt else ckpt


def build_encoder_from_sd(sd, freeze=True):
    projector_in = sd['model.projector.net.0.weight'].shape[1]

    if any(k.startswith('model.encoder.model.') for k in sd):
        encoder = DINOv2Encoder(freeze=freeze)
        hidden_dim = 384
    else:
        num_heads = 6 if projector_in == 192 else 12
        config = ViTConfig(
            hidden_size=projector_in,
            num_hidden_layers=12,
            num_attention_heads=num_heads,
            intermediate_size=projector_in * 4,
            patch_size=14,
            image_size=224,
        )
        encoder = ViTModel(config, add_pooling_layer=False)
        hidden_dim = projector_in

    enc_sd = {k.replace('model.encoder.', ''): v for k, v in sd.items() if k.startswith('model.encoder.')}
    encoder.load_state_dict(enc_sd, strict=True)
    return encoder, hidden_dim


def build_projector_from_sd(sd):
    projector_in = sd['model.projector.net.0.weight'].shape[1]
    projector = MLP(input_dim=projector_in, output_dim=192, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    proj_sd = {k.replace('model.projector.', ''): v for k, v in sd.items() if k.startswith('model.projector.')}
    projector.load_state_dict(proj_sd, strict=True)
    return projector


def load_encoder_projector(ckpt_path, device, freeze=True):
    sd = load_state_dict(ckpt_path)
    encoder, _ = build_encoder_from_sd(sd, freeze=freeze)
    projector = build_projector_from_sd(sd)
    encoder = encoder.to(device).eval()
    projector = projector.to(device).eval()
    return encoder, projector


def load_full_jepa(ckpt_path, device, freeze_encoder=True):
    sd = load_state_dict(ckpt_path)
    encoder, hidden_dim = build_encoder_from_sd(sd, freeze=freeze_encoder)
    projector = build_projector_from_sd(sd)

    embed_dim = 192
    pcfg = infer_predictor_config(sd)
    predictor_out_dim = sd['model.pred_proj.net.0.weight'].shape[1]

    predictor = ARPredictor(
        num_frames=pcfg['num_frames'],
        depth=pcfg['depth'],
        heads=pcfg['heads'],
        mlp_dim=pcfg['mlp_dim'],
        input_dim=pcfg['input_dim'],
        hidden_dim=pcfg['hidden_dim'],
        output_dim=predictor_out_dim,
        dim_head=pcfg['dim_head'],
    )
    pred_proj = MLP(input_dim=predictor_out_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)

    act_w = sd.get('model.action_encoder.patch_embed.weight')
    act_dim = act_w.shape[1] if act_w is not None else 4
    action_encoder = Embedder(input_dim=act_dim, emb_dim=embed_dim)

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )

    model_sd = {k.replace('model.', '', 1): v for k, v in sd.items() if k.startswith('model.')}
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    model = model.to(device).eval()
    return model, embed_dim, pcfg['num_frames'], act_dim, {'missing': missing, 'unexpected': unexpected, 'hidden_dim': hidden_dim, 'predictor_cfg': pcfg}
