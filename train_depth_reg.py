"""Train LeWM with depth-regularized embeddings.

The encoder processes both RGB and depth (as 3-channel image).
An auxiliary cosine loss pulls RGB embeddings toward depth embeddings,
encouraging the encoder to learn structure-aware representations.
At inference, only RGB is needed.
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack, SimpleImagePreprocess


class DepthImagePreprocess:
    """Convert uint8 (T, H, W) depth to float (T, 3, H, W) normalized with ImageNet stats."""

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, source='depth', target='depth'):
        self.source = source
        self.target = target

    def __call__(self, sample):
        x = sample[self.source].float() / 255.0  # (T, H, W)
        x = x.unsqueeze(1).expand(-1, 3, -1, -1)  # (T, 3, H, W)
        x = (x - self.MEAN) / self.STD
        sample[self.target] = x
        return sample


def lejepa_depth_forward(self, batch, stage, cfg):
    """JEPA forward with depth regularization."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd_sigreg = cfg.loss.sigreg.weight
    lambd_depth = cfg.loss.get('depth_reg_weight', 0.1)

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # Standard JEPA encode (RGB)
    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    # Standard losses
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    # Depth regularization: encode depth through same encoder, pull RGB emb toward depth emb
    if "depth" in batch:
        from einops import rearrange
        depth = batch["depth"].float()
        b, t = depth.shape[:2]
        depth_flat = rearrange(depth, "b t ... -> (b t) ...")

        with torch.no_grad():  # depth embeddings are targets, don't backprop through them
            depth_out = self.model.encoder(depth_flat, interpolate_pos_encoding=True)
            depth_cls = depth_out.last_hidden_state[:, 0]
            depth_emb = self.model.projector(depth_cls)
            depth_emb = rearrange(depth_emb, "(b t) d -> b t d", b=b)

        # Cosine similarity loss: pull RGB embeddings toward depth embeddings
        rgb_flat = rearrange(emb, "b t d -> (b t) d")
        depth_flat = rearrange(depth_emb, "b t d -> (b t) d")
        output["depth_reg_loss"] = (1.0 - F.cosine_similarity(rgb_flat, depth_flat, dim=-1)).mean()
    else:
        output["depth_reg_loss"] = torch.tensor(0.0, device=emb.device)

    output["loss"] = (output["pred_loss"]
                      + lambd_sigreg * output["sigreg_loss"]
                      + lambd_depth * output["depth_reg_loss"])

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm_tiny_depth")
def run(cfg):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

    transforms = [
        get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size),
    ]

    # Add depth preprocessing if depth is in keys_to_load
    if 'depth' in cfg.data.dataset.keys_to_load:
        transforms.append(DepthImagePreprocess(source='depth', target='depth'))

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels") or col == "depth":
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_depth_forward, cfg=cfg),
        optim=optimizers,
    )

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()


if __name__ == "__main__":
    run()
