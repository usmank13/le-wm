import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

class SimpleImagePreprocess:
    """Convert uint8 (B,T,C,H,W) pixels to float32, normalize with ImageNet stats."""
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

    def __init__(self, source='pixels', target='pixels'):
        self.source = source
        self.target = target

    def __call__(self, sample):
        x = sample[self.source].float() / 255.0
        # x shape: (T, C, H, W) — add batch dim for normalization
        mean = self.MEAN.squeeze(0)  # (1, C, 1, 1)
        std = self.STD.squeeze(0)
        x = (x - mean) / std
        sample[self.target] = x
        return sample


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    return SimpleImagePreprocess(source=source, target=target)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")