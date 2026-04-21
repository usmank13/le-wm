import torch


class DINOv2Encoder(torch.nn.Module):
    """Lightweight wrapper around DINOv2 to match HF-style encoder output."""

    def __init__(self, freeze=True):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        self.config = type('Config', (), {'hidden_size': 384})()
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x, interpolate_pos_encoding=True):
        features = self.model.forward_features(x)
        cls_token = features['x_norm_clstoken'].unsqueeze(1)
        patch_tokens = features['x_norm_patchtokens']
        hidden_states = torch.cat([cls_token, patch_tokens], dim=1)
        return type('Output', (), {'last_hidden_state': hidden_states})()
