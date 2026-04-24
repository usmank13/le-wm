import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class RepLinear(nn.Module):
    """Affine-only reparameterizable linear layer.

    Training-time structure (branches summed):
      - main: full-rank ``nn.Linear(in, out, bias)``
      - low-rank (optional): ``nn.Linear(in, r, bias=False)`` then
        ``nn.Linear(r, out, bias=False)`` — the up projection is zero-initialized
        so at construction this branch contributes exactly zero.
      - identity (optional, requires ``in == out``): residual ``y += x``.

    This is intentionally affine-only (no BN/LN in branches): the spec calls for
    easier attribution and clean folding. At inference time ``fold()`` returns a
    single ``nn.Linear`` equivalent to the summed branches.

    Per-forward branch contribution L2 norms are collected on ``self._last_contribs``
    only when ``self._collect_stats`` is True — default off, zero overhead.
    """

    _collect_stats: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        low_rank: int = 0,
        identity: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.main = nn.Linear(in_features, out_features, bias=bias)
        self.low_rank = int(low_rank)
        if self.low_rank > 0:
            self.lr_down = nn.Linear(in_features, self.low_rank, bias=False)
            self.lr_up = nn.Linear(self.low_rank, out_features, bias=False)
            # zero-init the up projection so initial output == main branch output
            with torch.no_grad():
                self.lr_up.weight.zero_()
        self.has_identity = bool(identity) and (in_features == out_features)
        if identity and not self.has_identity:
            raise ValueError(
                f"RepLinear identity branch requires in_features==out_features, "
                f"got {in_features} vs {out_features}"
            )
        self._last_contribs: dict = {}

    def forward(self, x):
        out = self.main(x)
        if self._collect_stats:
            self._last_contribs = {"main": out.detach().float().norm()}
        if self.low_rank > 0:
            lr = self.lr_up(self.lr_down(x))
            if self._collect_stats:
                self._last_contribs["low_rank"] = lr.detach().float().norm()
            out = out + lr
        if self.has_identity:
            if self._collect_stats:
                self._last_contribs["identity"] = x.detach().float().norm()
            out = out + x
        return out

    @torch.no_grad()
    def fold(self) -> nn.Linear:
        """Merge all training-time branches into one plain ``nn.Linear``."""
        W = self.main.weight.detach().clone()
        b = self.main.bias.detach().clone() if self.main.bias is not None else None
        if self.low_rank > 0:
            W = W + self.lr_up.weight.detach() @ self.lr_down.weight.detach()
        if self.has_identity:
            W = W + torch.eye(self.in_features, device=W.device, dtype=W.dtype)
        folded = nn.Linear(self.in_features, self.out_features, bias=(b is not None))
        folded.weight.copy_(W)
        if b is not None:
            folded.bias.copy_(b)
        return folded


def enable_stats(module: nn.Module, flag: bool = True) -> None:
    """Toggle RepLinear branch-contribution + Attention entropy collection.

    Walks ``module`` and sets per-instance flags on ``RepLinear`` and ``Attention``
    submodules. Collection is opt-in per forward; leaving it off incurs zero
    overhead (Attention keeps the fused SDPA path).
    """
    for m in module.modules():
        if isinstance(m, RepLinear):
            m._collect_stats = bool(flag)
        if isinstance(m, Attention):
            m._collect_attn_stats = bool(flag)


def pop_stats(module: nn.Module) -> dict:
    """Harvest the most recent forward's diagnostics and clear them.

    Returns ``{"attn_entropy": {name: scalar_tensor}, "branch_contribs": {name: {branch: scalar_tensor}}}``.
    Scalars are detached tensors; callers are responsible for ``.item()`` /
    logging conversions.
    """
    stats = {"attn_entropy": {}, "branch_contribs": {}}
    for name, m in module.named_modules():
        if isinstance(m, Attention):
            ent = getattr(m, "_last_attn_entropy", None)
            if ent is not None:
                stats["attn_entropy"][name] = ent
                m._last_attn_entropy = None
        if isinstance(m, RepLinear):
            if m._last_contribs:
                stats["branch_contribs"][name] = dict(m._last_contribs)
                m._last_contribs = {}
    return stats

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers.

    When ``rep_cfg is None`` the module is byte-identical to the legacy
    implementation: ``nn.Sequential`` with two plain ``nn.Linear`` layers and
    matching state-dict keys (``net.1.*`` / ``net.4.*``). This preserves
    backward compatibility with existing checkpoints and configs.

    When ``rep_cfg`` is a dict, both MLP-up (``net.1``) and MLP-down (``net.4``)
    become ``RepLinear``. Supported keys:
      - ``low_rank`` (int, default 0): rank of the low-rank side branch
      - ``identity`` (bool, default False): add identity branch on MLP-down
        (MLP-up has in!=out whenever ``dim != hidden_dim``, so its identity
        branch is silently skipped)
    """

    def __init__(self, dim, hidden_dim, dropout=0.0, rep_cfg=None):
        super().__init__()
        if rep_cfg is None:
            # legacy path — unchanged state-dict keys
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
            )
        else:
            low_rank = int(rep_cfg.get("low_rank", 0))
            use_identity = bool(rep_cfg.get("identity", False))
            up = RepLinear(
                dim, hidden_dim,
                low_rank=low_rank,
                identity=use_identity and (dim == hidden_dim),
            )
            down = RepLinear(
                hidden_dim, dim,
                low_rank=low_rank,
                identity=use_identity and (dim == hidden_dim),
            )
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                up,
                nn.GELU(),
                nn.Dropout(dropout),
                down,
                nn.Dropout(dropout),
            )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking.

    When ``self._collect_attn_stats`` is True, falls back from the fused
    ``F.scaled_dot_product_attention`` kernel to an explicit softmax path so
    the attention distribution can be inspected, and stores the mean per-token
    entropy (over heads/queries/batch) in ``self._last_attn_entropy`` as a
    detached scalar tensor. Default is False → no perf impact on training.
    """

    _collect_attn_stats: bool = False

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self._last_attn_entropy = None

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        if self._collect_attn_stats:
            # Explicit path: mirrors SDPA numerics while exposing probs for entropy.
            T = q.size(-2)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if causal:
                mask = torch.ones(T, T, device=scores.device, dtype=torch.bool).triu(1)
                scores = scores.masked_fill(mask, float("-inf"))
            probs = scores.softmax(dim=-1)
            with torch.no_grad():
                # Mean per-(query,head,batch) Shannon entropy in nats.
                # ``xlogy`` handles the p=0 edge cleanly (0*log(0) := 0).
                entropy = -torch.special.xlogy(probs, probs).sum(dim=-1)
                self._last_attn_entropy = entropy.mean().detach()
            if self.training and drop > 0:
                probs = F.dropout(probs, drop)
            out = torch.matmul(probs, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, rep_cfg=None):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout, rep_cfg=rep_cfg)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, rep_cfg=None):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout, rep_cfg=rep_cfg)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
        rep_cfg=None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout, rep_cfg=rep_cfg)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        rep_cfg=None,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        # ``rep_cfg`` defaults to None → byte-identical to legacy predictor:
        # same nn.Linear MLPs, same state-dict keys, backward-compatible with
        # every existing checkpoint. Set it to e.g. {"low_rank": 16} to opt
        # into Phase 1 reparameterized MLP branches.
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
            rep_cfg=rep_cfg,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
