from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass(frozen=True)
class UGPFormerConfig:
    in_channels: int = 32
    num_classes: int = 2

    branch_dim: int = 24
    micro_kernel: int = 5
    macro_kernel: int = 25
    stem_pool: int = 8
    initial_input_variance: float = 1e-3

    hidden_dim: int = 48
    num_plg_blocks: int = 7
    local_kernel: int = 5
    attention_heads: int = 4
    ffn_dim: Optional[int] = None
    dropout_rate: float = 0.3

    mc_samples: int = 20

    sampling_rate: float = 256.0
    num_bands: int = 5
    max_band_attenuation: float = 0.5
    augmentation_noise_max: float = 0.15
    consistency_weight: float = 1.0

    @classmethod
    def from_dict(cls, cfg: Optional[Dict]) -> "UGPFormerConfig":
        if not cfg:
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in known})

    def __post_init__(self) -> None:
        if self.hidden_dim != 2 * self.branch_dim:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must equal 2 * branch_dim "
                f"({2 * self.branch_dim})."
            )
        if self.ffn_dim is None:
            object.__setattr__(self, "ffn_dim", 2 * self.hidden_dim)


class ADFConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Optional[int] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        self.eps = eps

    def forward(self, mu_in: Tensor, var_in: Tensor) -> Tuple[Tensor, Tensor]:
        mu_out = self.conv(mu_in)
        var_out = F.conv1d(
            var_in,
            self.conv.weight.pow(2),
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        return mu_out, var_out + self.eps


class ADFAvgPool1d(nn.Module):
    def __init__(self, kernel: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.pool = nn.AvgPool1d(kernel_size=kernel, stride=kernel)

    def forward(self, mu: Tensor, var: Tensor) -> Tuple[Tensor, Tensor]:
        return self.pool(mu), self.pool(var) / self.kernel


class MultiScaleProbabilisticEmbedding(nn.Module):
    def __init__(self, cfg: UGPFormerConfig) -> None:
        super().__init__()
        self.initial_input_variance = cfg.initial_input_variance
        self.micro = ADFConv1d(cfg.in_channels, cfg.branch_dim, cfg.micro_kernel)
        self.macro = ADFConv1d(cfg.in_channels, cfg.branch_dim, cfg.macro_kernel)
        self.act = nn.GELU()
        self.pool = ADFAvgPool1d(cfg.stem_pool)

    def forward(self, x: Tensor, input_variance: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if input_variance is None:
            input_variance = torch.full_like(x, self.initial_input_variance)

        mu_micro, var_micro = self.micro(x, input_variance)
        mu_macro, var_macro = self.macro(x, input_variance)

        mu = torch.cat([mu_micro, mu_macro], dim=1)
        var = torch.cat([var_micro, var_macro], dim=1)

        var = var * _gelu_derivative(mu).square()
        return self.pool(self.act(mu), var)


def _gelu_derivative(values: Tensor) -> Tensor:
    cdf = 0.5 * (1 + torch.erf(values / math.sqrt(2)))
    pdf = torch.exp(-0.5 * values.square()) / math.sqrt(2 * math.pi)
    return cdf + values * pdf


class ConfidenceGating(nn.Module):
    def __init__(self, init_alpha: float = 1.0, init_beta: float = 0.5) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    def forward(self, mu: Tensor, var: Tensor) -> Tuple[Tensor, Tensor]:
        gate = torch.sigmoid(self.beta - F.softplus(self.alpha) * var)
        return mu * gate, gate


class _ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class PLGBlock(nn.Module):
    def __init__(self, cfg: UGPFormerConfig) -> None:
        super().__init__()
        c = cfg.hidden_dim

        self.local_norm = _ChannelLayerNorm(c)
        self.local_dwconv = nn.Conv1d(c, c, cfg.local_kernel, padding=cfg.local_kernel // 2, groups=c)
        self.local_pwconv = nn.Conv1d(c, c, kernel_size=1)

        self.attn_norm = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, cfg.attention_heads, dropout=cfg.dropout_rate, batch_first=True)

        self.fuse = nn.Conv1d(2 * c, c, kernel_size=1)
        self.ffn_norm = _ChannelLayerNorm(c)
        self.ffn = _FeedForward(c, cfg.ffn_dim, cfg.dropout_rate)

    def forward(self, z: Tensor) -> Tensor:
        local = self.local_pwconv(self.local_dwconv(F.gelu(self.local_norm(z))))

        seq = self.attn_norm(z.transpose(1, 2))
        glob, _ = self.attn(seq, seq, seq, need_weights=False)
        glob = glob.transpose(1, 2)

        fused = self.fuse(torch.cat([local, glob], dim=1)) + z
        return self.ffn(self.ffn_norm(fused)) + fused


class UncertaintyHead(nn.Module):
    def __init__(self, cfg: UGPFormerConfig) -> None:
        super().__init__()
        self.dropout = nn.Dropout(cfg.dropout_rate)
        self.classifier = nn.Linear(cfg.hidden_dim, cfg.num_classes)

    def forward(self, feat: Tensor, *, mc_dropout: bool = False) -> Tensor:
        if mc_dropout:
            feat = F.dropout(feat, p=self.dropout.p, training=True)
        else:
            feat = self.dropout(feat)
        return self.classifier(feat)


class UncertaintyAwareAugmenter(nn.Module):
    _DEFAULT_BANDS: Tuple[Tuple[float, float], ...] = (
        (1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0),
    )

    def __init__(self, cfg: UGPFormerConfig, bands: Optional[Sequence[Tuple[float, float]]] = None) -> None:
        super().__init__()
        self.sampling_rate = cfg.sampling_rate
        self.max_band_attenuation = cfg.max_band_attenuation
        self.noise_max = cfg.augmentation_noise_max
        self.bands = tuple(bands) if bands is not None else self._DEFAULT_BANDS[: cfg.num_bands]

        pooled_dim = 2 * cfg.hidden_dim
        self.policy = nn.Sequential(
            nn.Linear(pooled_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, len(self.bands) + 1),
        )

    def policy_from_features(self, mu: Tensor, var: Tensor) -> Dict[str, Tensor]:
        pooled = torch.cat([mu.mean(dim=-1), var.mean(dim=-1)], dim=1)
        logits = self.policy(pooled)
        band_strength = self.max_band_attenuation * torch.sigmoid(logits[:, : len(self.bands)])
        noise_scale = self.noise_max * torch.sigmoid(logits[:, len(self.bands):])
        return {"band_strength": band_strength, "noise_scale": noise_scale}

    def forward(self, x: Tensor, mu: Tensor, var: Tensor) -> Tensor:
        policy = self.policy_from_features(mu, var)
        band_strength = policy["band_strength"].to(dtype=x.dtype)
        noise_scale = policy["noise_scale"].to(dtype=x.dtype)

        n_time = x.shape[-1]
        spectrum = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(n_time, d=1.0 / self.sampling_rate).to(x.device)

        mask = torch.ones(x.shape[0], 1, spectrum.shape[-1], device=x.device, dtype=x.dtype)
        for idx, (lo, hi) in enumerate(self.bands):
            in_band = ((freqs >= lo) & (freqs < hi)).view(1, 1, -1)
            attenuation = band_strength[:, idx].view(-1, 1, 1)
            mask = torch.where(in_band, mask * (1.0 - attenuation), mask)

        x_masked = torch.fft.irfft(spectrum * mask, n=n_time, dim=-1)
        return x_masked + torch.randn_like(x_masked) * noise_scale.view(-1, 1, 1)


def _entropy(prob: Tensor, eps: float = 1e-8) -> Tensor:
    return -(prob * (prob.clamp_min(eps).log())).sum(dim=-1)


class UGPFormer(nn.Module):
    def __init__(self, cfg: Optional[UGPFormerConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or UGPFormerConfig()
        self.embedding = MultiScaleProbabilisticEmbedding(self.cfg)
        self.gating = ConfidenceGating()
        self.backbone = nn.ModuleList([PLGBlock(self.cfg) for _ in range(self.cfg.num_plg_blocks)])
        self.head = UncertaintyHead(self.cfg)
        self.augmenter = UncertaintyAwareAugmenter(self.cfg)

    def encode(self, x: Tensor) -> Dict[str, Tensor]:
        mu, var = self.embedding(x)
        z, gate = self.gating(mu, var)
        for block in self.backbone:
            z = block(z)

        return {"z_deep": z.mean(dim=-1), "mu_feat": mu, "var_feat": var, "gate": gate}

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        enc = self.encode(x)
        enc["logits"] = self.head(enc["z_deep"])
        return enc

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: Tensor,
        mc_samples: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        n_mc = self.cfg.mc_samples if mc_samples is None else mc_samples
        if n_mc < 1:
            raise ValueError("mc_samples must be positive.")
        previous_modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            feat = self.encode(x)["z_deep"]
            probs_stack = torch.stack(
                [self.head(feat, mc_dropout=True).softmax(dim=-1) for _ in range(n_mc)]
            )
        finally:
            for module, previous in previous_modes:
                module.training = previous

        probs = probs_stack.mean(dim=0)
        total = _entropy(probs)
        data = _entropy(probs_stack).mean(dim=0)
        model_uncertainty = (total - data).clamp_min(0.0)

        return {
            "probs": probs,
            "pred": probs.argmax(dim=-1),
            "total_uncertainty": total,
            "data_uncertainty": data,
            "model_uncertainty": model_uncertainty,
        }


def ugp_former_loss(
    model: UGPFormer,
    x: Tensor,
    y: Tensor,
    consistency_weight: Optional[float] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    lam = model.cfg.consistency_weight if consistency_weight is None else consistency_weight

    out = model(x)
    logits = out["logits"]
    ce = F.cross_entropy(logits, y)
    x_aug = model.augmenter(x, out["mu_feat"].detach(), out["var_feat"].detach())
    out_aug = model(x_aug)
    p_orig = F.softmax(logits.detach(), dim=-1)
    log_p_aug = F.log_softmax(out_aug["logits"], dim=-1)
    kl = F.kl_div(log_p_aug, p_orig, reduction="batchmean")

    loss = ce + lam * kl
    metrics = {
        "loss": loss.detach(),
        "ce": ce.detach(),
        "kl": kl.detach(),
    }
    return loss, metrics


def build_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-2) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


if __name__ == "__main__":
    torch.manual_seed(2021)
    cfg = UGPFormerConfig()
    model = UGPFormer(cfg)
    n_total = sum(p.numel() for p in model.parameters())
    n_infer = n_total - sum(p.numel() for p in model.augmenter.parameters())
    print(f"inference parameters: {n_infer / 1e3:.2f} K")
    x_demo = torch.randn(2, cfg.in_channels, 1024)
    y_demo = torch.tensor([0, 1])
    loss_demo, _ = ugp_former_loss(model, x_demo, y_demo)
    loss_demo.backward()
    pred_demo = model.predict_with_uncertainty(x_demo)
    print("loss:", float(loss_demo.detach()))
    print("MC passes:", cfg.mc_samples)
    print("probs shape:", tuple(pred_demo["probs"].shape))
    print("model uncertainty mean:", round(float(pred_demo["model_uncertainty"].mean()), 4))
