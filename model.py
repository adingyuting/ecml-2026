from __future__ import annotations
from dataclasses import asdict, dataclass, fields
import math
import torch
from torch import nn


@dataclass
class ModelConfig:
    nodes: int
    steps: int
    width: int = 64
    depth: int = 3
    heads: int = 1
    cheb_order: int = 3
    context_span: int = 8
    kernels: tuple[int, ...] = (3, 5, 7)
    dropout: float = 0.0
    variant: str = "full"
    single_head: int = 0
    backbone: str = "paper"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_checkpoint(cls, settings):
        if settings.get("backbone") != "paper":
            raise ValueError("Only checkpoints trained with paper UDCGL are supported")
        return cls(**{item.name: settings[item.name] for item in fields(cls) if item.name in settings})


def scaled_laplacian(adjacency):
    a = torch.as_tensor(adjacency, dtype=torch.float32)
    degree = a.sum(-1)
    inv = degree.clamp_min(1e-8).rsqrt()
    identity = torch.eye(len(a), dtype=a.dtype, device=a.device)
    lap = identity - inv[:, None] * a * inv[None, :]
    maximum = torch.linalg.eigvalsh(lap).max().clamp_min(1e-8)
    return 2 * lap / maximum - identity


class SpectralConv(nn.Module):
    def __init__(self, width, order, laplacian):
        super().__init__()
        if order < 1:
            raise ValueError("Spectral order must be positive")
        self.order = order
        self.register_buffer("laplacian", laplacian.to_sparse().coalesce().clone())
        self.projection = nn.Linear(width * order, width)

    def propagate(self, features):
        batch, steps, nodes, width = features.shape
        columns = features.permute(2, 0, 1, 3).reshape(nodes, -1)
        result = torch.sparse.mm(self.laplacian, columns)
        return result.reshape(nodes, batch, steps, width).permute(1, 2, 0, 3)

    def forward(self, x):
        terms = [x]
        if self.order > 1:
            terms.append(self.propagate(x))
        for _ in range(2, self.order):
            terms.append(2 * self.propagate(terms[-1]) - terms[-2])
        return self.projection(torch.cat(terms, -1))


class GatedAttention(nn.Module):
    def __init__(self, width, heads=1, dropout=0.0):
        super().__init__()
        if heads < 1 or width % heads:
            raise ValueError("Width must be divisible by the positive number of attention heads")
        self.heads, self.head_dim = heads, width // heads
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, keys, allowed, previous_logits=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if previous_logits is not None:
            logits = logits + previous_logits
        topology = allowed.to(device=x.device, dtype=torch.bool)[None, None, :, :]
        valid = topology & keys[:, None, None, :].bool()
        valid = valid | ((~valid.any(-1, keepdim=True)) & topology)
        if not bool(valid.any(-1).all()):
            raise ValueError("Attention topology must contain at least self-edges")
        scores = logits.masked_fill(~valid, float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        output = (weights @ v).transpose(1, 2).reshape(batch, length, width)
        return self.out(output), logits


class AxisAttention(nn.Module):
    def __init__(self, config, adjacency, radius=None, residual_norm=False):
        super().__init__()
        self.temporal = GatedAttention(config.width, config.heads, config.dropout)
        self.spatial = GatedAttention(config.width, config.heads, config.dropout)
        self.fusion = nn.Linear(2 * config.width, config.width)
        self.norm = nn.LayerNorm(config.width) if residual_norm else None
        self.radius = radius
        edges = torch.as_tensor(adjacency) > 0
        self.register_buffer("neighbors", edges | torch.eye(config.nodes, dtype=torch.bool, device=edges.device))

    def forward(self, x, key_mask, previous=None, residual=None):
        b, t, n, d = x.shape
        indices = torch.arange(t, device=x.device)
        temporal_allowed = torch.ones(t, t, dtype=torch.bool, device=x.device)
        if self.radius is not None:
            temporal_allowed = (indices[:, None] - indices[None, :]).abs() <= self.radius
        tx = x.permute(0, 2, 1, 3).reshape(b * n, t, d)
        tk = key_mask.permute(0, 2, 1).reshape(b * n, t)
        temp, temp_logits = self.temporal(tx, tk, temporal_allowed,
                                         previous[0] if previous is not None else None)
        temp = temp.reshape(b, n, t, d).permute(0, 2, 1, 3)
        spatial, spatial_logits = self.spatial(x.reshape(b * t, n, d), key_mask.reshape(b * t, n),
                                               self.neighbors,
                                               previous[1] if previous is not None else None)
        spatial = spatial.reshape(b, t, n, d)
        fused = self.fusion(torch.cat([temp, spatial], -1))
        if self.norm is not None:
            fused = self.norm(residual + fused)
        return fused, (temp_logits, spatial_logits)


class MultiScaleTCN(nn.Module):
    def __init__(self, width, kernels):
        super().__init__()
        if not kernels or any(k < 1 or k % 2 == 0 for k in kernels):
            raise ValueError("TCN kernels must be positive odd integers")
        self.branches = nn.ModuleList([nn.Conv1d(width, 2 * width, k, padding=k // 2) for k in kernels])
        self.output_width = width * len(kernels)

    def forward(self, x):
        b, t, n, d = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(b * n, d, t)
        branches = []
        for convolution in self.branches:
            value, gate = convolution(seq).chunk(2, dim=1)
            branches.append((torch.tanh(value) * torch.sigmoid(gate)).reshape(b, n, d, t).permute(0, 3, 1, 2))
        return torch.cat(branches, dim=-1)


class EncoderBlock(nn.Module):
    def __init__(self, config, adjacency, laplacian):
        super().__init__()
        self.context_graph = SpectralConv(config.width, config.cheb_order, laplacian)
        self.context_attention = AxisAttention(config, adjacency, config.context_span, residual_norm=True)
        self.attention = AxisAttention(config, adjacency)
        self.graph = SpectralConv(config.width, config.cheb_order, laplacian)
        self.temporal = MultiScaleTCN(config.width, config.kernels)
        self.ffn = nn.Sequential(nn.Linear(config.width + self.temporal.output_width, config.width * 2), nn.GELU(),
                                 nn.Dropout(config.dropout), nn.Linear(config.width * 2, config.width))
        self.norm = nn.LayerNorm(config.width)

    def forward(self, hidden, mask, previous=None):
        refined, logits = self.context_attention(self.context_graph(hidden), mask, previous, residual=hidden)
        refined = torch.where(mask[..., None], hidden, refined)
        fused, _ = self.attention(refined, mask)
        output = self.norm(hidden + self.ffn(torch.cat([self.graph(fused), self.temporal(fused)], -1)))
        return output, logits


class SharedEncoder(nn.Module):
    def __init__(self, config, adjacency):
        super().__init__()
        self.value = nn.Linear(1, config.width)
        self.missing = nn.Parameter(torch.empty(config.width))
        nn.init.normal_(self.missing, std=0.02)
        self.time = nn.Embedding(config.steps, config.width)
        self.zero_fill_ablation = config.variant == "zero_fill"
        laplacian = scaled_laplacian(adjacency).to_sparse().coalesce()
        self.blocks = nn.ModuleList([EncoderBlock(config, adjacency, laplacian) for _ in range(config.depth)])

    def forward(self, values, mask, times):
        safe = torch.where(mask.bool(), values, torch.zeros_like(values))
        observed = self.value(safe[..., None])
        missing = self.value(torch.zeros_like(values)[..., None]) if self.zero_fill_ablation else self.missing
        hidden = torch.where(mask[..., None].bool(), observed, missing) + self.time(times)[:, :, None, :]
        previous = None
        for block in self.blocks:
            hidden, previous = block(hidden, mask, previous)
        return hidden


class GatedDecoder(nn.Module):
    def __init__(self, config, adjacency):
        super().__init__()
        self.attention = AxisAttention(config, adjacency)
        self.observed_readout = nn.Linear(config.width, config.width)
        self.uncertain_readout = nn.Linear(config.width, config.width)
        self.gamma = nn.Parameter(torch.tensor([0.0, 1.0, -0.1]))
        self.output = nn.Linear(config.width, 1)

    def forward(self, hidden, mask):
        observed, _ = self.attention(hidden, mask)
        uncertain, _ = self.attention(hidden, ~mask.bool())
        xi = torch.linalg.vector_norm(hidden, dim=-1) / math.sqrt(hidden.shape[-1])
        gate = torch.sigmoid(self.gamma[0] + self.gamma[1] * mask + self.gamma[2] * xi)[..., None]
        fused = gate * self.observed_readout(observed) + (1 - gate) * self.uncertain_readout(uncertain)
        return self.output(fused).squeeze(-1)


class UDCGL(nn.Module):
    def __init__(self, config: ModelConfig, adjacency):
        super().__init__()
        if config.depth < 1 or config.width < 1 or config.steps < 1:
            raise ValueError("Positive model dimensions required")
        if config.variant not in {"full", "zero_fill", "single_head", "static_average"}:
            raise ValueError(config.variant)
        if not 0 <= config.single_head < 4:
            raise ValueError("Single-head index must lie in 0..3")
        if config.backbone != "paper":
            raise ValueError("Only the paper UDCGL architecture is supported")
        self.config = config
        self.head_indices = [config.single_head] if config.variant == "single_head" else list(range(4))
        self.encoder = SharedEncoder(config, adjacency)
        self.decoders = nn.ModuleList([GatedDecoder(config, adjacency) for _ in self.head_indices])

    def forward(self, values, mask, times):
        hidden = self.encoder(values, mask.bool(), times)
        return torch.stack([head(hidden, mask.bool()) for head in self.decoders], dim=0)


def head_objectives(model, predictions, target, fitting_mask, regularization=1e-4):
    if not bool(fitting_mask.any()):
        raise ValueError("Training window batch contains no fitting entry")
    losses = []
    for k, (index, decoder) in enumerate(zip(model.head_indices, model.decoders)):
        residual = (predictions[k] - target)[fitting_mask.bool()]
        fidelity = residual.abs() if index < 2 else residual.square()
        penalties = [p.abs().sum() if index in {0, 2} else p.square().sum() for p in decoder.parameters()]
        losses.append(fidelity.sum() + regularization * torch.stack(penalties).sum())
    return torch.stack(losses)


class Coordinator:
    def __init__(self, heads=4, delta=0.03, static=False):
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta, self.static = float(delta), bool(static)
        self.cumulative = torch.zeros(heads, dtype=torch.float64)

    def weights(self):
        if self.static:
            return torch.ones_like(self.cumulative) / len(self.cumulative)
        return torch.softmax(-self.delta * self.cumulative, dim=0)

    def observe(self, losses):
        losses = torch.as_tensor(losses, dtype=torch.float64).detach().cpu()
        if losses.shape != self.cumulative.shape or not bool(torch.isfinite(losses).all()) or (losses < 0).any():
            raise ValueError("Validation losses must be finite, nonnegative, one per head")
        self.cumulative += losses
        if not bool(torch.isfinite(self.cumulative).all()):
            raise FloatingPointError("Cumulative validation loss overflowed")
