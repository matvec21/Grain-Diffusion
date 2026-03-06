# %%

import numpy as np
import torch
import torch.nn as nn

# %%

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%

def _f(t):
    start = 1e-6
    end = 0.04
    return (end + 0.5 * (start - end) * (1 + np.cos(np.pi * t / TIME_STEPS))).item()

TIME_STEPS = 250
BETA = torch.tensor([_f(t) for t in range(TIME_STEPS)], device = device)

beta_prod = torch.cumprod(1 - BETA, 0).view(TIME_STEPS, 1, 1, 1)
sqrt_beta_prod = torch.sqrt(beta_prod)
sqrt_beta_prod_step = torch.sqrt(1 - beta_prod)

# %%

def timestep_embedding(timesteps, dim, max_period = 10000):
    half = dim // 2
    freqs = torch.exp(
        -np.log(max_period) * torch.arange(half, device = timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, t_dim, dilation):
        super().__init__()

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )

        self.time_proj = nn.Linear(t_dim, channels)

        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.norm(x)
        h = self.act(h)
        h = self.conv(h)

        h = h + self.time_proj(t_emb)[:, :, None]

        return x + h

class Diffusion1D(nn.Module):
    def __init__(
        self,
        base_channels = 64,
        t_dim = 32,
        num_blocks = 4,
        std = False
    ):
        super().__init__()

        self.input_conv = nn.Conv1d(2, base_channels, 1)

        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim)
        )

        dilations = [2 ** i for i in range(num_blocks)]

        self.blocks = nn.ModuleList([
            ResidualBlock1D(base_channels, t_dim, d)
            for d in dilations
        ])

        self.output_norm = nn.GroupNorm(1, base_channels)
        self.output_act = nn.SiLU()
        self.output_conv = nn.Conv1d(base_channels, 1, 1)

        self.std = std
        self.t_dim = t_dim

    def forward(self, x, t_emb):
        """
        x: (T * B, 1, 250) float
        t_emb: (T * B, t_dim) float
        """
        t_emb = self.time_mlp(t_emb)

        pos = torch.linspace(0, 1, x.shape[-1], device = x.device)
        pos = pos.view(1, 1, -1).repeat(x.shape[0], 1, 1)
        h = self.input_conv(torch.cat((x, pos), dim = 1))

        for block in self.blocks:
            h = block(h, t_emb)

        h = self.output_norm(h)
        h = self.output_act(h)
        h = self.output_conv(h)

        if self.std:
            return nn.functional.softplus(h)
        return h

# %%

def generate(model : nn.Module, count, mean : float, std : float):
    device = next(model.parameters()).device
    xt = torch.randn(count, 1, 250, device = device)
    xt[xt < -mean / std] = -mean / std

    for t in reversed(range(TIME_STEPS)):
        t_emb = timestep_embedding(torch.tensor([t], device = device), model.t_dim).repeat(count, 1)
        noise = torch.randn(xt.shape, device = device)

        if t == 0:
            k = 0
        else:
            k = torch.sqrt(BETA[t] * (1 - beta_prod[t - 1]) / (1 - beta_prod[t]))

        if t < 15:
            k *= t / 15

        pred = model(xt, t_emb)
        xt = k * noise + (xt - BETA[t] * pred / sqrt_beta_prod_step[t]) / torch.sqrt(1 - BETA[t])
        xt[xt < -mean / std] = -mean / std
    return xt

# %%
