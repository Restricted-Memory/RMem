import numpy as np
from matplotlib import pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.math import truncated_normal_


class Downsample2D(nn.Module):
    def __init__(self, mode='nearest', scale=4):
        super().__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        n, c, h, w = x.size()
        x = F.interpolate(x,
                          size=(h // self.scale + 1, w // self.scale + 1),
                          mode=self.mode)
        return x


def generate_coord(x):
    _, _, h, w = x.size()
    device = x.device
    col = torch.arange(0, h, device=device)
    row = torch.arange(0, w, device=device)
    grid_h, grid_w = torch.meshgrid(col, row)
    return grid_h, grid_w


class PositionEmbeddingSine(nn.Module):
    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        grid_y, grid_x = generate_coord(x)

        y_embed = grid_y.unsqueeze(0).float()
        x_embed = grid_x.unsqueeze(0).float()

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32,
                             device=x.device)
        # dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature**(2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, num_pos_feats=64, H=30, W=30):
        super().__init__()
        self.H = H
        self.W = W
        self.pos_emb = nn.Parameter(
            truncated_normal_(torch.zeros(1, num_pos_feats, H, W)))

    def forward(self, x):
        bs, _, h, w = x.size()
        pos_emb = self.pos_emb
        if h != self.H or w != self.W:
            pos_emb = F.interpolate(pos_emb, size=(h, w), mode="bilinear")
        return pos_emb


def get_temporal_positional_encoding(
        max_sequence_len,
        channels,
        device,
        is_normalize=False,
        scale=2*math.pi,
        is_debug=False,
):
    position = torch.arange(max_sequence_len, device=device)
    if is_normalize:
        position = position / position[-1] * scale
    if is_debug:
        print(f"{position = }")
    position.unsqueeze_(1)
    div_term = 1.0 / (10000.0 ** (
        torch.arange(0, channels, 2, device=device).float() / channels))
    position_div_term = position * div_term

    temporal_encoding = torch.zeros(
        (max_sequence_len, 1, 1, channels), device=device)
    temporal_encoding_sin = torch.sin(position_div_term)
    temporal_encoding_cos = torch.cos(position_div_term)
    temporal_encoding[:, 0, 0, 0::2] = temporal_encoding_sin
    temporal_encoding[:, 0, 0, 1::2] = temporal_encoding_cos

    if is_debug:
        position_np = position.detach().cpu().numpy()
        position_div_term_np = position_div_term.detach().cpu().numpy()
        for i, p in enumerate(position_np[:0x10]):
            plt.plot(
                np.arange(position_div_term_np.shape[1]),
                position_div_term_np[i],
                label=f"line {p}",
            )
        plt.legend()
        plt.savefig("position_div_term.png")
        plt.close()
        temporal_encoding_sin_np = temporal_encoding_sin.detach().cpu().numpy()
        for i, p in enumerate(position_np[:0x10]):
            plt.subplot(4, 4, i+1)
            plt.plot(
                np.arange(temporal_encoding_sin_np.shape[1]),
                temporal_encoding_sin_np[i],
                label=f"line {p}",
            )
            plt.ylim(-1.05, 1.05)
            # plt.legend()
        plt.savefig("temporal_encoding_sin.png", dpi=300)
        plt.close()
        temporal_encoding_cos_np = temporal_encoding_cos.detach().cpu().numpy()
        for i, p in enumerate(position_np[:0x10]):
            plt.subplot(4, 4, i+1)
            plt.plot(
                np.arange(temporal_encoding_cos_np.shape[1]),
                temporal_encoding_cos_np[i],
                label=f"line {p}",
            )
            plt.ylim(-1.05, 1.05)
            # plt.legend()
        plt.savefig("temporal_encoding_cos.png", dpi=300)
        plt.close()

    return temporal_encoding
