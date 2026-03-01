"""Swin Transformer-based watermark decoder.

Decodes a 48-bit binary message from a (possibly cropped / attacked) image.

Architecture overview (inspired by RoWSFormer):
  1. Patch embedding       — 4×4 non-overlapping patches → C-dim tokens
  2. Swin Transformer stages — shifted-window self-attention at multiple
     resolutions (patch merging between stages)
  3. Global average pooling — collapse spatial dims into a single vector
  4. Classification head   — FC layers → 48 logits (one per message bit)

The decoder accepts **arbitrary spatial sizes** (multiples of the window
size after padding) so it works on both full and cropped images.

References:
  - Swin Transformer: Liu et al., ICCV 2021
  - RoWSFormer: Fang et al., 2024 (robust watermark decoder with shifted
    window attention)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_to_window(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, int, int]:
    """Pad H, W so both are divisible by *window_size*. Returns padded tensor + original H, W."""
    _, _, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, H, W


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition (B, H, W, C) into windows of shape (num_windows*B, ws, ws, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def _window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse :func:`_window_partition`."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ---------------------------------------------------------------------------
# Core Swin building blocks
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA / SW-MSA).

    Relative position bias is added to each attention head.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table — (2*ws-1)*(2*ws-1) entries per head
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute pairwise relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, ws, ws)
        coords_flat = torch.flatten(coords, 1)  # (2, ws*ws)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer(
            "relative_position_index",
            relative_coords.sum(-1),  # (ws*ws, ws*ws)
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (num_windows*B, ws*ws, C)
        mask : optional attention mask for shifted windows
        """
        BW, N, C = x.shape
        qkv = self.qkv(x).reshape(BW, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, BW, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_size ** 2, self.window_size ** 2, -1)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, heads, N, N)
        attn = attn + bias

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(BW // num_win, num_win, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(BW, N, C)
        return self.proj(out)


class SwinTransformerBlock(nn.Module):
    """Single Swin Transformer block: W-MSA or SW-MSA → MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, H*W, C) — flattened spatial tokens.
        H, W : spatial resolution (after patch embedding / merging).
        """
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Pad spatial dims to be divisible by window_size
        ws = self.window_size
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r

        # Clamp shift_size if window is larger than padded resolution
        shift = self.shift_size if min(Hp, Wp) > ws else 0

        # Shifted-window: cyclic shift
        if shift > 0:
            shifted = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
            attn_mask = self._make_mask(Hp, Wp, x.device)
        else:
            shifted = x
            attn_mask = None

        # Partition into windows
        windows = _window_partition(shifted, ws)  # (nW*B, ws, ws, C)
        windows = windows.view(-1, ws * ws, C)

        # W-MSA / SW-MSA
        attn_out = self.attn(windows, mask=attn_mask)

        # Reverse windows
        attn_out = attn_out.view(-1, ws, ws, C)
        shifted = _window_reverse(attn_out, ws, Hp, Wp)

        # Reverse cyclic shift
        if shift > 0:
            x = torch.roll(shifted, shifts=(shift, shift), dims=(1, 2))
        else:
            x = shifted

        # Remove padding
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    def _make_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create an attention mask for SW-MSA (prevents cross-region attention)."""
        ws = self.window_size
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -ws), slice(-ws, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -ws), slice(-ws, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = _window_partition(img_mask, ws)  # (nW, ws, ws, 1)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask


class PatchMerging(nn.Module):
    """Downsample spatial resolution by 2× and double channel dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, int, int]:
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        # Pad if H or W is odd (follows official Swin implementation)
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        # Merge 2×2 patches
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        new_H = (H + pad_h) // 2
        new_W = (W + pad_w) // 2
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, new_H, new_W


class SwinStage(nn.Module):
    """A stage of *depth* Swin Transformer blocks + optional downsampling."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        downsample: bool = True,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, int, int]:
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Split image into non-overlapping 4×4 patches and project to *embed_dim*."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 96, patch_size: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)  # (B, C, H/ps, W/ps)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        return x, H, W


# ---------------------------------------------------------------------------
# Full decoder
# ---------------------------------------------------------------------------

class SwinWatermarkDecoder(nn.Module):
    """Swin Transformer decoder for watermark bit extraction.

    Parameters
    ----------
    message_length : int
        Number of payload bits to decode (default: 48).
    embed_dim : int
        Initial embedding dimension after patch embedding (default: 96).
    depths : tuple[int, ...]
        Number of Swin blocks per stage (default: (2, 2, 6, 2)).
    num_heads : tuple[int, ...]
        Attention heads per stage (default: (3, 6, 12, 24)).
    window_size : int
        Window size for W-MSA / SW-MSA (default: 7).
    patch_size : int
        Non-overlapping patch size (default: 4).
    mlp_ratio : float
        Hidden-dim expansion in MLP (default: 4.0).
    """

    def __init__(
        self,
        message_length: int = 48,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        patch_size: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        assert len(depths) == len(num_heads)

        self.window_size = window_size
        self.patch_size = patch_size
        self.num_stages = len(depths)

        # ---- Patch embedding ----------------------------------------------------
        self.patch_embed = PatchEmbedding(
            in_channels=3, embed_dim=embed_dim, patch_size=patch_size,
        )

        # ---- Swin stages --------------------------------------------------------
        self.stages = nn.ModuleList()
        dim = embed_dim
        for i in range(self.num_stages):
            stage = SwinStage(
                dim=dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                downsample=(i < self.num_stages - 1),  # no downsample on last stage
                mlp_ratio=mlp_ratio,
            )
            self.stages.append(stage)
            if i < self.num_stages - 1:
                dim *= 2  # PatchMerging doubles channels

        self.norm = nn.LayerNorm(dim)

        # ---- Message head --------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, message_length),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Extract watermark bit logits from *image*.

        Parameters
        ----------
        image : Tensor (B, 3, H, W) — possibly cropped / attacked image in [0, 1].
            H and W can be arbitrary; the input is padded internally so that
            patch / window sizes divide evenly.

        Returns
        -------
        Tensor (B, message_length) — raw logits (apply sigmoid for probabilities).
        """
        # Pad so spatial dims are divisible by patch_size * window_size
        effective_ws = self.patch_size * self.window_size
        image, orig_H, orig_W = _pad_to_window(image, effective_ws)

        # Patch embedding
        x, H, W = self.patch_embed(image)  # (B, H'*W', C)

        # Swin stages
        for stage in self.stages:
            x, H, W = stage(x, H, W)

        # Global average pooling → message head
        x = self.norm(x)             # (B, H'*W', C_final)
        x = x.mean(dim=1)           # (B, C_final)
        logits = self.head(x)       # (B, message_length)

        return logits
