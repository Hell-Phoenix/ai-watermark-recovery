"""Attention Decoding Network (ADN).

Identifies surviving watermark region-of-interest (ROI) patches in
partially cropped or occluded images using spatial self-attention,
then feeds the attended features into the existing SwinWatermarkDecoder
for bit extraction.

Architecture
------------
1. **Shallow feature extractor** — lightweight CNN produces a feature map
   ``(B, C, H', W')`` from the input image ``(B, 3, H, W)``.
2. **Spatial self-attention** — multi-head self-attention over the spatial
   tokens identifies which patches still contain watermark signal.
   Outputs a normalised attention mask ``(B, 1, H', W')`` in [0, 1].
3. **Feature re-weighting** — element-wise multiply masks the features
   so that destroyed regions are suppressed before pooling / decoding.
4. **Image-domain masking** — the attention mask is upsampled back to
   the original resolution and applied to the image, producing an
   *attended image* that can be passed directly to the
   ``SwinWatermarkDecoder`` for message extraction.

The module is designed to be inserted **before** the Swin decoder in
the pipeline::

    adn     = AttentionDecodingNetwork(...)
    decoder = SwinWatermarkDecoder(...)

    attended_image, attn_mask = adn(cropped_image)
    logits = decoder(attended_image)

References:
  - "Finding Robust Watermark Regions with Attention", Li et al.
  - "ADN: Attention Decoding Network for Watermark Recovery", concept.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU helper."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel_size,
                stride=stride, padding=kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ResidualBlock(nn.Module):
    """Pre-activation residual block (BN → ReLU → Conv × 2)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Spatial multi-head self-attention
# ---------------------------------------------------------------------------

class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention over 2-D spatial tokens.

    Treats each spatial position ``(h, w)`` in the input feature map as a
    token of dimension ``dim`` and computes standard scaled dot-product
    attention across all ``H' × W'`` tokens.

    Learnable 2-D positional embeddings are added so the network can
    reason about absolute spatial location (important for cropping).

    Parameters
    ----------
    dim : int
        Token / channel dimension.
    num_heads : int
        Number of attention heads (default 4).
    max_size : int
        Maximum spatial size for the positional embedding grid (default 64).
    qkv_bias : bool
        Whether to include bias in QKV projection (default True).
    attn_drop : float
        Dropout on attention weights (default 0.0).
    proj_drop : float
        Dropout after output projection (default 0.0).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        max_size: int = 64,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learnable 2-D positional embedding (row + column, interpolated)
        self.pos_embed_h = nn.Parameter(torch.zeros(1, max_size, dim))
        self.pos_embed_w = nn.Parameter(torch.zeros(1, max_size, dim))
        nn.init.trunc_normal_(self.pos_embed_h, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_w, std=0.02)

        self.norm = nn.LayerNorm(dim)

    def _interpolate_pos(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Interpolate positional embeddings to the target (H, W)."""
        # Row embedding: (1, max_size, dim) → (1, H, dim)
        pe_h = F.interpolate(
            self.pos_embed_h.permute(0, 2, 1),  # (1, dim, max_size)
            size=H,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)  # (1, H, dim)

        # Column embedding: (1, max_size, dim) → (1, W, dim)
        pe_w = F.interpolate(
            self.pos_embed_w.permute(0, 2, 1),
            size=W,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)  # (1, W, dim)

        # Broadcast-add to (1, H, W, dim) → (1, H*W, dim)
        pos = pe_h.unsqueeze(2) + pe_w.unsqueeze(1)  # (1, H, W, dim)
        return pos.reshape(1, H * W, -1)

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Apply spatial self-attention.

        Parameters
        ----------
        x : (B, N, C) where N = H × W spatial tokens.
        H, W : spatial dimensions.

        Returns
        -------
        (B, N, C) — attended tokens.
        """
        B, N, C = x.shape

        # Add positional embeddings
        pos = self._interpolate_pos(H, W, x.device)  # (1, N, C)
        x = x + pos

        x = self.norm(x)

        # QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


# ---------------------------------------------------------------------------
# Attention mask predictor
# ---------------------------------------------------------------------------

class AttentionMaskPredictor(nn.Module):
    """Predict a per-patch survival mask from feature tokens.

    Takes the attended features ``(B, N, C)`` and predicts a scalar
    confidence for each patch indicating whether it contains intact
    watermark signal.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict patch-level mask.

        Parameters
        ----------
        x : (B, N, C)

        Returns
        -------
        (B, N, 1) — mask values in [0, 1] via sigmoid.
        """
        return torch.sigmoid(self.mlp(x))


# ---------------------------------------------------------------------------
# Main module: Attention Decoding Network
# ---------------------------------------------------------------------------

class AttentionDecodingNetwork(nn.Module):
    """Spatial-attention module that localises surviving watermark ROIs.

    Pipeline::

        image (B,3,H,W)
          → shallow CNN feature extractor → (B, feat_dim, H', W')
          → reshape to (B, H'*W', feat_dim) tokens
          → K layers of spatial self-attention
          → attention mask predictor → (B, 1, H', W') mask
          → upsample mask to (B, 1, H, W)
          → attended_image = image ⊙ mask  (soft gating)

    The attended image can then be fed to ``SwinWatermarkDecoder``
    for message extraction.

    Parameters
    ----------
    feat_dim : int
        Feature dimension for the attention layers (default 128).
    num_heads : int
        Number of attention heads (default 4).
    num_layers : int
        Number of stacked self-attention layers (default 2).
    num_res_blocks : int
        Number of residual blocks in the feature extractor (default 2).
    stride : int
        Spatial downsampling factor in the feature extractor (default 4).
        Feature map size = input_size / stride.
    """

    def __init__(
        self,
        feat_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_res_blocks: int = 2,
        stride: int = 4,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.stride = stride

        # ---- Shallow CNN feature extractor -----------------------------------
        layers: list[nn.Module] = []

        # Initial conv with stride (3 → feat_dim/2)
        if stride >= 4:
            # Two strided convs: stride 2 each → total stride 4
            layers.append(_ConvBNReLU(3, feat_dim // 2, kernel_size=3, stride=2))
            layers.append(_ConvBNReLU(feat_dim // 2, feat_dim, kernel_size=3, stride=2))
        elif stride >= 2:
            layers.append(_ConvBNReLU(3, feat_dim, kernel_size=3, stride=2))
        else:
            layers.append(_ConvBNReLU(3, feat_dim, kernel_size=3, stride=1))

        # Residual refinement
        for _ in range(num_res_blocks):
            layers.append(_ResidualBlock(feat_dim))

        self.feature_extractor = nn.Sequential(*layers)

        # ---- Self-attention layers (Transformer encoder) ---------------------
        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(
                SpatialSelfAttention(
                    dim=feat_dim,
                    num_heads=num_heads,
                    max_size=64,
                )
            )
            self.ffn_layers.append(nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, feat_dim * 4),
                nn.GELU(),
                nn.Linear(feat_dim * 4, feat_dim),
            ))

        # ---- Mask predictor --------------------------------------------------
        self.mask_predictor = AttentionMaskPredictor(feat_dim)

        # ---- Output projection (for feature-level output) --------------------
        self.out_norm = nn.LayerNorm(feat_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Identify surviving watermark ROIs and produce an attended image.

        Parameters
        ----------
        image : (B, 3, H, W)
            Possibly cropped / attacked image in [0, 1].

        Returns
        -------
        attended_image : (B, 3, H, W)
            Input image soft-gated by the spatial attention mask.
            Destroyed regions are suppressed toward zero; surviving
            watermark patches are preserved.  Can be passed directly
            to ``SwinWatermarkDecoder.forward(attended_image)``.

        attention_mask : (B, 1, H, W)
            Per-pixel attention mask in [0, 1] at the original
            spatial resolution.  Values near 1 indicate patches the
            network believes contain intact watermark signal.
        """
        B, C, H, W = image.shape

        # --- 1. Extract spatial features ---
        feat = self.feature_extractor(image)  # (B, feat_dim, H', W')
        _, _, Hf, Wf = feat.shape

        # --- 2. Reshape to token sequence ---
        tokens = feat.flatten(2).transpose(1, 2)  # (B, H'*W', feat_dim)

        # --- 3. Self-attention layers (with residual connections) ---
        for attn_layer, ffn_layer in zip(self.attn_layers, self.ffn_layers):
            # Multi-head self-attention + residual
            tokens = tokens + attn_layer(tokens, Hf, Wf)
            # Feed-forward network + residual
            tokens = tokens + ffn_layer(tokens)

        # --- 4. Predict per-patch attention mask ---
        mask_tokens = self.mask_predictor(tokens)  # (B, H'*W', 1)
        mask_2d = mask_tokens.transpose(1, 2).reshape(B, 1, Hf, Wf)  # (B, 1, H', W')

        # --- 5. Upsample mask to original resolution ---
        attention_mask = F.interpolate(
            mask_2d,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (B, 1, H, W)

        # --- 6. Apply soft attention gating to input image ---
        attended_image = image * attention_mask  # (B, 3, H, W)

        return attended_image, attention_mask

    def get_attended_features(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return attended features in addition to the image-level outputs.

        Useful when downstream modules need the feature representation
        directly (e.g., for multi-task heads or auxiliary losses).

        Parameters
        ----------
        image : (B, 3, H, W)

        Returns
        -------
        attended_image : (B, 3, H, W)
        attention_mask : (B, 1, H, W)
        attended_features : (B, feat_dim, H', W')
            Feature map weighted by the attention mask.
        """
        B, C, H, W = image.shape

        feat = self.feature_extractor(image)
        _, _, Hf, Wf = feat.shape

        tokens = feat.flatten(2).transpose(1, 2)

        for attn_layer, ffn_layer in zip(self.attn_layers, self.ffn_layers):
            tokens = tokens + attn_layer(tokens, Hf, Wf)
            tokens = tokens + ffn_layer(tokens)

        # Outputs
        mask_tokens = self.mask_predictor(tokens)
        mask_2d = mask_tokens.transpose(1, 2).reshape(B, 1, Hf, Wf)

        # Attended features at feature resolution
        out_tokens = self.out_norm(tokens)
        out_features = out_tokens.transpose(1, 2).reshape(B, self.feat_dim, Hf, Wf)
        attended_features = out_features * mask_2d  # (B, feat_dim, H', W')

        # Full-res mask + attended image
        attention_mask = F.interpolate(mask_2d, size=(H, W), mode="bilinear", align_corners=False)
        attended_image = image * attention_mask

        return attended_image, attention_mask, attended_features


# ---------------------------------------------------------------------------
# Loss for training the ADN
# ---------------------------------------------------------------------------

class ADNLoss(nn.Module):
    """Training loss for the Attention Decoding Network.

    Combines:
      1. **Mask supervision** — BCE between predicted mask and ground-truth
         crop mask (if available).
      2. **Message accuracy** — BCE between decoder logits on attended image
         and the true message bits.
      3. **Sparsity regularisation** — encourages the mask to be selective
         (penalises mean mask value to avoid trivial all-ones solution).

    Parameters
    ----------
    lambda_mask : float
        Weight for mask supervision (default 1.0).
    lambda_msg : float
        Weight for message accuracy (default 1.0).
    lambda_sparse : float
        Sparsity penalty weight (default 0.1).
    """

    def __init__(
        self,
        lambda_mask: float = 1.0,
        lambda_msg: float = 1.0,
        lambda_sparse: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_mask = lambda_mask
        self.lambda_msg = lambda_msg
        self.lambda_sparse = lambda_sparse

    def forward(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        message: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute composite ADN loss.

        Parameters
        ----------
        pred_mask : (B, 1, H, W) — predicted attention mask.
        gt_mask : (B, 1, H, W) or None — ground-truth crop mask (1 = intact).
        logits : (B, N) or None — decoder logits on attended image.
        message : (B, N) or None — ground-truth watermark bits.

        Returns
        -------
        dict with ``loss``, and optionally ``mask_bce``, ``msg_bce``,
        ``sparsity``, ``mask_iou``.
        """
        total = torch.tensor(0.0, device=pred_mask.device)
        result: dict[str, torch.Tensor] = {}

        # Mask supervision
        if gt_mask is not None:
            mask_bce = F.binary_cross_entropy(
                pred_mask, gt_mask.float(),
            )
            total = total + self.lambda_mask * mask_bce
            result["mask_bce"] = mask_bce.detach()

            # IoU metric
            with torch.no_grad():
                intersection = (pred_mask * gt_mask).sum()
                union = pred_mask.sum() + gt_mask.sum() - intersection
                iou = intersection / (union + 1e-6)
                result["mask_iou"] = iou

        # Message accuracy
        if logits is not None and message is not None:
            msg_bce = F.binary_cross_entropy_with_logits(
                logits, message.float(),
            )
            total = total + self.lambda_msg * msg_bce
            result["msg_bce"] = msg_bce.detach()

            with torch.no_grad():
                predicted = (torch.sigmoid(logits) > 0.5).float()
                result["bit_acc"] = (predicted == message.float()).float().mean()

        # Sparsity regularisation
        sparsity = pred_mask.mean()
        total = total + self.lambda_sparse * sparsity
        result["sparsity"] = sparsity.detach()

        result["loss"] = total
        return result
