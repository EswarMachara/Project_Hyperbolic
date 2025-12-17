"""
model.py - HANS-Net (Hyperbolic Attention Network for Segmentation)

Complete implementation of HANS-Net for liver/tumor CT segmentation.

Components:
    - WaveletDecomposition / WaveletReconstruction: 2D DWT for multi-frequency features
    - SynapticPlasticity / PlasticConvBlock: Adaptive convolutions with Hebbian learning
    - HyperbolicConvBlock: Convolutions in Poincaré ball geometry
    - TemporalAttention: Cross-frame attention for temporal context (T=3 slices)
    - INRBranch: Implicit Neural Representation for boundary refinement
    - HANSNet: Complete U-Net style architecture

Input:  [B, T=3, 1, H, W] - 3 consecutive CT slices
Output: [B, 1, H, W]      - Segmentation logits for center slice
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Wavelet Decomposition
# =============================================================================

class WaveletDecomposition(nn.Module):
    """
    2D Discrete Wavelet Transform using Haar wavelets.
    
    Decomposes each input channel into 4 subbands (LL, LH, HL, HH),
    effectively converting [B, C, H, W] -> [B, C*4, H/2, W/2].
    This captures multi-frequency information useful for segmentation.
    """
    
    def __init__(self):
        super().__init__()
        # Haar wavelet filters (normalized)
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2.0   # LL: avg
        lh = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 2.0  # LH: horizontal
        hl = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 2.0  # HL: vertical
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2.0  # HH: diagonal
        
        # Stack filters: [4, 1, 2, 2]
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('filters', filters)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Wavelet coefficients [B, C*4, H//2, W//2]
        """
        B, C, H, W = x.shape
        
        # Apply filters to each channel independently
        x_reshape = x.view(B * C, 1, H, W)
        
        # Convolve with 4 wavelet filters, stride=2 for downsampling
        coeffs = F.conv2d(x_reshape, self.filters, stride=2, padding=0)
        
        # Reshape back: [B, C*4, H//2, W//2]
        _, _, H_out, W_out = coeffs.shape
        coeffs = coeffs.view(B, C * 4, H_out, W_out)
        
        return coeffs


class WaveletReconstruction(nn.Module):
    """
    Inverse 2D Discrete Wavelet Transform.
    Converts [B, C*4, H, W] -> [B, C, H*2, W*2]
    """
    
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2.0
        lh = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 2.0
        hl = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 2.0
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2.0
        
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(0)  # [1, 4, 2, 2]
        self.register_buffer('filters', filters)
    
    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coeffs: Wavelet coefficients [B, C*4, H, W] where C*4 is divisible by 4
        Returns:
            Reconstructed tensor [B, C, H*2, W*2]
        """
        B, C4, H, W = coeffs.shape
        C = C4 // 4
        
        coeffs = coeffs.view(B * C, 4, H, W)
        x = F.conv_transpose2d(coeffs, self.filters, stride=2, padding=0)
        x = x.view(B, C, H * 2, W * 2)
        
        return x


# =============================================================================
# 2. Synaptic Plasticity & PlasticConvBlock
# =============================================================================

class SynapticPlasticity(nn.Module):
    """
    Learnable weight modulation inspired by Hebbian learning.
    
    Implements activity-dependent gain modulation where the effective
    weights are scaled based on learned per-channel parameters and
    input statistics.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(channels))
        self.threshold = nn.Parameter(torch.zeros(channels))
        self.plasticity_rate = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            Modulated features [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Compute per-channel activation statistics
        channel_mean = x.mean(dim=(2, 3))
        
        # Hebbian modulation
        modulation = torch.sigmoid(
            self.plasticity_rate * (channel_mean - self.threshold.view(1, -1))
        )
        
        # Apply gain and modulation
        effective_scale = self.gain.view(1, -1, 1, 1) * (1 + modulation.view(B, C, 1, 1))
        
        return x * effective_scale


class PlasticConvBlock(nn.Module):
    """
    Convolutional block with synaptic plasticity, batch normalization, and GELU activation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_plasticity: bool = True,
        use_residual: bool = False,
        dropout_p: float = 0.0
    ):
        super().__init__()
        self.use_plasticity = use_plasticity
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if use_plasticity:
            self.plasticity = SynapticPlasticity(out_channels)
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0.0 else nn.Identity()
        
        if use_residual and (in_channels != out_channels or stride != 1):
            self.residual_proj = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_plasticity:
            out = self.plasticity(out)
        
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            out = out + identity
        
        out = self.dropout(out)
        out = self.act(out)
        
        return out


# =============================================================================
# 3. Hyperbolic Geometry Helpers (Poincaré Ball Model)
# =============================================================================

def exp_map_zero(v: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Exponential map from the origin in the Poincaré ball.
    Maps a Euclidean vector v to a point on the Poincaré ball with curvature -c.
    """
    sqrt_c = torch.sqrt(c)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    
    # Clamp the argument to tanh to avoid saturation
    tanh_arg = (sqrt_c * v_norm).clamp(max=15.0)
    
    return torch.tanh(tanh_arg) * v / (sqrt_c * v_norm + eps)


def log_map_zero(y: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Logarithmic map to the origin in the Poincaré ball.
    Maps a point y on the Poincaré ball back to Euclidean tangent space at origin.
    """
    sqrt_c = torch.sqrt(c)
    y_norm = y.norm(dim=-1, keepdim=True).clamp(min=eps)
    
    # Clamp to ensure arctanh input is strictly in (-1, 1)
    y_norm_scaled = (sqrt_c * y_norm).clamp(min=eps, max=1.0 - eps)
    
    return torch.arctanh(y_norm_scaled) * y / (sqrt_c * y_norm + eps)


def project_to_ball(x: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Project points to be inside the Poincaré ball.
    Ensures ||x|| < 1/sqrt(c) by scaling down points that are outside.
    """
    max_norm = (1.0 - eps) / torch.sqrt(c)
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    scale = torch.clamp(max_norm / x_norm, max=1.0)
    return x * scale


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Möbius addition in the Poincaré ball (hyperbolic analog of vector addition).
    """
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    
    num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
    denom = (1 + 2 * c * xy + c * c * x_sq * y_sq).clamp(min=eps)
    
    result = num / denom
    return project_to_ball(result, c, eps)


class HyperbolicConvBlock(nn.Module):
    """
    Convolution in hyperbolic (Poincaré ball) space.
    
    Hyperbolic geometry is well-suited for capturing hierarchical structures
    in medical images (organ → region → tissue → boundary).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        curvature: float = 1.0,
        learnable_curvature: bool = True
    ):
        super().__init__()
        
        if learnable_curvature:
            init_val = math.log(math.exp(curvature) - 1)
            self.curvature = nn.Parameter(torch.tensor(init_val))
        else:
            self.register_buffer('curvature', torch.tensor(curvature))
        
        self.input_norm = nn.GroupNorm(min(8, in_channels), in_channels)
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        self.output_norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act = nn.GELU()
        
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Ensure curvature is positive and bounded
        c = F.softplus(self.curvature).clamp(min=0.1, max=10.0)
        
        # Normalize input
        x = self.input_norm(x)
        
        # Scale features to be small (important for hyperbolic stability)
        x_scaled = x * torch.abs(self.scale)
        
        # Reshape for hyperbolic operations: [B, H, W, C]
        x_bhwc = x_scaled.permute(0, 2, 3, 1).contiguous()
        
        # Map to Poincaré ball
        x_hyp = exp_map_zero(x_bhwc, c)
        x_hyp = project_to_ball(x_hyp, c)
        
        # Map back to tangent space for convolution
        x_tangent = log_map_zero(x_hyp, c)
        
        # Back to conv format: [B, C, H, W]
        x_tangent = x_tangent.permute(0, 3, 1, 2).contiguous()
        
        # Apply convolution in tangent space
        out = self.conv(x_tangent)
        out = out + self.bias.view(1, -1, 1, 1)
        
        # Normalize and activate
        out = self.output_norm(out)
        out = self.act(out)
        
        return out


# =============================================================================
# 4. Temporal Attention
# =============================================================================

class TemporalAttention(nn.Module):
    """
    Cross-frame attention for aggregating temporal context from T consecutive slices.
    
    Uses the center frame as the query and all frames as keys/values.
    
    Input: [B, T, C, H, W] - T frames of features
    Output: [B, C, H, W] - Temporally-attended center frame features
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
        
        # Learnable temporal position embeddings for T=3
        self.temporal_pos = nn.Parameter(torch.randn(1, 3, embed_dim, 1, 1) * 0.02)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, C, H, W]
        Returns:
            Output tensor [B, C, H, W] (center frame with temporal context)
        """
        B, T, C, H, W = x.shape
        
        # Add temporal position embeddings
        x = x + self.temporal_pos[:, :T]
        
        # Extract center frame for query
        center_idx = T // 2
        q_input = x[:, center_idx]  # [B, C, H, W]
        
        # Compute query from center frame
        q = self.q_proj(q_input)
        
        # Compute keys and values from all frames
        x_flat = x.view(B * T, C, H, W)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)
        
        # Reshape for attention computation
        q = q.view(B, self.num_heads, self.head_dim, H * W)
        k = k.view(B, T, self.num_heads, self.head_dim, H * W)
        v = v.view(B, T, self.num_heads, self.head_dim, H * W)
        
        # Per-position temporal attention
        q = q.permute(0, 1, 3, 2)  # [B, num_heads, H*W, head_dim]
        k = k.permute(0, 2, 3, 4, 1)  # [B, num_heads, head_dim, H*W, T]
        v = v.permute(0, 2, 3, 4, 1)  # [B, num_heads, head_dim, H*W, T]
        
        attn = torch.einsum('bnsd,bndst->bnst', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('bnst,bndst->bnsd', attn, v)
        
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(B, C, H, W)
        
        out = self.out_proj(out)
        out = out + q_input
        
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)
        
        return out


# =============================================================================
# 5. INR Branch (Implicit Neural Representation)
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Fourier feature positional encoding for 2D coordinates.
    Maps coordinates from [-1, 1] to a higher-dimensional space using sinusoidal functions.
    """
    
    def __init__(self, num_frequencies: int = 10, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('freq_bands', freq_bands)
        
        self.out_dim = 2 * num_frequencies * 2
        if include_input:
            self.out_dim += 2
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Normalized coordinates [*, 2] in range [-1, 1]
        Returns:
            Encoded coordinates [*, out_dim]
        """
        scaled = coords.unsqueeze(-1) * self.freq_bands * math.pi
        encoded = torch.stack([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        encoded = encoded.view(*coords.shape[:-1], -1)
        
        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)
        
        return encoded


def make_coord_grid(H: int, W: int, device: torch.device = None) -> torch.Tensor:
    """Create a grid of normalized 2D coordinates in [-1, 1]."""
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1)
    return coords


class INRBranch(nn.Module):
    """
    Implicit Neural Representation branch for boundary refinement.
    
    Combines high-resolution coordinate encoding with image features
    to produce continuous, resolution-independent refinement signals.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_frequencies: int = 10,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(num_frequencies, include_input=True)
        coord_dim = self.pos_encoder.out_dim
        
        self.feature_proj = nn.Conv2d(feature_dim, hidden_dim // 2, 1)
        
        input_dim = coord_dim + hidden_dim // 2
        
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else 1
            
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: Image features [B, C, H, W]
            coords: Optional pre-computed coordinates [H, W, 2]
        Returns:
            Refinement logits [B, 1, H, W]
        """
        B, C, H, W = features.shape
        device = features.device
        
        if coords is None:
            coords = make_coord_grid(H, W, device)
        
        coord_enc = self.pos_encoder(coords)
        coord_enc = coord_enc.unsqueeze(0).expand(B, -1, -1, -1)
        
        feat_proj = self.feature_proj(features)
        feat_proj = feat_proj.permute(0, 2, 3, 1)
        
        combined = torch.cat([coord_enc, feat_proj], dim=-1)
        out = self.mlp(combined)
        out = out.permute(0, 3, 1, 2)
        
        return out


# =============================================================================
# 6. HANSNet - Complete U-Net Architecture
# =============================================================================

class HANSNet(nn.Module):
    """
    Hyperbolic Attention Network for Segmentation (HANS-Net).
    
    A U-Net style architecture for liver/tumor CT segmentation that combines:
    - Wavelet decomposition for multi-frequency feature extraction
    - Synaptic plasticity for adaptive feature learning
    - Temporal attention to fuse information from adjacent CT slices
    - Hyperbolic convolutions in the bottleneck for hierarchical representations
    - INR branch for boundary refinement
    
    Input:  [B, T=3, 1, H, W] - 3 consecutive CT slices
    Output: [B, 1, H, W]      - Segmentation logits for center slice
    """
    
    def __init__(self, base_channels: int = 32, num_classes: int = 1):
        super().__init__()
        
        self.base_channels = base_channels
        self.num_classes = num_classes
        
        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 4   # 128
        c4 = base_channels * 8   # 256
        
        # =====================================================================
        # ENCODER
        # =====================================================================
        
        self.wavelet = WaveletDecomposition()
        self.enc1 = PlasticConvBlock(4, c1, use_plasticity=True)
        
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = PlasticConvBlock(c1, c2, use_plasticity=True)
        
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = PlasticConvBlock(c2, c3, use_plasticity=True)
        
        # =====================================================================
        # TEMPORAL ATTENTION
        # =====================================================================
        
        self.temporal_attn = TemporalAttention(embed_dim=c3, num_heads=4)
        
        # =====================================================================
        # BOTTLENECK
        # =====================================================================
        
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = HyperbolicConvBlock(c3, c4, curvature=1.0, learnable_curvature=True)
        
        # =====================================================================
        # DECODER
        # =====================================================================
        
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = PlasticConvBlock(c3 + c3, c3, use_plasticity=True, dropout_p=0.3)
        
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = PlasticConvBlock(c2 + c2, c2, use_plasticity=True, dropout_p=0.3)
        
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = PlasticConvBlock(c1 + c1, c1, use_plasticity=True, dropout_p=0.3)
        
        self.final_up = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)
        self.final_conv = PlasticConvBlock(c1, c1, use_plasticity=True, dropout_p=0.3)
        
        # =====================================================================
        # OUTPUT HEADS
        # =====================================================================
        
        self.seg_head = nn.Conv2d(c1, num_classes, kernel_size=1)
        
        self.inr_branch = INRBranch(
            feature_dim=c1,
            hidden_dim=128,
            num_frequencies=10,
            num_layers=3
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of HANS-Net.
        
        Args:
            x: Input tensor [B, T=3, 1, H, W] - 3 consecutive CT slices
        Returns:
            Segmentation logits [B, 1, H, W] for center slice
        """
        B, T, C, H, W = x.shape
        assert T == 3, f"Expected T=3 slices, got T={T}"
        assert C == 1, f"Expected C=1 channel, got C={C}"
        
        # =====================================================================
        # ENCODER
        # =====================================================================
        
        x_flat = x.view(B * T, C, H, W)
        
        e1 = self.wavelet(x_flat)
        e1 = self.enc1(e1)
        
        e2 = self.pool1(e1)
        e2 = self.enc2(e2)
        
        e3 = self.pool2(e2)
        e3 = self.enc3(e3)
        
        # =====================================================================
        # TEMPORAL ATTENTION
        # =====================================================================
        
        _, c3_ch, h3, w3 = e3.shape
        e3_temporal = e3.view(B, T, c3_ch, h3, w3)
        f_center = self.temporal_attn(e3_temporal)
        
        # =====================================================================
        # BOTTLENECK
        # =====================================================================
        
        bottleneck = self.pool3(f_center)
        bottleneck = self.bottleneck(bottleneck)
        
        # =====================================================================
        # EXTRACT CENTER SLICE FEATURES FOR SKIP CONNECTIONS
        # =====================================================================
        
        center_idx = T // 2
        
        _, c1_ch, h1, w1 = e1.shape
        e1_center = e1.view(B, T, c1_ch, h1, w1)[:, center_idx]
        
        _, c2_ch, h2, w2 = e2.shape
        e2_center = e2.view(B, T, c2_ch, h2, w2)[:, center_idx]
        
        e3_center = f_center
        
        # =====================================================================
        # DECODER
        # =====================================================================
        
        d3 = self.up3(bottleneck)
        d3 = torch.cat([d3, e3_center], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2_center], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1_center], dim=1)
        d1 = self.dec1(d1)
        
        dec_out = self.final_up(d1)
        dec_out = self.final_conv(dec_out)
        
        # =====================================================================
        # OUTPUT
        # =====================================================================
        
        coarse_logits = self.seg_head(dec_out)
        refine_logits = self.inr_branch(dec_out)
        final_logits = coarse_logits + refine_logits
        
        return final_logits


# =============================================================================
# 7. Loss Functions & MC-Dropout Inference Utilities
# =============================================================================

def dice_loss(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute soft Dice loss for binary segmentation.
    
    Args:
        probs: Model output after sigmoid, shape [B, 1, H, W], values in [0, 1]
        targets: Ground-truth binary mask, shape [B, 1, H, W], values in {0, 1}
        eps: Small constant for numerical stability
    
    Returns:
        Scalar tensor: mean Dice loss over the batch
    """
    assert probs.shape == targets.shape, f"Shape mismatch: {probs.shape} vs {targets.shape}"
    
    B = probs.shape[0]
    
    probs_flat = probs.view(B, -1)
    targets_flat = targets.view(B, -1)
    
    intersection = (probs_flat * targets_flat).sum(dim=1)
    sum_probs = probs_flat.sum(dim=1)
    sum_targets = targets_flat.sum(dim=1)
    
    dice_coeff = (2.0 * intersection + eps) / (sum_probs + sum_targets + eps)
    loss = 1.0 - dice_coeff.mean()
    
    return loss


def combined_loss(logits: torch.Tensor, targets: torch.Tensor, 
                  bce_weight: float = 0.5, dice_weight: float = 0.5) -> torch.Tensor:
    """
    Combined BCE + Dice loss for segmentation.
    
    Args:
        logits: Raw model output (before sigmoid), shape [B, 1, H, W]
        targets: Ground-truth binary mask, shape [B, 1, H, W]
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
    
    Returns:
        Scalar tensor: weighted sum of BCE and Dice losses
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    dice = dice_loss(probs, targets)
    
    return bce_weight * bce + dice_weight * dice


def mc_predict(model: nn.Module, x: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Monte Carlo Dropout prediction for uncertainty estimation.
    
    Args:
        model: HANSNet instance (or any model with Dropout layers)
        x: Input tensor [B, T=3, 1, H, W]
        n_samples: Number of stochastic forward passes
    
    Returns:
        mean_probs: Mean predicted probabilities [B, 1, H, W]
        var_probs: Variance of predictions (uncertainty) [B, 1, H, W]
    """
    was_training = model.training
    model.train()
    
    samples = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(x)
            probs = torch.sigmoid(logits)
            samples.append(probs)
    
    samples = torch.stack(samples, dim=0)
    mean_probs = samples.mean(dim=0)
    var_probs = samples.var(dim=0)
    
    if not was_training:
        model.eval()
    
    return mean_probs, var_probs


def mc_predict_with_samples(model: nn.Module, x: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extended MC Dropout that also returns all individual samples.
    
    Args:
        model: HANSNet instance
        x: Input tensor [B, T=3, 1, H, W]
        n_samples: Number of stochastic forward passes
    
    Returns:
        mean_probs: Mean predicted probabilities [B, 1, H, W]
        var_probs: Variance of predictions [B, 1, H, W]
        all_samples: All prediction samples [n_samples, B, 1, H, W]
    """
    was_training = model.training
    model.train()
    
    samples = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(x)
            probs = torch.sigmoid(logits)
            samples.append(probs)
    
    samples = torch.stack(samples, dim=0)
    mean_probs = samples.mean(dim=0)
    var_probs = samples.var(dim=0)
    
    if not was_training:
        model.eval()
    
    return mean_probs, var_probs, samples


# =============================================================================
# 8. Testing & Validation
# =============================================================================

def test_hansnet():
    """Test the complete HANS-Net model."""
    print("=" * 60)
    print("HANS-Net Model Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")
    
    model = HANSNet(base_channels=32, num_classes=1).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    x_dummy = torch.randn(2, 3, 1, 128, 128, device=device)
    print(f"Input shape:  {list(x_dummy.shape)} [B, T, C, H, W]")
    
    model.eval()
    with torch.no_grad():
        output = model(x_dummy)
    
    print(f"Output shape: {list(output.shape)} [B, num_classes, H, W]")
    print(f"Expected:     [2, 1, 128, 128]")
    
    assert output.shape == (2, 1, 128, 128), f"Shape mismatch! Got {output.shape}"
    assert not torch.isnan(output).any(), "NaN in output!"
    assert not torch.isinf(output).any(), "Inf in output!"
    
    print("\n✓ HANS-Net forward pass successful!")
    print("✓ No NaN/Inf values!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    test_hansnet()
