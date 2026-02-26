"""
Enhanced Cross-Attention Fusion Module for YOWOFormer (V2)
Improvements over V1:
1. Learnable Positional Encoding for spatial awareness
2. Temperature scaling for attention sharpness control
3. Pre-normalization for training stability
4. Adaptive gating mechanism for smart fusion
5. Depthwise separable convolutions for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding to help model understand spatial positions
    """
    def __init__(self, max_len, embed_dim):
        super().__init__()
        # Initialize with small random values
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)

    def forward(self, x):
        """
        Args:
            x: [B, N, C] where N is sequence length
        Returns:
            x + positional encoding
        """
        B, N, C = x.shape

        if N > self.pos_embedding.shape[1]:
            # If input is larger, interpolate the positional encoding
            pos_embed = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=N,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            # Use the first N positions
            pos_embed = self.pos_embedding[:, :N, :]

        return x + pos_embed


class MultiHeadCrossAttentionV2(nn.Module):
    """
    Enhanced Multi-Head Cross-Attention with temperature scaling
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, use_temp_scaling=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Learnable temperature for each head (initialized to 1)
        if use_temp_scaling:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        else:
            self.register_buffer('temperature', torch.ones(num_heads, 1, 1))

        # Linear projections with bias for better flexibility
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization for better convergence
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, query, key, value, return_attention=False):
        B, N_q, C = query.shape
        N_kv = key.shape[1]

        # Project and reshape to multi-head format
        Q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention with temperature scaling
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores / self.temperature  # Temperature scaling

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = attn_weights @ V

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, N_q, C)
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)

        if return_attention:
            return output, attn_weights
        return output


class BidirectionalCrossAttentionBlockV2(nn.Module):
    """
    Bidirectional Cross-Attention with Pre-Normalization and Positional Encoding
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, mlp_ratio=4.0,
                 use_pos_encoding=True, max_spatial_size=784):  # 28*28 default
        super().__init__()

        # Positional encoding for spatial awareness
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_enc_2d = LearnablePositionalEncoding(max_spatial_size, embed_dim)
            self.pos_enc_3d = LearnablePositionalEncoding(max_spatial_size, embed_dim)

        # Pre-normalization (more stable than post-norm)
        self.norm1_2d_q = nn.LayerNorm(embed_dim)
        self.norm1_2d_kv = nn.LayerNorm(embed_dim)
        self.norm1_3d_q = nn.LayerNorm(embed_dim)
        self.norm1_3d_kv = nn.LayerNorm(embed_dim)
        self.norm2_2d = nn.LayerNorm(embed_dim)
        self.norm2_3d = nn.LayerNorm(embed_dim)

        # Cross-attention layers
        self.cross_attn_2d_to_3d = MultiHeadCrossAttentionV2(
            embed_dim, num_heads, dropout, use_temp_scaling=True
        )
        self.cross_attn_3d_to_2d = MultiHeadCrossAttentionV2(
            embed_dim, num_heads, dropout, use_temp_scaling=True
        )

        # MLP with SiLU activation (Swish)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp_2d = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.SiLU(),  # Better than GELU in many cases
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
        self.mlp_3d = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )

        # Learnable scaling for residual connections
        self.gamma_attn_2d = nn.Parameter(torch.ones(1) * 0.1)
        self.gamma_attn_3d = nn.Parameter(torch.ones(1) * 0.1)
        self.gamma_mlp_2d = nn.Parameter(torch.ones(1) * 0.1)
        self.gamma_mlp_3d = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, feat_2d, feat_3d):
        # Add positional encoding
        if self.use_pos_encoding:
            feat_2d = self.pos_enc_2d(feat_2d)
            feat_3d = self.pos_enc_3d(feat_3d)

        # Pre-norm cross-attention with residual
        # 2D attends to 3D
        feat_2d = feat_2d + self.gamma_attn_2d * self.cross_attn_2d_to_3d(
            self.norm1_2d_q(feat_2d),
            self.norm1_3d_kv(feat_3d),
            self.norm1_3d_kv(feat_3d)
        )

        # 3D attends to 2D
        feat_3d = feat_3d + self.gamma_attn_3d * self.cross_attn_3d_to_2d(
            self.norm1_3d_q(feat_3d),
            self.norm1_2d_kv(feat_2d),
            self.norm1_2d_kv(feat_2d)
        )

        # Pre-norm MLP with residual
        feat_2d = feat_2d + self.gamma_mlp_2d * self.mlp_2d(self.norm2_2d(feat_2d))
        feat_3d = feat_3d + self.gamma_mlp_3d * self.mlp_3d(self.norm2_3d(feat_3d))

        return feat_2d, feat_3d


class BSTFBlockV2(nn.Module):
    """
    Enhanced Fusion Block with Adaptive Gating and Efficient Convolutions
    """
    def __init__(self, channels_2d, channels_3d, inter_channels,
                 num_heads=8, dropout=0.1, use_depthwise=True):
        super().__init__()

        # Channel reduction with GroupNorm (more stable than BatchNorm)
        self.reduce_2d = nn.Sequential(
            nn.Conv2d(channels_2d, inter_channels, kernel_size=1),
            nn.GroupNorm(num_groups=32, num_channels=inter_channels),
            nn.SiLU(inplace=True)
        )

        self.reduce_3d = nn.Sequential(
            nn.Conv2d(channels_3d, inter_channels, kernel_size=1),
            nn.GroupNorm(num_groups=32, num_channels=inter_channels),
            nn.SiLU(inplace=True)
        )

        # Bidirectional cross-attention
        self.cross_attention = BidirectionalCrossAttentionBlockV2(
            embed_dim=inter_channels,
            num_heads=num_heads,
            dropout=dropout,
            use_pos_encoding=True
        )

        # Adaptive gating mechanism for smart fusion
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(inter_channels * 2, inter_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Output projection
        if use_depthwise:
            # Depthwise separable convolution (efficient)
            self.output_conv = nn.Sequential(
                # Depthwise
                nn.Conv2d(inter_channels, inter_channels, kernel_size=3,
                         padding=1, groups=inter_channels, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=inter_channels),
                nn.SiLU(inplace=True),
                # Pointwise
                nn.Conv2d(inter_channels, inter_channels, kernel_size=1),
                nn.GroupNorm(num_groups=32, num_channels=inter_channels),
                nn.SiLU(inplace=True)
            )
        else:
            # Standard convolution
            self.output_conv = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, inter_channels, kernel_size=1),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, feat_2d, feat_3d):
        B, _, H, W = feat_2d.shape

        # Reduce channels
        feat_2d_reduced = self.reduce_2d(feat_2d)
        feat_3d_reduced = self.reduce_3d(feat_3d)

        C = feat_2d_reduced.shape[1]

        # Flatten for attention
        feat_2d_flat = feat_2d_reduced.flatten(2).transpose(1, 2)  # [B, H*W, C]
        feat_3d_flat = feat_3d_reduced.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Apply bidirectional cross-attention
        feat_2d_attended, feat_3d_attended = self.cross_attention(feat_2d_flat, feat_3d_flat)

        # Reshape back
        feat_2d_attended = feat_2d_attended.transpose(1, 2).reshape(B, C, H, W)
        feat_3d_attended = feat_3d_attended.transpose(1, 2).reshape(B, C, H, W)

        # Adaptive gating fusion
        concat_features = torch.cat([feat_2d_attended, feat_3d_attended], dim=1)
        gate = self.fusion_gate(concat_features)

        # Weighted combination based on gate
        fused = gate * feat_2d_attended + (1 - gate) * feat_3d_attended

        # Output projection
        output = self.output_conv(fused)

        return output


class BSTFv2(nn.Module):
    """
    Main Enhanced Cross-Attention Fusion Module for YOWOFormer
    Key improvements:
    - Positional encoding for spatial awareness
    - Temperature-scaled attention
    - Pre-normalization for stability
    - Adaptive gating for smart fusion
    - Efficient depthwise separable convolutions
    """
    def __init__(self, channels_2D, channels_3D, interchannels, mode='decoupled'):
        super().__init__()
        assert mode in ['coupled', 'decoupled'], "mode must be 'coupled' or 'decoupled'"
        self.mode = mode

        # Configuration flags
        self.use_depthwise = True  # Use efficient convolutions
        self.num_heads = 8  # Number of attention heads
        self.dropout = 0.1  # Dropout rate

        if mode == 'coupled':
            # Single fusion for both box and class
            self.fusion_blocks = nn.ModuleList()
            for channels_2d in channels_2D:
                self.fusion_blocks.append(
                    BSTFBlockV2(
                        channels_2d=channels_2d + channels_3D,
                        channels_3d=channels_3D,
                        inter_channels=interchannels,
                        num_heads=self.num_heads,
                        dropout=self.dropout,
                        use_depthwise=self.use_depthwise
                    )
                )

        elif mode == 'decoupled':
            # Separate fusion for box and class branches
            self.box_blocks = nn.ModuleList()
            self.cls_blocks = nn.ModuleList()

            for channels_2d in channels_2D:
                # Box branch
                self.box_blocks.append(
                    BSTFBlockV2(
                        channels_2d=channels_2d[0],
                        channels_3d=channels_3D,
                        inter_channels=interchannels,
                        num_heads=self.num_heads,
                        dropout=self.dropout,
                        use_depthwise=self.use_depthwise
                    )
                )

                # Class branch
                self.cls_blocks.append(
                    BSTFBlockV2(
                        channels_2d=channels_2d[1],
                        channels_3d=channels_3D,
                        inter_channels=interchannels,
                        num_heads=self.num_heads,
                        dropout=self.dropout,
                        use_depthwise=self.use_depthwise
                    )
                )

    def forward(self, ft_2D, ft_3D):
        """
        Forward pass with enhanced cross-attention fusion
        """
        _, C_3D, H_3D, W_3D = ft_3D.shape
        fused_features = []

        if self.mode == 'coupled':
            for idx, feat_2d in enumerate(ft_2D):
                _, _, H_2D, W_2D = feat_2d.shape

                # Upsample 3D features if needed
                scale_factor = H_2D / H_3D
                if scale_factor != 1:
                    upsampler = nn.Upsample(
                        scale_factor=scale_factor,
                        mode='bilinear',
                        align_corners=False
                    )
                    ft_3D_upsampled = upsampler(ft_3D)
                else:
                    ft_3D_upsampled = ft_3D

                # Apply fusion
                fused = self.fusion_blocks[idx](feat_2d, ft_3D_upsampled)
                fused_features.append(fused)

        elif self.mode == 'decoupled':
            for idx, feat_2d in enumerate(ft_2D):
                _, _, H_2D, W_2D = feat_2d[0].shape

                # Upsample 3D features if needed
                scale_factor = H_2D / H_3D
                if scale_factor != 1:
                    upsampler = nn.Upsample(
                        scale_factor=scale_factor,
                        mode='bilinear',
                        align_corners=False
                    )
                    ft_3D_upsampled = upsampler(ft_3D)
                else:
                    ft_3D_upsampled = ft_3D

                # Apply fusion for box and class branches
                box_fused = self.box_blocks[idx](feat_2d[0], ft_3D_upsampled)
                cls_fused = self.cls_blocks[idx](feat_2d[1], ft_3D_upsampled)

                fused_features.append([box_fused, cls_fused])

        return fused_features


# Alias for compatibility
# Backward compatibility aliases
CrossAttentionFusionV2 = BSTFv2
CrossAttentionV2 = BSTFv2