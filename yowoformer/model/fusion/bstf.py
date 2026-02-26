"""
True Cross-Attention Fusion Module for YOWOFormer
This module performs bidirectional cross-attention between 2D and 3D features
allowing them to attend to each other for better feature fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention mechanism
    Allows one modality to attend to another modality
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, return_attention=False):
        """
        Args:
            query: [B, N_q, C] - features that will attend
            key: [B, N_kv, C] - features to attend to
            value: [B, N_kv, C] - features to aggregate
            return_attention: whether to return attention weights
        Returns:
            output: [B, N_q, C] - attended features
            attention_weights: [B, H, N_q, N_kv] - attention weights (optional)
        """
        B, N_q, C = query.shape
        N_kv = key.shape[1]

        # Linear projections in batch
        Q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, N_q, C)
        output = self.out_proj(attn_output)

        if return_attention:
            return output, attn_weights
        return output


class BidirectionalAttentionBlock(nn.Module):
    """
    Bidirectional Cross-Attention Block
    Performs cross-attention in both directions (2D->3D and 3D->2D)
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, mlp_ratio=4.0):
        super().__init__()

        # Cross-attention layers
        self.cross_attn_2d_to_3d = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.cross_attn_3d_to_2d = MultiHeadCrossAttention(embed_dim, num_heads, dropout)

        # Layer normalization
        self.norm1_2d = nn.LayerNorm(embed_dim)
        self.norm1_3d = nn.LayerNorm(embed_dim)
        self.norm2_2d = nn.LayerNorm(embed_dim)
        self.norm2_3d = nn.LayerNorm(embed_dim)

        # Feed-forward networks
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp_2d = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.mlp_3d = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, feat_2d, feat_3d):
        """
        Args:
            feat_2d: [B, N_2d, C] - 2D features
            feat_3d: [B, N_3d, C] - 3D features
        Returns:
            feat_2d_out: [B, N_2d, C] - enhanced 2D features
            feat_3d_out: [B, N_3d, C] - enhanced 3D features
        """
        # 2D features attend to 3D features
        feat_2d_attended = self.cross_attn_2d_to_3d(
            query=feat_2d,
            key=feat_3d,
            value=feat_3d
        )
        feat_2d = feat_2d + feat_2d_attended
        feat_2d = self.norm1_2d(feat_2d)

        # 3D features attend to 2D features
        feat_3d_attended = self.cross_attn_3d_to_2d(
            query=feat_3d,
            key=feat_2d,
            value=feat_2d
        )
        feat_3d = feat_3d + feat_3d_attended
        feat_3d = self.norm1_3d(feat_3d)

        # Feed-forward networks with residual connections
        feat_2d = feat_2d + self.mlp_2d(self.norm2_2d(feat_2d))
        feat_3d = feat_3d + self.mlp_3d(self.norm2_3d(feat_3d))

        return feat_2d, feat_3d


class BSTFBlock(nn.Module):
    """
    Cross-Attention Fusion Block for single branch (box or cls)
    """
    def __init__(self, channels_2d, channels_3d, inter_channels, num_heads=8, dropout=0.1):
        super().__init__()

        # Channel reduction layers
        self.reduce_2d = nn.Sequential(
            nn.Conv2d(channels_2d, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        self.reduce_3d = nn.Sequential(
            nn.Conv2d(channels_3d, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        # Bidirectional cross-attention
        self.cross_attention = BidirectionalAttentionBlock(
            embed_dim=inter_channels,
            num_heads=num_heads,
            dropout=dropout
        )

        # Output projection
        self.output_conv = nn.Sequential(
            nn.Conv2d(inter_channels * 2, inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_2d, feat_3d):
        """
        Args:
            feat_2d: [B, C_2d, H, W] - 2D feature map
            feat_3d: [B, C_3d, H, W] - 3D feature map (upsampled to match 2D resolution)
        Returns:
            output: [B, inter_channels, H, W] - fused features
        """
        B, _, H, W = feat_2d.shape

        # Reduce channels
        feat_2d_reduced = self.reduce_2d(feat_2d)  # [B, inter_channels, H, W]
        feat_3d_reduced = self.reduce_3d(feat_3d)  # [B, inter_channels, H, W]

        C = feat_2d_reduced.shape[1]

        # Flatten spatial dimensions for attention
        feat_2d_flat = feat_2d_reduced.flatten(2).transpose(1, 2)  # [B, H*W, C]
        feat_3d_flat = feat_3d_reduced.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Apply bidirectional cross-attention
        feat_2d_attended, feat_3d_attended = self.cross_attention(feat_2d_flat, feat_3d_flat)

        # Reshape back to spatial format
        feat_2d_attended = feat_2d_attended.transpose(1, 2).reshape(B, C, H, W)
        feat_3d_attended = feat_3d_attended.transpose(1, 2).reshape(B, C, H, W)

        # Concatenate and fuse
        fused = torch.cat([feat_2d_attended, feat_3d_attended], dim=1)
        output = self.output_conv(fused)

        return output


class BSTF(nn.Module):
    """
    Main Cross-Attention Fusion Module for YOWOFormer
    Performs bidirectional cross-attention between 2D and 3D features
    """
    def __init__(self, channels_2D, channels_3D, interchannels, mode='decoupled'):
        super().__init__()
        assert mode in ['coupled', 'decoupled'], "mode must be 'coupled' or 'decoupled'"
        self.mode = mode

        if mode == 'coupled':
            # For coupled mode: single fusion for both box and class
            self.fusion_blocks = nn.ModuleList()
            for channels_2d in channels_2D:
                self.fusion_blocks.append(
                    BSTFBlock(
                        channels_2d=channels_2d + channels_3D,
                        channels_3d=channels_3D,
                        inter_channels=interchannels,
                        num_heads=8,
                        dropout=0.1
                    )
                )

        elif mode == 'decoupled':
            # For decoupled mode: separate fusion for box and class
            self.box_blocks = nn.ModuleList()
            self.cls_blocks = nn.ModuleList()

            for channels_2d in channels_2D:
                # Box branch
                self.box_blocks.append(
                    BSTFBlock(
                        channels_2d=channels_2d[0],
                        channels_3d=channels_3D,
                        inter_channels=interchannels,
                        num_heads=8,
                        dropout=0.1
                    )
                )

                # Class branch
                self.cls_blocks.append(
                    BSTFBlock(
                        channels_2d=channels_2d[1],
                        channels_3d=channels_3D,
                        inter_channels=interchannels,
                        num_heads=8,
                        dropout=0.1
                    )
                )

    def forward(self, ft_2D, ft_3D):
        """
        Args:
            ft_2D: list of 2D features at different scales
                   - coupled mode: list of [B, C, H, W] tensors
                   - decoupled mode: list of [[B, C_box, H, W], [B, C_cls, H, W]] tensors
            ft_3D: 3D features [B, C, H, W]
        Returns:
            fused_features: fused features in the same format as input
        """
        _, C_3D, H_3D, W_3D = ft_3D.shape
        fused_features = []

        if self.mode == 'coupled':
            for idx, feat_2d in enumerate(ft_2D):
                _, _, H_2D, W_2D = feat_2d.shape

                # Upsample 3D features to match 2D resolution
                scale_factor = H_2D / H_3D
                if scale_factor != 1:
                    upsampler = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
                    ft_3D_upsampled = upsampler(ft_3D)
                else:
                    ft_3D_upsampled = ft_3D

                # Apply cross-attention fusion
                fused = self.fusion_blocks[idx](feat_2d, ft_3D_upsampled)
                fused_features.append(fused)

        elif self.mode == 'decoupled':
            for idx, feat_2d in enumerate(ft_2D):
                _, _, H_2D, W_2D = feat_2d[0].shape  # Use box branch to get dimensions

                # Upsample 3D features to match 2D resolution
                scale_factor = H_2D / H_3D
                if scale_factor != 1:
                    upsampler = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
                    ft_3D_upsampled = upsampler(ft_3D)
                else:
                    ft_3D_upsampled = ft_3D

                # Apply cross-attention fusion for box and class branches
                box_fused = self.box_blocks[idx](feat_2d[0], ft_3D_upsampled)
                cls_fused = self.cls_blocks[idx](feat_2d[1], ft_3D_upsampled)

                fused_features.append([box_fused, cls_fused])

        return fused_features


# For backward compatibility
# Backward compatibility aliases
CrossAttentionFusion = BSTF
CrossAttention = BSTF