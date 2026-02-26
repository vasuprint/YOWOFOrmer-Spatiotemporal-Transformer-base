from .cfam import CFAMFusion
from .se import SEFusion
from .simple import SimpleFusion
from .multihead import MultiHeadFusion
from .channel import ChannelFusion
from .spatial import SpatialFusion
from .cbam import CBAMFusion
from .lka import LKAFusion
from .bstf import BSTF
from .bstf_v2 import BSTFv2

# major of Attention are adopted from https://viblo.asia/p/mot-chut-ve-co-che-attention-trong-computer-vision-x7Z4D622LnX
def build_fusion_block(out_channels_2D, out_channels_3D, inter_channels_fusion, mode, fusion_block, lastdimension):
    if fusion_block == 'CFAM':
        return CFAMFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'SE':
        return SEFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'Simple':
        return SimpleFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'MultiHead':
        return MultiHeadFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, lastdimension, mode, h=1)
    elif fusion_block == 'Channel':
        return ChannelFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'Spatial':
        return SpatialFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'CBAM':
        return CBAMFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'LKA':
        return LKAFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block in ('CrossAttention', 'BSTF'):
        return BSTF(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block in ('CrossAttentionV2', 'BSTFv2'):
        return BSTFv2(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)