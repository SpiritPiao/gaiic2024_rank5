# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .cspnext import CSPNeXt
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .dual_resnet import Dual_ResNet
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .dual_swin import Dual_SwinTransformer
from .dual_swin_reg import Dual_SwinTransformer_Reg
from .dual_swin_c2former import Dual_SwinTransformer_C2Former
from .dual_swin_cbnet_pki import Dual_SwinTransformer_CBPki
from .dual_swin_cbnet_pkiv2 import Dual_SwinTransformer_CBPkiv2
from .dual_swin_cbnet_pkiv3 import Dual_SwinTransformer_CBPkiv3
from .dual_swin_cbnet_swin import Dual_SwinTransformer_CBSwin
from .trident_resnet import TridentResNet
from .vit import ViT

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'EfficientNet', 'CSPNeXt','Dual_ResNet','Dual_SwinTransformer','Dual_SwinTransformer_Reg',
    'Dual_SwinTransformer_C2Former', 'ViT'
]
