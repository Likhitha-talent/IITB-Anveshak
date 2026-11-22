"""
Detector Architecture for Open-World Object Detection from Scratch.
Features:
  - ResNet backbone with Group Normalization (GN) instead of BatchNorm (crucial for scratch training stability)
  - Random weight initialization (Zero pretrained weights)
  - Cosine Similarity Classification Head with Margin
  - Helmholtz Free Energy Head for Unknown Object Discovery
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


class CosineClassifier(nn.Module):
    """
    Cosine Similarity Classifier Head.
    Normalizes feature representations and class weight vectors to prevent
    catastrophic weight magnitude drift during continual few-shot learning.
    """
    def __init__(self, in_features: int, num_classes: int, scale: float = 20.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2-normalize features and weights
        x_norm = F.normalize(x, p=2, dim=-1)
        w_norm = F.normalize(self.weight, p=2, dim=-1)
        logits = self.scale * F.linear(x_norm, w_norm)
        return logits


class FreeEnergyScorer:
    """
    Calculates Helmholtz Free Energy of classification logits to detect
    Out-of-Distribution (OOD) unknown objects:
      E(x) = -T * log(sum(exp(logits_i / T)))
    Lower energy = High probability of being a known class.
    Higher energy = High probability of being an unknown object.
    """
    @staticmethod
    def compute_energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        # logits shape: [N, num_classes]
        return -temperature * torch.logsumexp(logits / temperature, dim=-1)


def build_scratch_detector(
    num_classes: int = 40,
    pretrained: bool = False
) -> FasterRCNN:
    """
    Constructs a detector initialized strictly from scratch with Group Normalization.
    Pretrained weights are NEVER loaded per Anveshak competition rules.
    """
    assert not pretrained, "Competition strictly forbids pretrained weights!"

    # Create ResNet-50 with GroupNorm initialized randomly
    backbone = torchvision.models.resnet50(weights=None, norm_layer=lambda num_channels: nn.GroupNorm(32, num_channels))
    
    # Extract feature stages for FPN
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    in_channels_list = [256, 512, 1024, 2048]
    out_channels = 256

    fpn = torchvision.ops.FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=torchvision.ops.feature_pyramid_network.LastLevelMaxPool()
    )

    class BackboneWithFPN(nn.Module):
        def __init__(self, bb, fpn_net):
            super().__init__()
            self.body = torchvision.models._utils.IntermediateLayerGetter(bb, return_layers=return_layers)
            self.fpn = fpn_net
            self.out_channels = out_channels

        def forward(self, x):
            x = self.body(x)
            x = self.fpn(x)
            return x

    backbone_with_fpn = BackboneWithFPN(backbone, fpn)

    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    # Instantiate detector (num_classes includes background)
    model = FasterRCNN(
        backbone=backbone_with_fpn,
        num_classes=num_classes + 1,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    # Initialize weights explicitly with Kaiming Normal for scratch stability
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    return model
