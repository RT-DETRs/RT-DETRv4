"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
"""

import torch.nn as nn
from ..core import register


__all__ = ['RTv4', ]


@register()
class RTv4(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None, teacher_encoder_output=None):
        x_backbone = self.backbone(x)  # [S3, S4, S5] features from backbone

        encoder_output = self.encoder(x_backbone)
        # tuple: (fpn_features, student_distill_output) or fpn_features (list) if not training or distillation is off.

        student_distill_output = None
        if self.training and isinstance(encoder_output, tuple) and len(encoder_output) == 2:
            x_fpn_features, student_distill_output = encoder_output
        else:
            x_fpn_features = encoder_output

        x_decoder_out = self.decoder(x_fpn_features, targets)

        if self.training and student_distill_output is not None and teacher_encoder_output is not None:
            x_decoder_out['student_distill_output'] = student_distill_output
            x_decoder_out['teacher_encoder_output'] = teacher_encoder_output

        return x_decoder_out

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
