"""ONNX export wrapper for RT-DETRv4 with spatial prior inputs."""

import torch
import torch.nn as nn


class SpatialPriorExportWrapper(nn.Module):
    """Wraps a deployed RTv4 model for ONNX export with spatial prior.

    Inputs:
        images             [1, 3, 640, 640]  float32
        spatial_prior      [1, 4]            float32  — [cx, cy, w, h] in [0,1]
        use_spatial_prior  [1]               int64    — 1 = inject, 0 = normal

    Output:
        detections         [1, 300, 4+C]     float32
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, spatial_prior, use_spatial_prior):
        outputs = self.model(images,
                             spatial_prior=spatial_prior,
                             use_spatial_prior=use_spatial_prior)
        boxes = outputs["pred_boxes"]
        scores = torch.sigmoid(outputs["pred_logits"])
        return torch.cat([boxes, scores], dim=-1)
