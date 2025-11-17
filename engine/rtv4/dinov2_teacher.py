"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core import register
import logging
from torchvision.transforms import v2 as transforms
import torchvision.transforms.functional as TF
import math

_logger = logging.getLogger(__name__)


@register()
class DINOv2TeacherModel(nn.Module):
    """
    DINOv2 (ViT-L/14 w/ Reg) Teacher Model.
    - Loads model from a local torch.hub repo.
    - Dynamically resizes input images so that the output patch grid
      matches the student's target feature map grid.
    """

    def __init__(self,
                 dinov2_repo_path: str,
                 model_name: str = 'dinov2_vitl14_reg',
                 weights_path: str = None,
                 patch_size: int = 14,
                 target_downsample_factor: int = 32,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.patch_size = patch_size
        self.target_downsample_factor = target_downsample_factor

        _logger.info(f"[Teacher Model] Initializing DINOv2 teacher via torch.hub.load...")
        _logger.info(f"[Teacher Model] DINOv2 repo path: {dinov2_repo_path}")
        _logger.info(f"[Teacher Model] Model name: {model_name}")

        try:
            # Load model architecture
            self.model = torch.hub.load(
                dinov2_repo_path,
                model_name,
                source='local'
            )

            # Load weights
            if weights_path:
                _logger.info(f"[Teacher Model] Loading weights from: {weights_path}")
                state_dict = torch.load(weights_path, map_location='cpu')
                msg = self.model.load_state_dict(state_dict, strict=False)
                _logger.info(f"[Teacher Model] Weight loading info: {msg}")
            else:
                _logger.warning("[Teacher Model] No 'weights_path' provided. Using random weights.")

            self.teacher_feature_dim = self.model.embed_dim  # 1024 for ViT-L
            _logger.info(f"[Teacher Model] Feature dimension: {self.teacher_feature_dim}")

            # Freeze model and set to eval mode
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        except Exception as e:
            _logger.error(f"[Teacher Model] Failed to load or setup DINOv2: {e}", exc_info=True)
            raise

        # Input normalization transform
        self.normalize_transform = transforms.Normalize(mean=mean, std=std)

        _logger.info(f"[Teacher Model] DINOv2 (Hub) initialized.")

    def forward(self, images: torch.Tensor):
        B, _, H_in, W_in = images.shape

        normalized_images = self.normalize_transform(images)

        target_H_out = torch.tensor(H_in / self.target_downsample_factor)
        target_W_out = torch.tensor(W_in / self.target_downsample_factor)

        target_H_vit_in = torch.round(target_H_out * self.patch_size).int().item()
        target_W_vit_in = torch.round(target_W_out * self.patch_size).int().item()

        processed_images = TF.resize(
            normalized_images,
            [target_H_vit_in, target_W_vit_in],
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True
        )

        with torch.no_grad():
            # DINOv2 returns a dict
            outputs_dict = self.model.forward_features(processed_images)

            # 'x_norm_patchtokens' contains only patch tokens
            patch_tokens = outputs_dict['x_norm_patchtokens']  # [B, N_patches, C]

            B, N_patches, C_teacher = patch_tokens.shape

            H_patches_expected = processed_images.shape[2] // self.patch_size
            W_patches_expected = processed_images.shape[3] // self.patch_size

            if H_patches_expected * W_patches_expected != N_patches:
                _logger.error(
                    f"[Teacher Model] DINOv2 patch token mismatch! "
                    f"Input: {processed_images.shape[2:]}, Patches: {self.patch_size} -> Expected {H_patches_expected * W_patches_expected} tokens. "
                    f"Got {N_patches} tokens."
                )
                raise ValueError("DINOv2 patch token count mismatch.")

            # Reshape to spatial feature map
            output_feature_map = patch_tokens.permute(0, 2, 1).reshape(
                B, C_teacher, H_patches_expected, W_patches_expected
            )  # Shape: [B, 1024, H_in/32, W_in/32]

            return output_feature_map.detach()