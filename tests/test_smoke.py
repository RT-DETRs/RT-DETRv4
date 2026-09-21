"""
Minimal smoke tests for correctness regressions that a full training/eval run
wouldn't easily surface in CI (no dataset, no pretrained weights, and on most
CI runners no GPU).

These specifically guard the two bugs fixed alongside this test:
  - HybridEncoder's cached `pos_embed{idx}` tensors must be real buffers so
    they move with the module via `.to()` / `.cuda()` (previously a plain
    `setattr` attribute that silently stayed on CPU after `model.cuda()`).
  - RTv4Criterion.loss_distillation must not hardcode `torch.device('cuda')`
    when both student/teacher feature maps are absent.
"""
import pytest
import torch

from engine.rtv4.hybrid_encoder import HybridEncoder
from engine.rtv4.rtv4_criterion import RTv4Criterion


def test_hybrid_encoder_pos_embed_is_a_real_buffer():
    model = HybridEncoder(eval_spatial_size=(640, 640))
    model.eval()

    buffer_names = dict(model.named_buffers())
    pos_embed_name = f"pos_embed{model.use_encoder_idx[0]}"
    assert pos_embed_name in buffer_names, (
        f"{pos_embed_name} must be a registered buffer so it moves with "
        "the module via .to()/.cuda()"
    )

    # A real buffer follows dtype/device conversions; a plain attribute set
    # via setattr() would not, which was the root cause of the regression.
    model = model.to(torch.float64)
    assert getattr(model, pos_embed_name).dtype == torch.float64


def test_hybrid_encoder_eval_forward_with_fixed_spatial_size():
    model = HybridEncoder(eval_spatial_size=(640, 640))
    model.eval()

    feats = [
        torch.randn(1, 512, 80, 80),
        torch.randn(1, 1024, 40, 40),
        torch.randn(1, 2048, 20, 20),
    ]
    with torch.no_grad():
        outputs = model(feats)

    assert len(outputs) == len(feats)
    for out, feat in zip(outputs, feats):
        # HybridEncoder projects every input to `hidden_dim` channels;
        # only the spatial dims are preserved.
        assert out.shape == (feat.shape[0], model.hidden_dim, *feat.shape[2:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_hybrid_encoder_eval_forward_on_cuda_after_model_to_cuda():
    # Reproduces the exact real-world regression: build on CPU (default),
    # move the whole model to GPU, then run eval-mode forward with the
    # fixed eval_spatial_size path. Before the fix this raised:
    #   RuntimeError: Expected all tensors to be on the same device...
    model = HybridEncoder(eval_spatial_size=(640, 640))
    model.eval()
    model = model.cuda()

    feats = [
        torch.randn(1, 512, 80, 80, device="cuda"),
        torch.randn(1, 1024, 40, 40, device="cuda"),
        torch.randn(1, 2048, 20, 20, device="cuda"),
    ]
    with torch.no_grad():
        outputs = model(feats)

    for out in outputs:
        assert out.device.type == "cuda"


def test_criterion_distill_loss_uses_available_device_not_hardcoded_cuda():
    criterion = RTv4Criterion(matcher=None, weight_dict={}, losses=[])
    outputs = {"pred_boxes": torch.randn(1, 10, 4)}

    result = criterion.loss_distillation(
        outputs, targets=None, indices=None, num_boxes=None
    )

    loss = result["loss_distill"]
    assert loss.device == outputs["pred_boxes"].device
    assert loss.item() == 0.0
    assert loss.requires_grad
