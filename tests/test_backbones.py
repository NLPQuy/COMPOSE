from pathlib import Path

import pytest
import torch

from compose.data.slot_store import phi_full

timm = pytest.importorskip("timm")
from compose.slots.backbones import PatchSource, readout  # noqa: E402

CONFIGS = Path(__file__).parents[1] / "configs" / "backbones"


def test_backbone_configs_load():
    names = {PatchSource.from_yaml(p).name for p in CONFIGS.glob("*.yaml")}
    assert {"dinov1_b16", "ibot_b16", "openclip_l14", "mae_vith14_L15", "sam_vitb16_L8_pool", "eva02_b14"} <= names
    for p in CONFIGS.glob("*.yaml"):
        assert PatchSource.from_yaml(p).name == p.stem


def test_readout_swaps_tokens():
    backbone = timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=0, img_size=224).eval()
    attn = torch.softmax(torch.randn(2, 7, 256), dim=1)
    dino_out = {"attn_maps": attn, "cls": torch.randn(2, 768), "phi": torch.randn(2, 7, 768)}
    out = readout(PatchSource("t", "vit_small_patch14_dinov2"), backbone, dino_out, torch.randn(2, 3, 224, 224))
    assert out["phi"].shape == (2, 7, 384) and out["cls"].shape == (2, 384)
    with pytest.raises(RuntimeError, match="patch grids differ"):
        readout(PatchSource("t", "x"), backbone, {**dino_out, "attn_maps": attn[..., :196]}, torch.randn(2, 3, 224, 224))


def test_store_without_cls_keeps_every_slot():
    data = {"s_phi_tilde": torch.nn.functional.normalize(torch.randn(4, 7, 8), dim=-1), "s_labels": torch.arange(4)}
    full, _ = phi_full(data, "s")
    assert full.shape == (4, 8, 8) and torch.equal(full[:, 0], torch.zeros(4, 8))
    assert torch.equal(full[:, 1:], data["s_phi_tilde"])
