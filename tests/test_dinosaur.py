import os

import pytest
import torch

timm = pytest.importorskip("timm")

from compose.slots.dinosaur import DINOSAUR  # noqa: E402


def test_forward_shapes_with_random_weights():
    model = DINOSAUR(n_slots=7).eval()
    out = model(torch.randn(2, 3, 224, 224), generator=torch.Generator().manual_seed(0))
    assert out["phi"].shape == (2, 7, 768) and out["cls"].shape == (2, 768)
    assert out["slots"].shape == (2, 7, 256) and out["attn_maps"].shape == (2, 7, 256)
    assert torch.allclose(out["attn_maps"].sum(1), torch.ones(2, 256), atol=1e-5)   # softmax over slots


@pytest.mark.skipif(not os.environ.get("DINOSAUR_CKPT"), reason="set DINOSAUR_CKPT to the released checkpoint")
def test_checkpoint_loads():
    DINOSAUR.from_checkpoint(os.environ["DINOSAUR_CKPT"])
