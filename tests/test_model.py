import torch
import torch.nn.functional as F

from compose.model import ChamferTrainedRouter, InferenceConfig, SlotRouter, compose_scores


def _episode(n_way=3, shots=2, queries=4, k=6, d=12):
    torch.manual_seed(0)
    sup = F.normalize(torch.randn(n_way * shots, k + 1, d), dim=-1)
    qry = F.normalize(torch.randn(queries, k + 1, d), dim=-1)
    return sup, torch.arange(n_way).repeat_interleave(shots), qry


def test_holistic_weight_one_is_compose():
    sup, lbl, qry = _episode()
    torch.manual_seed(1)
    a = SlotRouter(12, hidden_dim=8)
    torch.manual_seed(1)
    b = ChamferTrainedRouter(12, hidden_dim=8, holistic_weight=1.0)
    la, ra = a.episode_logits(sup, lbl, qry)
    lb, rb = b.episode_logits(sup, lbl, qry)
    assert torch.allclose(la, lb) and torch.allclose(ra, rb)


def test_inference_switches():
    sup, lbl, qry = _episode()
    model = SlotRouter(12, hidden_dim=8)
    base, _ = compose_scores(model, sup, lbl, qry, InferenceConfig(gamma=0.0, kappa=3))
    for cfg in (InferenceConfig(gamma=0.0, kappa=3, center=False), InferenceConfig(gamma=0.0, kappa=3, matcher="hungarian"),
                InferenceConfig(gamma=1.0, kappa=3, pool="mean")):
        other, _ = compose_scores(model, sup, lbl, qry, cfg)
        assert other.shape == base.shape and torch.isfinite(other).all()
        assert not torch.allclose(other, base), cfg


def test_router_starts_from_identity_projection():
    model = SlotRouter(12, hidden_dim=8)
    assert torch.equal(model.proj.weight, torch.eye(12)) and torch.equal(model.proj.bias, torch.zeros(12))
