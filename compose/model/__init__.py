from compose.model.ct import ChamferTrainedRouter
from compose.model.inference import InferenceConfig, compose_scores, predict
from compose.model.router import BiGRURouter, SlotRouter


def build_router(cfg, phi_dim: int) -> SlotRouter:
    """Router for a :class:`compose.config.TrainConfig`."""
    if cfg.router == "compose":
        return SlotRouter(phi_dim, hidden_dim=cfg.hidden_dim, lambda_d=cfg.lambda_d)
    if cfg.router == "compose_ct":
        return ChamferTrainedRouter(
            phi_dim, hidden_dim=cfg.hidden_dim, lambda_d=cfg.lambda_d,
            holistic_weight=cfg.holistic_weight, kappa=cfg.kappa,
        )
    if cfg.router == "bigru":
        return BiGRURouter(phi_dim, hidden_dim=cfg.hidden_dim, lambda_d=cfg.lambda_d)
    raise ValueError(f"unknown router {cfg.router!r}")


__all__ = [
    "BiGRURouter", "ChamferTrainedRouter", "InferenceConfig", "SlotRouter",
    "build_router", "compose_scores", "predict",
]
