from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class TrainingMetrics:
    loss: float
    reward_mean: float
    grad_norm: float
    learning_rate: float
    step_time_s: float
    kl_divergence: float
    epoch: int = 0
    step: int = 0
    reward_std: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        metrics = {
            "train/loss": self.loss,
            "train/reward_mean": self.reward_mean,
            "train/reward_std": self.reward_std,
            "train/grad_norm": self.grad_norm,
            "train/learning_rate": self.learning_rate,
            "train/step_time_s": self.step_time_s,
            "train/kl_divergence": self.kl_divergence,
            "train/epoch": self.epoch,
            "train/step": self.step,
        }
        metrics.update(self.custom_metrics)
        return metrics


@dataclass(frozen=True)
class EvaluationMetrics:
    task_name: str
    pass_rate: Optional[float] = None
    perplexity: Optional[float] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, any]:
        metrics = {}
        prefix = f"eval/{self.task_name}"
        if self.pass_rate is not None:
            metrics[f"{prefix}/pass_rate"] = self.pass_rate
        if self.perplexity is not None:
            metrics[f"{prefix}/perplexity"] = self.perplexity
        for k, v in self.additional_info.items():
            # Handle nested dicts from some evals
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    metrics[f"{prefix}/{k}/{sub_k}"] = sub_v
            else:
                metrics[f"{prefix}/{k}"] = v
        return metrics
