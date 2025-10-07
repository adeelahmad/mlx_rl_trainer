"""
Configuration management system using Pydantic for validation and predictability.
"""
import logging
import re
import string
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Literal

from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    NonNegativeFloat,
    ValidationError,
    model_validator,
    ConfigDict,
)
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

THINK_STYLE_PROMPT_LITERAL = """THINKING RULES - Use maximally compressed notation:
... [Full text of THINK_STYLE_PROMPT_LITERAL as in CURRENT_STATE] ..."""

class RewardConfig(BaseModel):
    name: str = Field(..., description="Registered name of the reward function.")
    weight: float = Field(1.0, ge=0.0, le=1.0)
    config: Dict[str, Any] = Field(default_factory=dict)

class EvaluatorConfig(BaseModel):
    name: str = Field(..., description="Registered name of the evaluator.")
    config: Dict[str, Any] = Field(default_factory=dict)

class DataConfig(BaseModel):
    train_path: Path
    val_path: Optional[Path] = None
    max_prompt_len: PositiveInt = 512
    max_gen_len: PositiveInt = 384
    loader_type: Literal["jsonl", "hf_dataset", "mock"] = "jsonl"
    shuffle_data: bool = True
    dataset_prompt_key: str = "prompt"
    dataset_answer_key: str = "completion"
    dataset_filter_keywords: List[str] = Field(default_factory=list)

class ModelConfig(BaseModel):
    model_path: Path
    ref_model_path: Optional[Path] = None
    use_lora: bool = False
    lora_rank: PositiveInt = 8
    lora_alpha: float = 16.0
    lora_dropout: NonNegativeFloat = 0.0
    lora_scale_by_rank: bool = True
    lora_target_modules: List[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])

    @model_validator(mode="after")
    def set_default_ref_model_path(self) -> "ModelConfig":
        if self.ref_model_path is None:
            self.ref_model_path = self.model_path
        return self

class CheckpointConfig(BaseModel):
    save_dir: Path = Path("checkpoints")
    save_every: PositiveInt = 100
    keep_last_n: PositiveInt = 3
    save_optimizer_state: bool = False

class MonitoringConfig(BaseModel):
    use_wandb: bool = True
    wandb_project: Optional[str] = "mlx-rl-trainer"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_samples_every: PositiveInt = 10
    max_logged_samples: PositiveInt = 5
    log_prompts: bool = True
    sample_log_path: Optional[Path] = None

class TrainerParams(BaseModel):
    algorithm: Literal["grpo", "ppo"] = "grpo"
    output_dir: Path = Path("./outputs")
    num_training_steps: PositiveInt = 10000
    learning_rate: NonNegativeFloat = 2e-6
    ppo_batch_size: PositiveInt = 4
    num_rollout_samples: PositiveInt = 2
    grad_accum_steps: PositiveInt = 4
    grpo_beta: NonNegativeFloat = 0.05
    seed: int = 42

    optimizer_beta1: NonNegativeFloat = 0.9
    optimizer_beta2: NonNegativeFloat = 0.95
    optimizer_weight_decay: NonNegativeFloat = 0.01
    grad_clip_norm: Optional[NonNegativeFloat] = 0.5
    
    lr_schedule_config: Dict[str, Any] = Field(default_factory=dict)
    
    use_grad_checkpointing: bool = False
    eval_every: PositiveInt = 250
    reward_smoothing_window: PositiveInt = 20
    effective_batch_size: int = Field(0, init=False, exclude=True)

class GenerationConfig(BaseModel):
    think_start_tag: str = "<think>"
    think_end_tag: str = "</think>"
    answer_start_tag: str = "<answer>"
    answer_end_tag: str = "</answer>"
    think_boost_tokens: int = 32
    think_temperature: NonNegativeFloat = 0.23
    answer_temperature: NonNegativeFloat = 0.24
    sampling_top_p: NonNegativeFloat = 0.7
    sampling_min_p: NonNegativeFloat = 0.02
    sampling_top_k: int = 20
    repetition_penalty: float = 1.15
    repetition_context_size: int = 20
    
    min_think_tokens: int = 32
    think_end_early_bias: float = -12.0
    bias_answer_start_after_min_think: bool = True
    bias_close_think: float = 9.0
    bias_answer_start: float = 6.0
    punish_extra_think_end: float = -12.0
    punish_reopen_think: float = -10.0
    punish_reopen_answer: float = -9.0
    bias_eos_after_answer: float = 3.0

    hard_mask_mcq_first_token: bool = True
    mcq_letter_lift: float = 8.0
    mcq_ban_first_bias: float = -14.0
    nonmcq_ban_first_bias: float = -12.0
    mcq_close_after_k: int = 1
    min_answer_tokens: int = 8
    min_answer_tokens_mcq: int = 1
    mcq_answer_end_bias: float = 9.0

    ban_phrases_for_bias: List[str] = Field(default_factory=list)
    encourage_phrases_for_bias: List[str] = Field(default_factory=list)
    encourage_think_bias: float = 4.5
    ban_think_bias: float = -3.0

    allow_tool_calls: bool = True
    tool_call_penalty: NonNegativeFloat = 0.0

class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trainer: TrainerParams
    model: ModelConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    rewards: List[RewardConfig] = Field(default_factory=list)
    data: DataConfig
    evaluation: List[EvaluatorConfig] = Field(default_factory=list)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    max_kv_size: PositiveInt = 1536
    system_prompt: str = "" # Default to empty, can be overridden in YAML

    @model_validator(mode="after")
    def populate_derived_fields(self) -> "ExperimentConfig":
        self.trainer.effective_batch_size = (
            self.trainer.ppo_batch_size * self.trainer.num_rollout_samples * self.trainer.grad_accum_steps
        )
        cfg = self.trainer.lr_schedule_config
        init_lr = float(self.trainer.learning_rate)
        total_steps = int(self.trainer.num_training_steps)
        warmup_steps = int(cfg.get("warmup", 500))
        decay_steps = max(total_steps - warmup_steps, 1)
        end_lr = max(init_lr * 0.1, 1e-7)
        cfg.setdefault("name", "cosine_decay")
        cfg.setdefault("arguments", [init_lr, decay_steps, end_lr])
        cfg.setdefault("warmup", warmup_steps)
        cfg.setdefault("warmup_init", min(init_lr, max(init_lr * 0.1, 1e-8)))
        return self

    @classmethod
    def load_from_yaml(cls, path: Path) -> "ExperimentConfig":
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
            instance = cls(**raw_config)
            instance.trainer.output_dir.mkdir(parents=True, exist_ok=True)
            return instance
        except ValidationError as e:
            console.print(f"[bold red]Configuration Validation Error in {path}:[/bold red]\n{e}")
            raise ValueError(f"Invalid configuration in {path}.") from e
        except Exception as e:
            console.print(f"[bold red]Failed to load configuration from {path}:[/bold red] {e}")
            raise ValueError(f"Failed to load configuration from {path}.") from e
