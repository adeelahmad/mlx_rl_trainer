# file_path: mlx_rl_trainer/src/mlx_rl_trainer/core/config.py
# revision_no: 005
# goals_of_writing_code_block: Define Pydantic models for configuration, ensuring all previous training arguments and reward parameters are present, correctly typed, validated, and defaulted. This includes comprehensive generation and bias controls.
# type_of_code_response: change existing
"""
Configuration management system using Pydantic for validation and predictability.
"""
import logging
import re
import string
import yaml
import dataclasses  # For dataclasses.fields
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Literal, Union, get_origin

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

# --- Base THINK_STYLE_PROMPT as a module-level constant ---
THINK_STYLE_PROMPT_LITERAL = """THINKING RULES - Use maximally compressed notation:

═══ SYMBOLS & NOTATION ═══
Math: ∴(therefore) ∵(because) ⇒(implies) ≈(approx) ∈(in) ∀(forall) ∃(exists) ≠ ≤ ≥
Logic: ✓(yes) ✗(no) ?(unknown) !(important) ⚠(warning) ∧(and) ∨(or) ¬(not) ⊕(xor)
Flow: →(then) ←(from) ↔(bidirect) ⇄(exchange) ▸(next) ◂(prev) ⊃(implies) ⊂(subset)
Status: ✓(done) ○(pending) ●(active) ◐(partial) ⊗(blocked) ⊘(invalid)

═══ UNIVERSAL ABBREVIATIONS ═══
w/(with) w/o(without) b/c(because) re:(regarding) vs(versus) via per thru
@(at/location) #(number) &(and) +(plus/also) -(minus/without) /(per/or) |(or/pipe)
i.e.(that is) e.g.(example) etc.(and so on) cf(compare) viz(namely) NB(note well)

═══ ACTION SHORTHAND ═══
chk(check) calc(calculate) eval(evaluate) cmp(compare) est(estimate) approx(approximate)
find get set test run init proc(process) upd(update) del(delete) add sub mul div
verify confirm validate analyze extract parse transform merge split filter sort

═══ DOMAIN-SPECIFIC SHORTHAND ═══
- CODE/TECH: func var obj arr str int bool dict list async await req res API DB
  impl(implement) refactor debug deploy config exec cmd arg param ret val idx len

- BUSINESS: rev(revenue) exp(expense) proj(projection) KPI ROI Q1/Q2/Q3/Q4 YoY MoM
  stakeholder cust(customer) mkt(market) comp(competitor) strat(strategy) ops(operations)

- SCIENCE: exp(experiment) obs(observation) hyp(hypothesis) ctrl(control) var(variable)
  sig(significant) corr(correlation) data pt(point) meas(measure) temp pres vol mass

- LOGIC/REASONING: IF/THEN/ELSE WHEN/WHILE FOR/EACH CASE/SWITCH TRY/CATCH
  premise→conclusion assumption→inference cause→effect condition→result

═══ TIME & QUANTITY ═══
mins hrs days wks mos yrs NOW ASAP prev next cur(current) hist(historical)
approx ~100 <10 >50 ≤5 ≥20 between±5 range[1-10] max min avg sum total count

═══ COMPARISON & RELATIONSHIPS ═══
better/worse higher/lower more/less same≠diff equal>unequal similar≈different
vs opt1/opt2/opt3 pros/cons trade-off cost/benefit risk/reward

═══ STRICTLY FORBIDDEN PHRASES ═══
✗ "I think" "I believe" "I feel" "In my opinion" "It seems" "It appears"
✗ "Let me" "I should" "I need to" "I want to" "I'm going to"
✗ "This is interesting" "Looking at" "Considering" "Taking into account"
✗ "First of all" "On the other hand" "In this case" "As we can see"
✗ "It's worth noting" "It's important to" "We should consider"
✗ "Taking into account" "With that in mind" "Given this information" "Based on this"
✗ "Confused" "stuck" "frustrated" "Uncertain" "Unclear" "I'm guessing"
✗ "maybe the answer is" "I'm not sure" "Probably" "Perhaps" "Possibly"
✗ "Circular reasoning" "In some way" "Magically" "For some reason" "Too complicated" "It just works"
✗ "Something is off" "Wait, but" "Wait, maybe" "Wait, actually" "Hold on" "another thought:"
✗ "Alternatively", "Actually", "Or maybe", "Flowery language, hedging, or conversational filler"
✗ "Furthermore", "Moreover", "Nevertheless", "Nonetheless", "Subsequently", "Therefore, it can be concluded", "In conclusion", "To summarize", "As mentioned previously"
✗ Any emoji unless user explicitly requests them

═══ REQUIRED FORMAT ═══
- Write as compact telegraphic notes, NOT full sentences
- Use vertical lists w/ bullets or dashes for multi-items
- Group related info with indentation or symbols
- One idea per line when possible
- Omit articles (a/an/the), auxiliary verbs (is/are/was), obvious subjects

EXAMPLES:
❌ BAD: "I think we should first check if the value is greater than 10, and if it is, then we need to calculate..."
✓ GOOD: "chk val>10 → calc x²+3 → ∴ result≈42"

❌ BAD: "Looking at the data, it seems that the customer retention rate is lower than expected"
✓ GOOD: "data: cust retention<expected (est 65% vs target 80%) → need improve"

❌ BAD: "Let me break this down. We have three options here. Option A would cost more but..."
✓ GOOD: "3 opts: A(↑cost ✓quality) B(balanced) C(↓cost ✗quality) → rec: B"

═══ WHEN UNCERTAIN ═══ DO NOT guess or assume. Instead: ? = flag uncertainty w/ question mark ASK: "need clarification on X" or "X not specified - options: A/B/C?" CONSTRAINT: "cannot solve b/c: missing info Y" If problem unsolvable → state why concisely, don't elaborate Think like: debugger output, medical chart notes, trading floor shorthand, or military briefing. COMPRESS EVERYTHING. Every word must earn its place."""

# Default lists for reward configuration (moved to constants for clarity and reuse)
DEFAULT_SYMBOLIC_CHARS = [
    "∴",
    "∵",
    "⇒",
    "≈",
    "∈",
    "∀",
    "∃",
    "≠",
    "≤",
    "≥",
    "✓",
    "✗",
    "?",
    "!",
    "⚠",
    "∧",
    "∨",
    "¬",
    "⊕",
    "→",
    "←",
    "↔",
    "⇄",
    "▸",
    "◂",
    "⊃",
    "⊂",
    "○",
    "●",
    "◐",
    "⊗",
    "⊘",
    "@",
    "#",
    "&",
    "+",
    "-",
    "/",
    "|",
    "~",
    "<",
    ">",
    "±",
]

DEFAULT_ABBREVIATIONS = [
    "w/",
    "w/o",
    "b/c",
    "re:",
    "vs",
    "via",
    "per",
    "thru",
    "i.e.",
    "e.g.",
    "etc.",
    "cf",
    "viz",
    "NB",
    "chk",
    "calc",
    "eval",
    "cmp",
    "est",
    "approx",
    "find",
    "get",
    "set",
    "test",
    "run",
    "init",
    "proc",
    "upd",
    "del",
    "add",
    "sub",
    "mul",
    "div",
    "verify",
    "confirm",
    "validate",
    "analyze",
    "extract",
    "parse",
    "transform",
    "merge",
    "split",
    "filter",
    "sort",
    "func",
    "var",
    "obj",
    "arr",
    "str",
    "int",
    "bool",
    "dict",
    "list",
    "async",
    "await",
    "req",
    "res",
    "API",
    "DB",
    "impl",
    "refactor",
    "debug",
    "deploy",
    "config",
    "exec",
    "cmd",
    "arg",
    "param",
    "ret",
    "val",
    "idx",
    "len",
    "rev",
    "exp",
    "proj",
    "KPI",
    "ROI",
    "YoY",
    "MoM",
    "cust",
    "mkt",
    "comp",
    "strat",
    "ops",
    "obs",
    "hyp",
    "ctrl",
    "sig",
    "corr",
    "data",
    "pt",
    "meas",
    "temp",
    "pres",
    "vol",
    "mass",
    "IF",
    "THEN",
    "ELSE",
    "WHEN",
    "WHILE",
    "FOR",
    "EACH",
    "CASE",
    "SWITCH",
    "TRY",
    "CATCH",
    "mins",
    "hrs",
    "days",
    "wks",
    "mos",
    "yrs",
    "NOW",
    "ASAP",
    "prev",
    "next",
    "cur",
    "hist",
    "max",
    "min",
    "avg",
    "sum",
    "total",
    "count",
    "better",
    "worse",
    "higher",
    "lower",
    "more",
    "less",
    "same",
    "diff",
    "equal",
    "unequal",
    "similar",
    "different",
    "opt1",
    "opt2",
    "opt3",
    "pros",
    "cons",
    "trade-off",
    "cost",
    "benefit",
    "risk",
    "reward",
]

DEFAULT_BAN_KEYWORDS = [
    "I think",
    "I believe",
    "I feel",
    "In my opinion",
    "It seems",
    "It appears",
    "Let me",
    "I should",
    "I need to",
    "I want to",
    "I'm going to",
    "This is interesting",
    "Looking at",
    "Considering",
    "Taking into account",
    "First of all",
    "On the other hand",
    "In this case",
    "As we can see",
    "It's worth noting",
    "It's important to",
    "We should consider",
    "Taking into account",
    "With that in mind",
    "Given this information",
    "Based on this",
    "Confused",
    "stuck",
    "frustrated",
    "Uncertain",
    "Unclear",
    "I'm guessing",
    "maybe the answer is",
    "I'm not sure",
    "Probably",
    "Perhaps",
    "Possibly",
    "Circular reasoning",
    "In some way",
    "Magically",
    "For some reason",
    "Too complicated",
    "It just works",
    "Something is off",
    "Wait, but",
    "Wait, maybe",
    "Wait, actually",
    "Hold on",
    "another thought:",
    "Alternatively",
    "Actually",
    "Or maybe",
    "Flowery language, hedging, or conversational filler",
    "Furthermore",
    "Moreover",
    "Nevertheless",
    "Nonetheless",
    "Subsequently",
    "Therefore, it can be concluded",
    "In conclusion",
    "To summarize",
    "As mentioned previously",
]


class RewardConfig(BaseModel):
    """Configuration for a single reward function component."""

    name: str = Field(..., description="Registered name of the reward function.")
    weight: float = Field(
        1.0, ge=0.0, le=1.0, description="Weighting factor for this reward signal."
    )

    # Tags for content extraction are common across rewards, can be overridden here if needed
    think_start_tag: str = Field(
        "<think>", description="Think start tag for content extraction."
    )
    think_end_tag: str = Field(
        "</think>", description="Think end tag for content extraction."
    )
    answer_start_tag: str = Field(
        "<answer>", description="Answer start tag for content extraction."
    )
    answer_end_tag: str = Field(
        "</answer>", description="Answer end tag for content extraction."
    )

    # All other specific reward parameters go into the `config` dictionary
    # For a pluggable reward, its specific tunable parameters should be in here.
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Reward-specific parameters."
    )


class EvaluatorConfig(BaseModel):
    """Configuration for a single evaluation benchmark."""

    name: str = Field(..., description="Registered name of the evaluator.")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Evaluator-specific parameters."
    )


class GenerationConfig(BaseModel):
    """Configuration for text generation parameters during rollouts and evaluation."""

    max_tokens: PositiveInt = Field(
        512, description="Maximum number of tokens to generate."
    )
    temperature: NonNegativeFloat = Field(
        0.7, gt=0.0, le=2.0, description="Sampling temperature."
    )
    top_p: NonNegativeFloat = Field(
        0.9, gt=0.0, le=1.0, description="Top-p sampling probability."
    )
    top_k: int = Field(0, ge=0, description="Top-k sampling cutoff (0 to disable).")
    num_samples_per_prompt: PositiveInt = Field(
        4, description="Number of unique responses to generate per prompt for rollouts."
    )

    # These are internal dynamic fields for Metal safety, not directly exposed in YAML
    orig_max_gen_len: Optional[int] = Field(None, exclude=True)


class DataConfig(BaseModel):
    """Configuration for data loading and preprocessing."""

    train_path: Path = Field(
        ...,
        description="Path to training data (e.g., local JSONL or Hugging Face dataset name).",
    )
    val_path: Optional[Path] = Field(
        None, description="Path to validation data (optional)."
    )
    max_prompt_len: PositiveInt = Field(
        512,
        description="Maximum token length for input prompts. Prompts will be truncated if longer.",
    )
    max_gen_len: PositiveInt = Field(
        384, description="Maximum token length for generated responses from the model."
    )
    loader_type: Literal["jsonl", "hf_dataset", "mock"] = Field(
        "jsonl", description="Type of data loader to use."
    )
    shuffle_data: bool = Field(
        True,
        description="Whether to shuffle training data at the beginning of each epoch.",
    )

    dataset_prompt_key: str = Field(
        "prompt",
        description="Key in the raw dataset dictionary that holds the prompt text.",
    )
    dataset_answer_key: str = Field(
        "completion",
        description="Key in the raw dataset dictionary that holds the reference answer/completion.",
    )
    dataset_filter_keywords: List[str] = Field(
        default_factory=list,
        description="List of keywords (case-insensitive) to filter out samples from the dataset if found in prompt or completion.",
    )


class ModelConfig(BaseModel):
    """Configuration for actor and reference models, including LoRA parameters."""

    model_path: Path = Field(
        ...,
        description="Path to the actor model directory (MLX format) or Hugging Face model ID.",
    )
    ref_model_path: Optional[Path] = Field(
        None,
        description="Path to the reference model directory. Defaults to `model_path` if not specified.",
    )
    use_lora: bool = Field(
        False, description="Enable LoRA (Low-Rank Adaptation) fine-tuning."
    )
    lora_rank: PositiveInt = Field(8, description="LoRA adapter rank.")

    lora_alpha: float = Field(
        16.0, description="LoRA alpha parameter, scaling factor for LoRA weights."
    )
    lora_dropout: NonNegativeFloat = Field(
        0.0, le=1.0, description="LoRA dropout rate."
    )
    lora_scale_by_rank: bool = Field(
        True,
        description="Whether to scale LoRA weights by rank for better performance.",
    )
    lora_target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="List of module names (e.g., 'q_proj', 'mlp') to apply LoRA adapters to.",
    )

    @model_validator(mode="after")
    def set_default_ref_model_path(self) -> "ModelConfig":
        if self.ref_model_path is None:
            self.ref_model_path = self.model_path
        return self


class CheckpointConfig(BaseModel):
    """Configuration for checkpoint saving and loading."""

    save_dir: Path = Field(
        "checkpoints",
        description="Directory relative to `output_dir` to save checkpoints.",
    )
    save_every: PositiveInt = Field(
        100, description="Save a full checkpoint every N training updates."
    )
    keep_best_n: PositiveInt = Field(
        3,
        description="Number of most recent checkpoints to retain (older ones are deleted, excluding the 'best').",
    )
    save_optimizer_state: bool = Field(
        False,
        description="Whether to save the optimizer's state along with model weights (increases checkpoint size).",
    )


class MonitoringConfig(BaseModel):
    """Configuration for monitoring, logging, and visualization."""

    use_wandb: bool = Field(
        True,
        description="Enable Weights & Biases (W&B) logging for experiment tracking.",
    )
    wandb_project: Optional[str] = Field(
        "mlx-rl-trainer", description="W&B project name. Defaults to 'mlx-rl-trainer'."
    )
    wandb_entity: Optional[str] = Field(
        None, description="Your W&B entity (username or team name)."
    )
    wandb_run_name: Optional[str] = Field(
        None, description="Custom name for the W&B run."
    )
    log_samples_every: PositiveInt = Field(
        10,
        description="Log generated text samples to W&B and a local NDJSON file every N updates.",
    )
    max_logged_samples: PositiveInt = Field(
        5, description="Maximum number of generated samples to log per log event."
    )
    log_prompts: bool = Field(
        True,
        description="Include full input prompts in sample logs for easier debugging.",
    )
    sample_log_path: Optional[Path] = Field(
        None,
        description="Custom path to save NDJSON sample logs. Defaults to `output_dir/samples_debug_*.jsonl`.",
    )


class TrainerParams(BaseModel):
    """Core training loop parameters."""

    algorithm: Literal["grpo", "ppo"] = Field(
        ...,
        description="The Reinforcement Learning algorithm to use (e.g., 'grpo', 'ppo').",
    )
    output_dir: Path = Field(
        Path("./outputs"),
        description="Base directory for all training outputs (checkpoints, logs, plots).",
    )
    num_training_steps: PositiveInt = Field(
        10000,
        description="Total number of training steps (optimizer updates) to perform.",
    )
    learning_rate: NonNegativeFloat = Field(
        2e-6, description="Optimizer's initial learning rate."
    )
    ppo_batch_size: PositiveInt = Field(
        4,
        description="Number of unique prompts to process in a single micro-batch during rollout generation.",
    )
    num_rollout_samples: PositiveInt = Field(
        2,
        description="Number of distinct responses to generate per prompt for RL training.",
    )
    grad_accum_steps: PositiveInt = Field(
        4,
        description="Number of micro-batches to accumulate gradients over before performing an optimizer step.",
    )
    grpo_beta: NonNegativeFloat = Field(
        0.05,
        description="GRPO beta parameter, controlling the KL divergence penalty strength.",
    )
    seed: int = Field(42, description="Random seed for reproducibility across runs.")

    effective_batch_size: Optional[Any] = Field(None, exclude=True)

    # Dynamic Hyperparameters
    enable_dynamic_beta: bool = Field(
        False,
        description="Enable dynamic adjustment of GRPO beta based on training reward.",
    )
    high_reward_threshold: float = Field(
        0.85, description="Reward threshold to trigger an increase in beta."
    )
    beta_increase_high_reward: float = Field(
        0.08, description="Amount to increase beta by when high reward is detected."
    )
    cooldown_duration: PositiveInt = Field(
        2, description="Number of updates to hold the adjusted beta before reverting."
    )
    early_train_steps: PositiveInt = Field(
        32,
        description="Number of initial training steps with reduced generation length/KV size for stability.",
    )

    # Optimizer Parameters
    optimizer_beta1: NonNegativeFloat = Field(0.9, description="AdamW beta1 parameter.")
    optimizer_beta2: NonNegativeFloat = Field(
        0.95, description="AdamW beta2 parameter."
    )
    optimizer_weight_decay: NonNegativeFloat = Field(
        0.01, description="AdamW weight decay (L2 regularization)."
    )
    grad_clip_norm: Optional[NonNegativeFloat] = Field(
        0.5, description="Maximum gradient norm for clipping (0 or None to disable)."
    )
    save_optimizer_state: bool = Field(
        False, description="Save optimizer state with checkpoints."
    )

    # Learning Rate Schedule
    lr_schedule_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "name": "cosine_decay",
            "arguments": [
                2e-6,
                20000,
                2e-7,
            ],  # These will be dynamically overwritten by learning_rate, num_training_steps
            "warmup": 500,
            "warmup_init": 2e-7,
        },
        description="Configuration for the learning rate scheduler.",
    )

    # Gradient Checkpointing
    use_grad_checkpointing: bool = Field(
        False, description="Enable gradient checkpointing to save GPU memory."
    )
    grad_checkpoint_layers: PositiveInt = Field(
        0,
        description="Number of layers to apply gradient checkpointing to (0 for all applicable layers).",
    )

    # Gradient Manipulation (Banding)
    low_band: Tuple[int, int] = Field(
        (0, 15),
        description="Layer index range for low gradient multiplier band (inclusive).",
    )
    mid_band: Tuple[int, int] = Field(
        (16, 23),
        description="Layer index range for mid gradient multiplier band (inclusive).",
    )
    top_band: Tuple[int, int] = Field(
        (24, 35),
        description="Layer index range for top gradient multiplier band (inclusive).",
    )
    low_mul: NonNegativeFloat = Field(
        0.10, description="Gradient multiplier for layers in the low band."
    )
    mid_mul: NonNegativeFloat = Field(
        0.95, description="Gradient multiplier for layers in the mid band."
    )
    top_mul: NonNegativeFloat = Field(
        1.5, description="Gradient multiplier for layers in the top band."
    )
    head_mul: NonNegativeFloat = Field(
        1.2, description="Gradient multiplier for the LM head layer."
    )
    train_layer_start: int = Field(
        26,
        description="Starting layer index (inclusive) for which gradients are actively updated.",
    )
    train_layer_end: int = Field(
        35,
        description="Ending layer index (inclusive) for which gradients are actively updated.",
    )

    # Custom Invalid Sample Handling (Robustness Training)
    use_custom_batch_builder: bool = Field(
        False,
        description="Enable a custom batching strategy to include 'invalid' samples periodically.",
    )
    invalid_sample_layers: str = Field(
        "33,34,35",
        description="Comma-separated string of layer indices to target during 'invalid' sample updates.",
    )
    invalid_sample_frequency: PositiveInt = Field(
        2,
        description="Frequency (every N updates) to incorporate 'invalid' samples in a dual-update step.",
    )
    invalid_sample_layers_set: Set[int] = Field(
        default_factory=set, exclude=True
    )  # Populated dynamically in `model_validator`

    # Evaluation Frequency
    eval_every: PositiveInt = Field(
        250, description="Run full evaluation every N training updates."
    )
    benchmark_every: PositiveInt = Field(
        0, description="Run a quick benchmark every N updates (0 to disable)."
    )

    # Monitoring (specific fields from original TrainingArgs)
    reward_smoothing_window: PositiveInt = Field(
        20,
        description="Window size for calculating a rolling average of rewards for display.",
    )

    think_temperature: NonNegativeFloat = Field(
        0.23, description="Sampling temperature for thinking phase."
    )
    answer_temperature: NonNegativeFloat = Field(
        0.24, description="Sampling temperature for answer phase."
    )
    repetition_penalty: float = Field(
        1.15, ge=1.0, description="Repetition penalty factor."
    )
    repetition_context_size: PositiveInt = Field(
        20, description="Number of previous tokens to consider for repetition penalty."
    )

    @model_validator(mode="after")
    def populate_derived_fields(self) -> "TrainerParams":
        """
        Calculates derived fields and ensures consistency after initial parsing.
        """
        self.effective_batch_size = (
            self.ppo_batch_size * self.num_rollout_samples * self.grad_accum_steps
        )

        if isinstance(self.invalid_sample_layers, str):
            try:
                self.invalid_sample_layers_set = {
                    int(i.strip())
                    for i in self.invalid_sample_layers.split(",")
                    if i.strip()
                }
            except ValueError:
                logger.warning(
                    f"Invalid format for `invalid_sample_layers`: '{self.invalid_sample_layers}'. Using an empty set."
                )
                self.invalid_sample_layers_set = set()
        else:
            self.invalid_sample_layers_set = set()

        cfg = self.lr_schedule_config
        init_lr = float(self.learning_rate)
        total_steps = int(self.num_training_steps)
        warmup_steps = int(cfg.get("warmup", 500))
        decay_steps = max(total_steps - warmup_steps, 1)
        end_lr = max(init_lr * 0.1, 1e-07)

        cfg.setdefault("name", "cosine_decay")
        cfg["arguments"] = [init_lr, decay_steps, end_lr]
        cfg.setdefault("warmup", warmup_steps)
        cfg.setdefault("warmup_init", min(init_lr, max(init_lr * 0.1, 1e-08)))

        return self


class ExperimentConfig(BaseModel):
    """The complete configuration for an MLX RL training experiment."""

    model_config = ConfigDict(extra="forbid")

    trainer: TrainerParams
    model: ModelConfig
    generation: GenerationConfig = Field(
        default_factory=GenerationConfig, description="Parameters for text generation."
    )
    rewards: List[RewardConfig] = Field(
        default_factory=list, description="List of reward function configurations."
    )
    data: DataConfig
    evaluation: List[EvaluatorConfig] = Field(
        default_factory=list,
        description="List of evaluator configurations for benchmarks.",
    )
    checkpointing: CheckpointConfig = Field(
        default_factory=CheckpointConfig, description="Configuration for checkpointing."
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Configuration for monitoring and logging.",
    )

    # Global/derived fields that apply across multiple components
    max_kv_size: PositiveInt = Field(
        1536, description="Maximum KV cache size (in tokens) for model inference."
    )
    kv_bits: Optional[PositiveInt] = Field(
        None,
        description="Quantization bits for KV cache (e.g., 4, 8). None for disabled.",
    )
    kv_group_size: PositiveInt = Field(
        64, description="Group size for KV cache quantization."
    )
    quantized_kv_start: int = Field(
        10, description="Layer index to start KV cache quantization (inclusive)."
    )
    run_server: bool = Field(
        False,
        description="If true, the script will start an async inference server instead of training.",
    )
    use_paged_kv_cache: bool = Field(
        False,
        description="Enable the PagedKVCache for efficient memory management during rollouts/serving.",
    )
    kv_cache_block_size: PositiveInt = Field(
        16, description="Number of tokens per block in PagedKVCache."
    )
    kv_cache_num_blocks: PositiveInt = Field(
        2048, description="Total number of blocks to pre-allocate for PagedKVCache."
    )
    allow_cross_arch_ref: bool = Field(
        False,
        description="Allow using a reference model with a different architecture (requires an alignment bridge).",
    )
    align_bridge_path: Optional[Path] = Field(
        None,
        description="Path to a pre-trained alignment bridge model's weights if cross-arch ref is enabled.",
    )
    align_bridge_weight: NonNegativeFloat = Field(
        1.0,
        description="Weighting factor for the alignment bridge's reward/KL contribution.",
    )
    align_pool: Literal["mean", "last"] = Field(
        "mean",
        description="Pooling method for alignment bridge embeddings ('mean' or 'last').",
    )
    align_after_tag: str = Field(
        "</think>",
        description="The tag after which text should be extracted for alignment computation.",
    )

    # Benchmark specific settings
    benchmark_every: PositiveInt = Field(
        0,
        description="Run a quick benchmark evaluation every N updates (0 to disable).",
    )
    benchmark_dataset: str = Field(
        "gsm8k", description="Name of the Hugging Face dataset for quick benchmarking."
    )
    benchmark_dataset_config: Optional[str] = Field(
        "main", description="Configuration for the benchmark dataset."
    )
    benchmark_split: str = Field(
        "test",
        description="Dataset split to use for benchmarking (e.g., 'test', 'validation').",
    )
    benchmark_prompt_key: str = Field(
        "question", description="Key for prompt in benchmark dataset."
    )
    benchmark_answer_key: str = Field(
        "answer", description="Key for answer in benchmark dataset."
    )
    benchmark_samples: PositiveInt = Field(
        10, description="Number of samples to use for the quick benchmark."
    )
    benchmark_max_new_tokens: PositiveInt = Field(
        196, description="Maximum new tokens to generate for benchmark solutions."
    )
    benchmark_temperature: NonNegativeFloat = Field(
        0.0,
        description="Sampling temperature for benchmark generation (0.0 for greedy).",
    )
    benchmark_top_p: NonNegativeFloat = Field(
        1.0, description="Top-p sampling for benchmark generation."
    )
    benchmark_top_k: int = Field(
        0, description="Top-k sampling for benchmark generation (0 to disable)."
    )
    benchmark_use_chat_template: bool = Field(
        True, description="Whether to apply chat template for benchmark prompts."
    )
    benchmark_stop_on_error: bool = Field(
        False, description="Stop benchmark if any single sample causes an error."
    )

    system_prompt: str = Field(
        THINK_STYLE_PROMPT_LITERAL,
        description="The system prompt or thinking style guidance applied to all chat interactions.",
    )

    @model_validator(mode="after")
    def validate_reward_weights_sum(self) -> "ExperimentConfig":
        """Ensures that reward weights sum to approximately 1.0 if rewards are defined."""
        if self.rewards:
            total_weight = sum(reward.weight for reward in self.rewards)
            if not (0.99 <= total_weight <= 1.01):
                logger.warning(
                    f"Reward weights in configuration do not sum to 1.0 (got {total_weight:.2f}). This may lead to unexpected reward scaling. Consider normalizing them."
                )
        return self

    @classmethod
    def load_from_yaml(cls, path: Path) -> "ExperimentConfig":
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
            # Temporary workaround: Remove 'effective_batch_size' if it somehow appears in trainer config
            if (
                "trainer" in raw_config
                and "effective_batch_size" in raw_config["trainer"]
            ):
                del raw_config["trainer"]["effective_batch_size"]
            instance = cls(**raw_config)
            instance.trainer.output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Loaded and validated configuration from {path}.")
            return instance
        except ValidationError as e:
            console.print(
                f"[bold red]Configuration Validation Error in {path}:[/bold red]\n{e}"
            )
            raise ValueError(f"Invalid configuration in {path}.") from e
        except Exception as e:
            console.print(
                f"[bold red]Failed to load configuration from {path}:[/bold red] {e}"
            )
            raise ValueError(f"Failed to load configuration from {path}.") from e
