"""
Configuration management system using Pydantic for validation and predictability.
"""
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Literal, Union

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
    ✗ "Let me" "I should" "I need to" "I want to" "I\'m going to"
    ✗ "This is interesting" "Looking at" "Considering" "Taking into account"
    ✗ "First of all" "On the other hand" "In this case" "As we can see"
    ✗ "It\'s worth noting" "It\'s important to" "We should consider"
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

    ═══ WHEN UNCERTAIN ═══ DO NOT guess or assume. Instead: ? = flag uncertainty w/ question mark ASK: "need clarification on X" or "X not specified - options: A/B/C?" CONSTRAINT: "cannot solve b/c: missing info Y" If problem unsolvable → state why concisely, don\'t elaborate Think like: debugger output, medical chart notes, trading floor shorthand, or military briefing. COMPRESS EVERYTHING. Every word must earn its place."""


class RewardConfig(BaseModel):
    name: str = Field(..., description="Registered name of the reward function.")
    weight: float = Field(
        1.0, ge=0.0, le=1.0, description="Weighting factor for this reward signal."
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Reward-specific parameters."
    )


class EvaluatorConfig(BaseModel):
    name: str = Field(..., description="Registered name of the evaluator.")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Evaluator-specific parameters."
    )


class DataConfig(BaseModel):
    train_path: Path = Field(..., description="Path to training data.")
    val_path: Optional[Path] = Field(None, description="Path to validation data.")
    max_prompt_len: PositiveInt = Field(
        350, description="Maximum token length for input prompts."
    )
    max_gen_len: PositiveInt = Field(
        96, description="Maximum token length for generated responses."
    )
    loader_type: Literal["jsonl", "hf_dataset", "mock"] = Field(
        "jsonl", description="Type of data loader to use."
    )
    shuffle_data: bool = Field(True, description="Whether to shuffle training data.")
    dataset_prompt_key: str = Field("prompt", description="Key for prompt text.")
    dataset_answer_key: str = Field(
        "completion", description="Key for reference answer/completion."
    )
    dataset_filter_keywords: List[str] = Field(
        default_factory=lambda: [
            "http://",
            "**other**",
            "https://",
            "png",
            "jpg",
            "Another way",
            "Adeel",
        ],
        description="Keywords to filter out samples.",
    )


class ModelConfig(BaseModel):
    model_path: Path = Field(..., description="Path to the actor model directory.")
    ref_model_path: Optional[Path] = Field(
        None, description="Path to the reference model directory."
    )
    use_lora: bool = Field(False, description="Enable LoRA fine-tuning.")
    lora_rank: PositiveInt = Field(8, description="LoRA adapter rank.")
    lora_alpha: float = Field(16.0, description="LoRA alpha parameter.")
    lora_dropout: NonNegativeFloat = Field(
        0.0, le=1.0, description="LoRA dropout rate."
    )
    lora_scale_by_rank: bool = Field(
        True, description="Whether to scale LoRA weights by rank."
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
        description="Modules to apply LoRA to.",
    )

    @model_validator(mode="after")
    def set_default_ref_model_path(self) -> "ModelConfig":
        if self.ref_model_path is None:
            self.ref_model_path = self.model_path
        return self


class CheckpointConfig(BaseModel):
    save_dir: Path = Field(
        "./checkpoints", description="Directory relative to  to save checkpoints."
    )
    save_every: PositiveInt = Field(
        10, description="Save a full checkpoint every N training updates."
    )
    keep_last_n: PositiveInt = Field(
        3, description="Number of most recent checkpoints to retain."
    )
    save_optimizer_state: bool = Field(
        False, description="Whether to save the optimizer's state."
    )


class MonitoringConfig(BaseModel):
    use_wandb: bool = Field(True, description="Enable Weights & Biases (W&B) logging.")
    wandb_project: Optional[str] = Field("mlx-grpo", description="W&B project name.")
    wandb_entity: Optional[str] = Field(
        None, description="Your W&B entity (username or team name)."
    )
    wandb_run_name: Optional[str] = Field(
        None, description="Custom name for the W&B run."
    )
    log_samples_every: PositiveInt = Field(
        1, description="Log generated text samples every N updates."
    )
    max_logged_samples: PositiveInt = Field(
        50, description="Maximum number of generated samples to log per event."
    )
    log_prompts: bool = Field(
        True, description="Include full input prompts in sample logs."
    )
    sample_log_path: Optional[Path] = Field(
        None, description="Custom path to save NDJSON sample logs."
    )


class GenerationConfig(BaseModel):
    # Tags & Format
    think_start_tag: str = Field("<think>")
    think_end_tag: str = Field("</think>")
    answer_start_tag: str = Field("<answer>")
    answer_end_tag: str = Field("</answer>")

    # Sampling parameters
    think_boost_tokens: int = Field(32)
    think_temperature: NonNegativeFloat = Field(0.23)
    answer_temperature: NonNegativeFloat = Field(0.24)
    sampling_top_p: NonNegativeFloat = Field(0.7)
    sampling_min_p: NonNegativeFloat = Field(0.02)
    sampling_top_k: int = Field(20)
    repetition_penalty: Optional[float] = Field(1.15)
    repetition_context_size: Optional[int] = Field(20)

    # Dynamic Bias Controls (from BEFORE_STATE)
    min_think_tokens: int = Field(32)
    think_end_early_bias: float = Field(-12.0)
    bias_answer_start_after_min_think: bool = Field(True)
    bias_close_think: float = Field(9.0)
    bias_answer_start: float = Field(6.0)
    punish_extra_think_end: float = Field(-12.0)
    punish_reopen_think: float = Field(-10.0)
    punish_reopen_answer: float = Field(-9.0)
    bias_eos_after_answer: float = Field(3.0)

    # MCQ Specific Biases
    hard_mask_mcq_first_token: bool = Field(True)
    mcq_letter_lift: float = Field(8.0)
    mcq_ban_first_bias: float = Field(-14.0)
    nonmcq_ban_first_bias: float = Field(-12.0)
    mcq_close_after_k: int = Field(1)
    min_answer_tokens: int = Field(8)
    min_answer_tokens_mcq: int = Field(1)
    mcq_answer_end_bias: float = Field(9.0)

    # Verbosity Biasing for Rollouts
    ban_phrases_for_bias: List[str] = Field(
        default_factory=lambda: [
            "I think the answer",
            "I believe that",
            "Confused",
            "stuck",
            "frustrated",
        ]
    )
    encourage_phrases_for_bias: List[str] = Field(
        default_factory=lambda: ["chk", "calc", "∴", "w/"]
    )
    encourage_think_bias: float = Field(4.5)
    ban_think_bias: float = Field(-3.0)

    # Tool Use Configuration
    allow_tool_calls: bool = Field(True)
    tool_call_penalty: NonNegativeFloat = Field(0.0)

    # Think Length Penalty Config (used by Reward logic)
    think_length_target_min: PositiveInt = Field(32)
    think_length_target_max: PositiveInt = Field(128)
    think_length_penalty_strength: NonNegativeFloat = Field(0.15)


class TrainerParams(BaseModel):
    algorithm: Literal["grpo", "ppo"] = Field("grpo")
    output_dir: Path = Field(Path("./outputs"))
    num_training_steps: PositiveInt = Field(45869)
    learning_rate: NonNegativeFloat = Field(2e-6)
    ppo_batch_size: PositiveInt = Field(1)
    num_rollout_samples: PositiveInt = Field(2)
    grad_accum_steps: PositiveInt = Field(2)
    grpo_beta: NonNegativeFloat = Field(0.025)
    seed: int = Field(432)

    # Optimizer Parameters
    optimizer_beta1: NonNegativeFloat = Field(0.9)
    optimizer_beta2: NonNegativeFloat = Field(0.95)
    optimizer_weight_decay: NonNegativeFloat = Field(0.01)
    grad_clip_norm: Optional[NonNegativeFloat] = Field(0.5)

    # Learning Rate Schedule
    lr_schedule_config: Dict[str, Any] = Field(default_factory=dict)

    # Gradient Control
    use_grad_checkpointing: bool = Field(False)
    grad_checkpoint_layers: PositiveInt = Field(0)
    low_band: Tuple[int, int] = Field((0, 15))
    mid_band: Tuple[int, int] = Field((16, 23))
    top_band: Tuple[int, int] = Field((24, 35))
    low_mul: NonNegativeFloat = Field(0.1)
    mid_mul: NonNegativeFloat = Field(0.95)
    top_mul: NonNegativeFloat = Field(1.5)
    head_mul: NonNegativeFloat = Field(1.2)
    train_layer_start: Optional[int] = Field(26)
    train_layer_end: Optional[int] = Field(35)

    # Custom Invalid Sample Handling
    use_custom_batch_builder: bool = Field(False)
    invalid_sample_layers: str = Field("33,34,35")
    invalid_sample_frequency: PositiveInt = Field(2)
    invalid_sample_layers_set: Set[int] = Field(default_factory=set, exclude=True)

    # Evaluation Frequency
    eval_every: PositiveInt = Field(10)
    reward_smoothing_window: PositiveInt = Field(20)

    # Add the missing field
    effective_batch_size: int = Field(0, exclude=True)

    @model_validator(mode="after")
    def populate_derived_fields(self) -> "TrainerParams":
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
                self.invalid_sample_layers_set = set()

        cfg = self.lr_schedule_config
        init_lr = float(self.learning_rate)
        total_steps = int(self.num_training_steps)
        warmup_steps = int(cfg.get("warmup", 500))
        decay_steps = max(total_steps - warmup_steps, 1)
        end_lr = max(init_lr * 0.1, 1e-07)
        cfg.setdefault("name", "cosine_decay")
        cfg.setdefault("arguments", [init_lr, decay_steps, end_lr])
        cfg.setdefault("warmup", warmup_steps)
        cfg.setdefault("warmup_init", min(init_lr, max(init_lr * 0.1, 1e-08)))
        return self


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

    max_kv_size: PositiveInt = Field(1536)
    system_prompt: str = Field(THINK_STYLE_PROMPT_LITERAL)

    use_paged_kv_cache: bool = Field(True)
    kv_cache_block_size: PositiveInt = Field(16)
    kv_cache_num_blocks: PositiveInt = Field(2048)

    allow_cross_arch_ref: bool = Field(False)
    align_bridge_path: Optional[Path] = Field(None)
    align_bridge_weight: NonNegativeFloat = Field(1.0)
    align_pool: Literal["mean", "last"] = Field("mean")
    align_after_tag: str = Field("</think>")

    @model_validator(mode="after")
    def validate_reward_weights_sum(self) -> "ExperimentConfig":
        if self.rewards:
            total_weight = sum(reward.weight for reward in self.rewards)
            if not (0.99 <= total_weight <= 1.01):
                logger.warning(
                    f"Reward weights in configuration do not sum to 1.0 (got {total_weight:.2f})."
                )
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
            console.print(
                f"[bold red]Configuration Validation Error in {path}:[/bold red]\n{e}"
            )
            raise ValueError(f"Invalid configuration in {path}.") from e
        except Exception as e:
            console.print(
                f"[bold red]Failed to load configuration from {path}:[/bold red] {e}"
            )
            raise ValueError(f"Failed to load configuration from {path}.") from e
