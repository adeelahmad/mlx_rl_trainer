"""
Configuration management system using Pydantic for validation and predictability.
"""
import logging
import re
import string
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

# --- Base THINK_STYLE_PROMPT as a module-level constant (from BEFORE_STATE) ---
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

DEFAULT_SYMBOLIC_CHARS = [
    "∴", "∵", "⇒", "≈", "∈", "∀", "∃", "≠", "≤", "≥", "✓", "✗", "?", "!", "⚠", "∧", "∨", "¬", "⊕", "→", "←", "↔", "⇄", "▸", "◂", "⊃", "⊂", "○", "●", "◐", "⊗", "⊘", "@", "#", "&", "+", "-", "/", "|", "~", "<", ">", "±",
]
DEFAULT_ABBREVIATIONS = [
    "w/", "w/o", "b/c", "re:", "vs", "via", "per", "thru", "i.e.", "e.g.", "etc.", "cf", "viz", "NB", "chk", "calc", "eval", "cmp", "est", "approx", "find", "get", "set", "test", "run", "init", "proc", "upd", "del", "add", "sub", "mul", "div", "verify", "confirm", "validate", "analyze", "extract", "parse", "transform", "merge", "split", "filter", "sort", "func", "var", "obj", "arr", "str", "int", "bool", "dict", "list", "async", "await", "req", "res", "API", "DB", "impl", "refactor", "debug", "deploy", "config", "exec", "cmd", "arg", "param", "ret", "val", "idx", "len", "rev", "exp", "proj", "KPI", "ROI", "YoY", "MoM", "cust", "mkt", "comp", "strat", "ops", "obs", "hyp", "ctrl", "sig", "corr", "data", "pt", "meas", "temp", "pres", "vol", "mass", "IF", "THEN", "ELSE", "WHEN", "WHILE", "FOR", "EACH", "CASE", "SWITCH", "TRY", "CATCH", "mins", "hrs", "days", "wks", "mos", "yrs", "NOW", "ASAP", "prev", "next", "cur", "hist", "max", "min", "avg", "sum", "total", "count", "better", "worse", "higher", "lower", "more", "less", "same", "diff", "equal", "unequal", "similar", "different", "opt1", "opt2", "opt3", "pros", "cons", "trade-off", "cost", "benefit", "risk", "reward",
]
DEFAULT_BAN_KEYWORDS = [
    "I think", "I believe", "I feel", "In my opinion", "It seems", "It appears", "Let me", "I should", "I need to", "I want to", "I'm going to", "This is interesting", "Looking at", "Considering", "Taking into account", "First of all", "On the other hand", "In this case", "As we can see", "It's worth noting", "It's important to", "We should consider", "Taking into account", "With that in mind", "Given this information", "Based on this", "Confused", "stuck", "frustrated", "Uncertain", "Unclear", "I'm guessing", "maybe the answer is", "I'm not sure", "Probably", "Perhaps", "Possibly", "Circular reasoning", "In some way", "Magically", "For some reason", "Too complicated", "It just works", "Something is off", "Wait, but", "Wait, maybe", "Wait, actually", "Hold on", "another thought:", "Alternatively", "Actually", "Or maybe", "Flowery language, hedging, or conversational filler", "Furthermore", "Moreover", "Nevertheless", "Nonetheless", "Subsequently", "Therefore, it can be concluded", "In conclusion", "To summarize", "As mentioned previously",
]

class RewardConfig(BaseModel):
    name: str = Field(..., description="Registered name of the reward function.")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="Weighting factor for this reward signal.")
    # Tags for content extraction are common across rewards, can be overridden here if needed (from BEFORE_STATE)
    think_start_tag: str = Field("<think>", description="Think start tag for content extraction.")
    think_end_tag: str = Field("</think>", description="Think end tag for content extraction.")
    answer_start_tag: str = Field("<answer>", description="Answer start tag for content extraction.")
    answer_end_tag: str = Field("</answer>", description="Answer end tag for content extraction.")
    config: Dict[str, Any] = Field(default_factory=dict, description="Reward-specific parameters.")

class EvaluatorConfig(BaseModel):
    name: str = Field(..., description="Registered name of the evaluator.")
    config: Dict[str, Any] = Field(default_factory=dict, description="Evaluator-specific parameters.")

class DataConfig(BaseModel):
    train_path: Path = Field(..., description="Path to training data (e.g., local JSONL or Hugging Face dataset name).")
    val_path: Optional[Path] = Field(None, description="Path to validation data (optional).")
    max_prompt_len: PositiveInt = Field(350, description="Maximum token length for input prompts. Prompts will be truncated if longer.") # From BEFORE_STATE: 350
    max_gen_len: PositiveInt = Field(96, description="Maximum token length for generated responses from the model.") # From BEFORE_STATE: 96
    loader_type: Literal["jsonl", "hf_dataset", "mock"] = Field("jsonl", description="Type of data loader to use.")
    shuffle_data: bool = Field(True, description="Whether to shuffle training data at the beginning of each epoch.")
    dataset_prompt_key: str = Field("prompt", description="Key in the raw dataset dictionary that holds the prompt text.")
    dataset_answer_key: str = Field("completion", description="Key in the raw dataset dictionary that holds the reference answer/completion.")
    dataset_filter_keywords: List[str] = Field(default_factory=lambda: ['http://', '**other**', 'https://', 'png', 'jpg', 'Another way', 'Adeel'], description="List of keywords (case-insensitive) to filter out samples from the dataset if found in prompt or completion.") # From BEFORE_STATE

class ModelConfig(BaseModel):
    model_path: Path = Field(..., description="Path to the actor model directory (MLX format) or Hugging Face model ID.")
    ref_model_path: Optional[Path] = Field(None, description="Path to the reference model directory. Defaults to `model_path` if not specified.")
    use_lora: bool = Field(False, description="Enable LoRA (Low-Rank Adaptation) fine-tuning.")
    lora_rank: PositiveInt = Field(8, description="LoRA adapter rank.")
    lora_alpha: float = Field(16.0, description="LoRA alpha parameter, scaling factor for LoRA weights.")
    lora_dropout: NonNegativeFloat = Field(0.0, le=1.0, description="LoRA dropout rate.")
    lora_scale_by_rank: bool = Field(True, description="Whether to scale LoRA weights by rank for better performance.")
    lora_target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], description="List of module names (e.g., 'q_proj', 'mlp') to apply LoRA adapters to.")

    @model_validator(mode="after")
    def set_default_ref_model_path(self) -> "ModelConfig":
        if self.ref_model_path is None:
            self.ref_model_path = self.model_path
        return self

class CheckpointConfig(BaseModel):
    save_dir: Path = Field("checkpoints", description="Directory relative to `output_dir` to save checkpoints.")
    save_every: PositiveInt = Field(100, description="Save a full checkpoint every N training updates.") # From BEFORE_STATE: 10
    keep_last_n: PositiveInt = Field(3, description="Number of most recent checkpoints to retain (older ones are deleted, excluding the 'best').")
    save_optimizer_state: bool = Field(False, description="Whether to save the optimizer's state along with model weights (increases checkpoint size).")

class MonitoringConfig(BaseModel):
    use_wandb: bool = Field(True, description="Enable Weights & Biases (W&B) logging for experiment tracking.")
    wandb_project: Optional[str] = Field("mlx-grpo", description="W&B project name. Defaults to 'mlx-grpo'.") # From BEFORE_STATE: default mlx-grpo
    wandb_entity: Optional[str] = Field(None, description="Your W&B entity (username or team name).")
    wandb_run_name: Optional[str] = Field(None, description="Custom name for the W&B run.")
    log_samples_every: PositiveInt = Field(10, description="Log generated text samples to W&B and a local NDJSON file every N updates.") # From BEFORE_STATE: 1
    max_logged_samples: PositiveInt = Field(5, description="Maximum number of generated samples to log per log event.") # From BEFORE_STATE: 50
    log_prompts: bool = Field(True, description="Include full input prompts in sample logs for easier debugging.")
    sample_log_path: Optional[Path] = Field(None, description="Custom path to save NDJSON sample logs. Defaults to `output_dir/samples_debug_*.jsonl`.")

class GenerationConfig(BaseModel):
    # Generation parameters (from BEFORE_STATE's TrainingArgs)
    think_start_tag: str = Field("<think>")
    think_end_tag: str = Field("</think>")
    answer_start_tag: str = Field("<answer>")
    answer_end_tag: str = Field("</answer>")
    think_boost_tokens: int = Field(32)
    think_temperature: NonNegativeFloat = Field(0.23)
    answer_temperature: NonNegativeFloat = Field(0.24)
    sampling_top_p: NonNegativeFloat = Field(0.7)
    sampling_min_p: NonNegativeFloat = Field(0.02)
    sampling_top_k: int = Field(20)
    repetition_penalty: Optional[float] = Field(1.15)
    repetition_context_size: Optional[int] = Field(20)

    # Dynamic Bias Controls
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
    ban_phrases_for_bias: List[str] = Field(default_factory=lambda: ['I think the answer', 'I believe that', 'In my view', 'From what I can tell', 'It seems to me', 'It appears that', 'My understanding is', 'As far as I know', 'Let me start by', 'Let me first', 'I should probably', 'I need to figure out', "I'm trying to", "I'm going to try", "I'll attempt to", 'Confused', 'stuck', 'frustrated', 'frustrating', 'Alternatively', 'Actually', 'Probably not sure', 'Uncertain about', 'Unclear whether', "I'm guessing that", 'maybe this is', 'Could be that', 'Might be because', "I'm not 100% sure", "I'm not sure if", "I'm not certain", 'Hard to say', 'Difficult to tell', 'Circular reasoning detected', 'In some way or another', 'Magically works', 'For some unknown reason', 'Too complicated', 'It just somehow', 'Something seems off', 'False assumption', 'Insufficient information to', 'Wait, what if', 'Wait, actually no', 'Wait, on second thought', 'Hold on, maybe', 'Hmm, perhaps', 'Or wait, could', 'Looking at this more closely', 'Upon further reflection', 'Taking a step back', 'Thinking about it more', 'Now that I consider', 'When I really think', 'If I had to guess', 'To be completely honest', 'In all honesty', 'You know what', 'The thing is', 'What I mean is', 'In other words', 'Put simply', 'Basically what happens', 'Long story short', 'At the end of the day'])
    encourage_phrases_for_bias: List[str] = Field(default_factory=list) # From BEFORE_STATE, default empty. Will populate dynamically.
    encourage_think_bias: float = Field(4.5)
    ban_think_bias: float = Field(-3.0)

    # Tool Use Configuration
    allow_tool_calls: bool = Field(True)
    tool_call_penalty: NonNegativeFloat = Field(0.0)

class TrainerParams(BaseModel):
    algorithm: Literal["grpo", "ppo"] = Field("grpo", description="The Reinforcement Learning algorithm to use (e.g., 'grpo', 'ppo').")
    output_dir: Path = Field(Path("./outputs"), description="Base directory for all training outputs (checkpoints, logs, plots).")
    num_training_steps: PositiveInt = Field(45869, description="Total number of training steps (optimizer updates) to perform.") # From BEFORE_STATE: 45869
    learning_rate: NonNegativeFloat = Field(2e-6, description="Optimizer's initial learning rate.")
    ppo_batch_size: PositiveInt = Field(1, description="Number of unique prompts to process in a single micro-batch during rollout generation.") # From BEFORE_STATE: 1
    num_rollout_samples: PositiveInt = Field(2, description="Number of distinct responses to generate per prompt for RL training.") # From BEFORE_STATE: 2
    grad_accum_steps: PositiveInt = Field(2, description="Number of micro-batches to accumulate gradients over before performing an optimizer step.") # From BEFORE_STATE: 2
    grpo_beta: NonNegativeFloat = Field(0.025, description="GRPO beta parameter, controlling the KL divergence penalty strength.") # From BEFORE_STATE: 0.025
    seed: int = Field(432, description="Random seed for reproducibility across runs.") # From BEFORE_STATE: 432

    # Optimizer Parameters
    optimizer_beta1: NonNegativeFloat = Field(0.9, description="AdamW beta1 parameter.")
    optimizer_beta2: NonNegativeFloat = Field(0.95, description="AdamW beta2 parameter.")
    optimizer_weight_decay: NonNegativeFloat = Field(0.01, description="AdamW weight decay (L2 regularization).")
    grad_clip_norm: Optional[NonNegativeFloat] = Field(0.5, description="Maximum gradient norm for clipping (0 or None to disable).")

    # Learning Rate Schedule
    lr_schedule_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration for the learning rate scheduler.")

    # Gradient Checkpointing
    use_grad_checkpointing: bool = Field(False, description="Enable gradient checkpointing to save GPU memory.")
    grad_checkpoint_layers: PositiveInt = Field(0, description="Number of layers to apply gradient checkpointing to (0 for all applicable layers).")

    # Gradient Manipulation (Banding)
    low_band: Tuple[int, int] = Field((0, 15), description="Layer index range for low gradient multiplier band (inclusive).")
    mid_band: Tuple[int, int] = Field((16, 23), description="Layer index range for mid gradient multiplier band (inclusive).")
    top_band: Tuple[int, int] = Field((24, 35), description="Layer index range for top gradient multiplier band (inclusive).")
    low_mul: NonNegativeFloat = Field(0.1, description="Gradient multiplier for layers in the low band.") # From BEFORE_STATE: .1
    mid_mul: NonNegativeFloat = Field(0.95, description="Gradient multiplier for layers in the mid band.") # From BEFORE_STATE: .95
    top_mul: NonNegativeFloat = Field(1.5, description="Gradient multiplier for layers in the top band.") # From BEFORE_STATE: 1.5
    head_mul: NonNegativeFloat = Field(1.2, description="Gradient multiplier for the LM head layer.") # From BEFORE_STATE: 1.2
    train_layer_start: Optional[int] = Field(26)
    train_layer_end: Optional[int] = Field(35)
    
    # Custom Invalid Sample Handling (from BEFORE_STATE)
    use_custom_batch_builder: bool = Field(False)
    invalid_sample_layers: str = Field("33,34,35")
    invalid_sample_frequency: PositiveInt = Field(2)
    invalid_sample_layers_set: Set[int] = Field(default_factory=set, exclude=True)

    # Evaluation Frequency
    eval_every: PositiveInt = Field(10) # From BEFORE_STATE: 10
    reward_smoothing_window: PositiveInt = Field(20) # From BEFORE_STATE
    
    @model_validator(mode="after")
    def populate_derived_fields(self) -> "TrainerParams":
        self.effective_batch_size = self.ppo_batch_size * self.num_rollout_samples * self.grad_accum_steps
        
        if isinstance(self.invalid_sample_layers, str):
            try: self.invalid_sample_layers_set = {int(i.strip()) for i in self.invalid_sample_layers.split(",") if i.strip()}
            except ValueError: self.invalid_sample_layers_set = set() # Fallback to empty set
        
        # Auto-populate LR schedule defaults if not fully defined (from BEFORE_STATE logic)
        cfg = self.lr_schedule_config
        init_lr = float(self.learning_rate)
        total_steps = int(self.num_training_steps)
        warmup_steps = int(cfg.get("warmup", 500)) # Default from BEFORE_STATE: was 16 initially, but 500 later in example
        decay_steps = max(total_steps - warmup_steps, 1)
        end_lr = max(init_lr * 0.1, 1e-07)

        cfg.setdefault("name", "cosine_decay")
        cfg.setdefault("arguments", [init_lr, decay_steps, end_lr])
        cfg.setdefault("warmup", warmup_steps)
        cfg.setdefault("warmup_init", min(init_lr, max(init_lr * 0.1, 1e-08)))
        
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid") # Forbid extra fields

    trainer: TrainerParams
    model: ModelConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig, description="Parameters for text generation.")
    rewards: List[RewardConfig] = Field(default_factory=list, description="List of reward function configurations.")
    data: DataConfig
    evaluation: List[EvaluatorConfig] = Field(default_factory=list, description="List of evaluator configurations for benchmarks.")
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig, description="Configuration for checkpointing.")
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="Configuration for monitoring and logging.")

    # Global/overall config parameters
    max_kv_size: PositiveInt = Field(1536, description="Maximum KV cache size (in tokens) for model inference.")
    system_prompt: str = Field(THINK_STYLE_PROMPT_LITERAL, description="The system prompt or thinking style guidance applied to all chat interactions.")

    # Paged KV Cache (from BEFORE_STATE)
    use_paged_kv_cache: bool = Field(True, description="Enable the PagedKVCache.")
    kv_cache_block_size: PositiveInt = Field(16)
    kv_cache_num_blocks: PositiveInt = Field(2048)

    # Cross-architecture alignment (from BEFORE_STATE)
    allow_cross_arch_ref: bool = Field(False)
    align_bridge_path: Optional[Path] = Field(None)
    align_bridge_weight: NonNegativeFloat = Field(1.0)
    align_pool: Literal["mean", "last"] = Field("mean")
    align_after_tag: str = Field("</think>") # From BEFORE_STATE `align_after_tag`
    
    @model_validator(mode="after")
    def validate_reward_weights_sum(self) -> "ExperimentConfig":
        if self.rewards:
            total_weight = sum(reward.weight for reward in self.rewards)
            if not (0.99 <= total_weight <= 1.01):
                logger.warning(f"Reward weights in configuration do not sum to 1.0 (got {total_weight:.2f}).")
        return self

    @classmethod
    def load_from_yaml(cls, path: Path) -> "ExperimentConfig":
        if not path.exists(): raise FileNotFoundError(f"Config file not found: {path}")
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
