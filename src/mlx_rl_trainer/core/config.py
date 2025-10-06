# file_path: mlx_rl_trainer/src/mlx_rl_trainer/core/config.py
# revision_no: 004
# goals_of_writing_code_block: Define Pydantic models for configuration, fixing bugs related to post-init validation and dynamic fields.
# type_of_code_response: change existing
"""
Configuration management system using Pydantic for validation and predictability.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import yaml
import logging
from pydantic import BaseModel, Field, PositiveInt, NonNegativeFloat, ValidationError, model_validator
from rich.console import Console

console = Console()

# FIX: Define THINK_STYLE_PROMPT before TrainingArgs
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
    approx ~100 <10 >50 ≤5 ≥20 between±5 range[1-10] max min avg sum total total count

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
    ✗ "Alternatively" "Actually" "Or maybe" "Flowery language, hedging, or conversational filler"
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
    "∴", "∵", "⇒", "≈", "∈", "∀", "∃", "≠", "≤", "≥",
    "✓", "✗", "?", "!", "⚠", "∧", "∨", "¬", "⊕",
    "→", "←", "↔", "⇄", "▸", "◂", "⊃", "⊂",
    "○", "●", "◐", "⊗", "⊘",
    "@", "#", "&", "+", "-", "/", "|", "~", "<", ">", "±"
]

DEFAULT_ABBREVIATIONS = [
    "w/", "w/o", "b/c", "re:", "vs", "via", "per", "thru", "i.e.", "e.g.", "etc.", "cf", "viz", "NB",
    "chk", "calc", "eval", "cmp", "est", "approx", "find", "get", "set", "test", "run", "init", "proc", "upd", "del", "add", "sub", "mul", "div",
    "verify", "confirm", "validate", "analyze", "extract", "parse", "transform", "merge", "split", "filter", "sort",
    "func", "var", "obj", "arr", "str", "int", "bool", "dict", "list", "async", "await", "req", "res", "API", "DB",
    "impl", "refactor", "debug", "deploy", "config", "exec", "cmd", "arg", "param", "ret", "val", "idx", "len",
    "rev", "exp", "proj", "KPI", "ROI", "YoY", "MoM", "cust", "mkt", "comp", "strat", "ops",
    "obs", "hyp", "ctrl", "sig", "corr", "data", "pt", "meas", "temp", "pres", "vol", "mass",
    "IF", "THEN", "ELSE", "WHEN", "WHILE", "FOR", "EACH", "CASE", "SWITCH", "TRY", "CATCH",
    "mins", "hrs", "days", "wks", "mos", "yrs", "NOW", "ASAP", "prev", "next", "cur", "hist",
    "max", "min", "avg", "sum", "total", "count",
    "better", "worse", "higher", "lower", "more", "less", "same", "diff", "equal", "unequal", "similar", "different",
    "opt1", "opt2", "opt3", "pros", "cons", "trade-off", "cost", "benefit", "risk", "reward"
]

DEFAULT_BAN_KEYWORDS = [
    "I think", "I believe", "I feel", "In my opinion", "It seems", "It appears",
    "Let me", "I should", "I need to", "I want to", "I'm going to",
    "This is interesting", "Looking at", "Considering", "Taking into account",
    "First of all", "On the other hand", "In this case", "As we can see",
    "It's worth noting", "It's important to", "We should consider",
    "Taking into account", "With that in mind", "Given this information", "Based on this",
    "Confused", "stuck", "frustrated", "Uncertain", "Unclear", "I'm guessing",
    "maybe the answer is", "I'm not sure", "Probably", "Perhaps", "Possibly",
    "Circular reasoning", "In some way", "Magically", "For some reason", "Too complicated", "It just works",
    "Something is off", "Wait, but", "Wait, maybe", "Wait, actually", "Hold on", "another thought:"
    "Alternatively", "Actually", "Or maybe", "Flowery language, hedging, or conversational filler",
    "Furthermore", "Moreover", "Nevertheless", "Nonetheless", "Subsequently", "Therefore, it can be concluded", "In conclusion", "To summarize", "As mentioned previously"
]


class RewardConfig(BaseModel):
    name: str = Field(..., description="Registered name of the reward function.")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="Weighting factor for this reward signal.")

    # Tags can be configured here, overriding global defaults if needed
    think_start_tag: str = Field("<think>", description="Think start tag.")
    think_end_tag: str = Field("</think>", description="Think end tag.")
    answer_start_tag: str = Field("<answer>", description="Answer start tag.")
    answer_end_tag: str = Field("</answer>", description="Answer end tag.")

    # All other attributes from original TrainingArgs for RewardConfig are now part of the `config` sub-dictionary
    # or moved to TrainerParams if they are global training parameters
    config: Dict[str, Any] = Field(default_factory=dict, description="Reward-specific parameters.")

class EvaluatorConfig(BaseModel):
    name: str = Field(..., description="Registered name of the evaluator.")
    config: Dict[str, Any] = Field(default_factory=dict, description="Evaluator-specific parameters.")

class GenerationConfig(BaseModel):
    """Generation parameters configuration"""
    max_tokens: PositiveInt = Field(512, description="Maximum number of tokens to generate.")
    temperature: NonNegativeFloat = Field(0.7, gt=0.0, le=2.0, description="Sampling temperature.")
    top_p: NonNegativeFloat = Field(0.9, gt=0.0, le=1.0, description="Top-p sampling probability.")
    top_k: int = Field(0, ge=0, description="Top-k sampling cutoff (0 to disable).")
    num_samples_per_prompt: PositiveInt = Field(4, description="Number of unique responses to generate per prompt for rollouts.")

    # Dynamic fields for Metal safety (used internally, not directly configurable in YAML)
    orig_max_gen_len: Optional[int] = Field(None, exclude=True)


class DataConfig(BaseModel):
    train_path: Path = Field(..., description="Path to training data.")
    val_path: Optional[Path] = Field(None, description="Path to validation data.")
    max_prompt_len: PositiveInt = Field(512, description="Maximum length for input prompts.")
    max_gen_len: PositiveInt = Field(384, description="Maximum length for generated responses.")
    loader_type: str = Field("jsonl", description="Type of data loader (e.g., 'jsonl', 'hf').")
    shuffle_data: bool = Field(True, description="Whether to shuffle training data each epoch.")

    dataset_prompt_key: str = Field("prompt", description="Key in dataset dictionary for prompt text.")
    dataset_answer_key: str = Field("completion", description="Key in dataset dictionary for reference answer text.")
    dataset_filter_keywords: List[str] = Field(default_factory=list, description="List of keywords to filter out dataset rows.")


class ModelConfig(BaseModel):
    model_path: Path = Field(..., description="Path to the actor model directory or HuggingFace ID.")
    ref_model_path: Optional[Path] = Field(None, description="Path to the reference model directory or HuggingFace ID. Defaults to model_path.")
    use_lora: bool = Field(False, description="Enable LoRA training.")
    lora_rank: PositiveInt = Field(8, description="LoRA adapter rank.")

    lora_alpha: float = Field(16.0, description="LoRA alpha parameter.")
    lora_dropout: NonNegativeFloat = Field(0.0, description="LoRA dropout rate.")
    lora_scale_by_rank: bool = Field(True, description="Whether to scale LoRA weights by rank.")
    lora_target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], description="Modules to apply LoRA to.")

    @model_validator(mode='after')
    def set_default_ref_model_path(self) -> 'ModelConfig':
        if self.ref_model_path is None:
            self.ref_model_path = self.model_path
        return self

class TrainerParams(BaseModel):
    algorithm: str = Field(..., description="RL algorithm to use (e.g., 'grpo', 'ppo').")
    output_dir: Path = Field(Path("./outputs"), description="Directory to save checkpoints and logs.")
    num_training_steps: PositiveInt = Field(10000, description="Total number of training steps.")
    learning_rate: NonNegativeFloat = Field(2e-6, description="Optimizer's learning rate.")
    ppo_batch_size: PositiveInt = Field(4, description="Number of unique prompts per micro-batch during rollouts.")
    num_rollout_samples: PositiveInt = Field(2, description="Number of responses to generate per prompt in rollouts.")
    grad_accum_steps: PositiveInt = Field(4, description="Number of micro-batches to accumulate gradients over.")
    save_every: PositiveInt = Field(500, description="Save a checkpoint every N updates.")
    eval_every: PositiveInt = Field(250, description="Run evaluation every N updates.")
    grpo_beta: NonNegativeFloat = Field(0.05, description="GRPO beta parameter for KL penalty.")
    seed: int = Field(42, description="Random seed for reproducibility.")

    # Gradient Scaling/Masking
    low_band: Tuple[int, int] = Field((0, 15), description="Layer index range for low gradient multiplier band.")
    mid_band: Tuple[int, int] = Field((16, 23), description="Layer index range for mid gradient multiplier band.")
    top_band: Tuple[int, int] = Field((24, 35), description="Layer index range for top gradient multiplier band.")
    low_mul: NonNegativeFloat = Field(0.10, description="Gradient multiplier for low band layers.")
    mid_mul: NonNegativeFloat = Field(0.95, description="Gradient multiplier for mid band layers.")
    top_mul: NonNegativeFloat = Field(1.5, description="Gradient multiplier for top band layers.")
    head_mul: NonNegativeFloat = Field(1.2, description="Gradient multiplier for LM head.")
    train_layer_start: int = Field(26, description="Starting layer index for gradient masking.")
    train_layer_end: int = Field(35, description="Ending layer index for gradient masking.")
    max_grad_norm: NonNegativeFloat = Field(0.5, description="Maximum gradient norm for clipping (0 to disable).")

    # Dynamic Bias Controls for Generation
    min_think_tokens: PositiveInt = Field(32, description="Minimum desired tokens in thinking region before allowing early end.")
    think_end_early_bias: float = Field(-12.0, description="Negative bias for ending think region too early.")
    bias_answer_start_after_min_think: bool = Field(True, description="Only bias for <answer> start after min_think_tokens.")
    bias_close_think: float = Field(9.0, description="Positive bias for closing <think> tag.")
    bias_answer_start: float = Field(6.0, description="Positive bias for starting <answer> tag.")
    punish_extra_think_end: float = Field(-12.0, description="Penalty for extra </think> tags.")
    punish_reopen_think: float = Field(-10.0, description="Penalty for re-opening <think> tag.")
    punish_reopen_answer: float = Field(-9.0, description="Penalty for re-opening <answer> tag.")
    bias_eos_after_answer: float = Field(3.0, description="Positive bias for EOS after <answer>.")

    # MCQ Specific Biases
    hard_mask_mcq_first_token: bool = Field(True, description="Hard mask logits to only allow valid MCQ letters as first token.")
    mcq_letter_lift: NonNegativeFloat = Field(8.0, description="Positive bias for valid MCQ letter tokens.")
    mcq_ban_first_bias: float = Field(-14.0, description="Negative bias for banned phrases as first token in MCQ.")
    nonmcq_ban_first_bias: float = Field(-12.0, description="Negative bias for banned phrases as first token in non-MCQ.")
    mcq_close_after_k: PositiveInt = Field(1, description="Allow MCQ answer to close after K tokens.")
    min_answer_tokens: PositiveInt = Field(8, description="Minimum tokens for non-MCQ answer before allowing end.")
    min_answer_tokens_mcq: PositiveInt = Field(1, description="Minimum tokens for MCQ answer before allowing end.")
    mcq_answer_end_bias: float = Field(9.0, description="Positive bias for closing MCQ answer tag.")

    # Penalties
    non_ascii_penalty: NonNegativeFloat = Field(1.0, description="Penalty multiplier for non-ASCII characters.")
    off_topic_jaccard_threshold: NonNegativeFloat = Field(0.05, description="Jaccard threshold below which off-topic penalty applies.")
    off_topic_penalty: NonNegativeFloat = Field(1.0, description="Penalty multiplier for off-topic responses.")
    ban_penalty: NonNegativeFloat = Field(3.0, description="Penalty multiplier for banned keywords.")

    # Verbosity Biasing for Rollouts
    ban_phrases_for_bias: List[str] = Field(default_factory=list, description="Phrases whose first token triggers negative logit bias.")
    encourage_phrases_for_bias: List[str] = Field(default_factory=list, description="Phrases whose first token triggers positive logit bias.")
    encourage_think_bias: NonNegativeFloat = Field(4.5, description="Positive bias for encouraged compact notation in <think>.")
    ban_think_bias: float = Field(-3.0, description="Negative bias for verbose phrases in <think>.")

    # Tool Use Configuration
    allow_tool_calls: bool = Field(True, description="Whether to allow tool call generation.")
    tool_call_penalty: NonNegativeFloat = Field(0.0, description="Penalty multiplier for generating tool call tokens.")

    # Think Length Penalty/Reward
    think_length_target_min: PositiveInt = Field(32, description="Minimum desired think tokens (soft floor).")
    think_length_target_max: PositiveInt = Field(128, description="Maximum desired think tokens (soft ceiling).")
    think_length_penalty_strength: NonNegativeFloat = Field(0.15, description="Strength of length penalty.")
    think_length_penalty_type: str = Field("quadratic", description="Penalty curve type: 'linear', 'quadratic', 'exponential'.")
    enable_think_length_penalty: bool = Field(True, description="Enable think length penalty in reward calculation.")

    # Custom Invalid Sample Handling
    use_custom_batch_builder: bool = Field(False, description="Enable custom batching for invalid samples.")
    invalid_sample_layers: str = Field("33,34,35", description="Comma-separated layer indices for invalid sample updates.")
    invalid_sample_frequency: PositiveInt = Field(2, description="Frequency (in updates) for processing invalid samples.")
    invalid_sample_layers_set: Set[int] = Field(default_factory=set, exclude=True) # Exclude from config dump, populated in post_init

    # W&B Configuration
    use_wandb: bool = Field(True, description="Enable Weights & Biases logging.")
    wandb_project: Optional[str] = Field("mlx-rl-trainer", description="W&B project name.")
    wandb_entity: Optional[str] = Field(None, description="W&B entity name.")
    wandb_run_name: Optional[str] = Field(None, description="W&B run name.")
    log_samples_every: PositiveInt = Field(10, description="Log generated samples to W&B/file every N updates.")
    max_logged_samples: PositiveInt = Field(5, description="Maximum number of samples to log to W&B/file.")
    log_prompts: bool = Field(True, description="Log full prompts in sample logs.")

    # Schedule
    lr_schedule_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration for learning rate scheduler.")

    # Monitoring
    reward_smoothing_window: PositiveInt = Field(20, description="Window size for rolling average reward display.")
    enable_dynamic_beta: bool = Field(False, description="Enable dynamic adjustment of GRPO beta.") # NEW default
    high_reward_threshold: NonNegativeFloat = Field(0.85, description="Reward threshold to trigger beta increase.")
    beta_increase_high_reward: NonNegativeFloat = Field(0.08, description="Amount to increase beta on high reward.")
    cooldown_duration: PositiveInt = Field(2, description="Cooldown steps after dynamic beta adjustment.")


    # Derived fields, initialized in post_init (exclude from direct YAML parsing)
    effective_batch_size: PositiveInt = Field(1, init=False, exclude=True) # Total samples processed per optimization step
    current_update: int = Field(0, init=False, exclude=True) # Internal tracker for current update step

    @model_validator(mode='after')
    def populate_derived_fields(self) -> 'TrainerParams':
        # Ensure positive values
        for f_name in ['min_answer_tokens', 'min_answer_tokens_mcq', 'think_len_min', 'think_len_max']:
            current_val = getattr(self, f_name)
            if current_val < 0:
                logging.warning(f"{f_name} cannot be negative. Setting to 0.")
                setattr(self, f_name, 0)

        # Ensure think_len_max is always greater than or equal to think_len_min
        if self.think_len_max < self.think_len_min:
            logging.warning(f"think_len_max ({self.think_len_max}) cannot be less than think_len_min ({self.think_len_min}). Setting think_len_max = think_len_min + 1.")
            setattr(self, 'think_len_max', self.think_len_min + 1)

        self.effective_batch_size = self.ppo_batch_size * self.num_rollout_samples * self.grad_accum_steps

        # Populate invalid_sample_layers_set
        if isinstance(self.invalid_sample_layers, str):
            try:
                self.invalid_sample_layers_set = {int(i.strip()) for i in self.invalid_sample_layers.split(',') if i.strip()}
            except ValueError:
                logging.warning(f"Invalid format for invalid_sample_layers: {self.invalid_sample_layers}. Using empty set.")
                self.invalid_sample_layers_set = set()
        else: # Ensure it's always a set, even if config didn't provide a string
            self.invalid_sample_layers_set = set()

        # Initialize lr_schedule_config defaults (if not already set in YAML)
        cfg = self.lr_schedule_config
        init_lr = float(self.learning_rate)
        total_steps = int(self.num_training_steps)
        warmup_steps = int(cfg.get('warmup', 500))
        decay_steps = max(total_steps - warmup_steps, 1)
        end_lr = max(init_lr * .1, 1e-07)

        cfg.setdefault('name', 'cosine_decay')
        cfg.setdefault('arguments', [init_lr, decay_steps, end_lr])
        cfg.setdefault('warmup', warmup_steps)
        cfg.setdefault('warmup_init', min(init_lr, max(init_lr * .1, 1e-08)))

        # Dynamically set encourage_phrases_for_bias if default is empty (from TrainingArgs)
        if not self.encourage_phrases_for_bias:
             # Use the new constants
             setattr(self, 'encourage_phrases_for_bias', list(DEFAULT_SYMBOLIC_CHARS) + list(DEFAULT_ABBREVIATIONS))

        # Also ensure ban_keywords are consistent
        if not self.ban_phrases_for_bias:
            # Use the new constant
            setattr(self, 'ban_phrases_for_bias', DEFAULT_BAN_KEYWORDS)

        return self


class CheckpointConfig(BaseModel):
    save_dir: Path = Field("checkpoints", description="Directory to save checkpoints.")
    keep_best_n: int = Field(3, description="Number of best checkpoints to keep.")


class MonitoringConfig(BaseModel):
    use_wandb: bool = Field(True, description="Enable Weights & Biases logging.")


class ExperimentConfig(BaseModel):
    """Complete experiment configuration for an RL training run."""
    trainer: TrainerParams
    model: ModelConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig, description="Parameters for text generation.")
    rewards: List[RewardConfig] = Field(default_factory=list, description="List of reward function configurations.")
    data: DataConfig
    evaluation: List[EvaluatorConfig] = Field(default_factory=list, description="List of evaluator configurations.") # FIX: Changed to List[EvaluatorConfig] for clarity
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig, description="Configuration for checkpointing.")
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="Configuration for monitoring and logging.")

    # Global/derived fields directly in ExperimentConfig
    max_kv_size: PositiveInt = Field(1536, description="Maximum KV cache size for generation.")
    kv_bits: Optional[PositiveInt] = Field(0, description="Quantization bits for KV cache (0 for none).")
    kv_group_size: PositiveInt = Field(64, description="Group size for KV cache quantization.")
    quantized_kv_start: int = Field(10, description="Layer index to start KV cache quantization.") # Can be 0
    run_server: bool = Field(False, description="Run in async inference server mode instead of training.")
    use_paged_kv_cache: bool = Field(False, description="Enable the PagedKVCache for rollout generation.")
    kv_cache_block_size: PositiveInt = Field(16, description="Number of tokens per block in PagedKVCache.")
    kv_cache_num_blocks: PositiveInt = Field(2048, description="Total number of blocks in PagedKVCache.")
    allow_cross_arch_ref: bool = Field(False, description="Allow cross-architecture reference model (requires alignment bridge).")
    align_bridge_path: Optional[Path] = Field(None, description="Path to cross-architecture alignment bridge weights.")
    align_bridge_weight: NonNegativeFloat = Field(1.0, description="Weight for alignment bridge reward.")
    align_pool: str = Field("mean", description="Pooling method for alignment bridge ('mean' or 'last').")
    align_after_tag: str = Field("</think>", description="Tag after which alignment reward is computed.")

    system_prompt: str = Field(THINK_STYLE_PROMPT_LITERAL, description="System prompt to use for chat template.")

    @model_validator(mode='after')
    def validate_reward_weights_sum(self) -> 'ExperimentConfig':
        """Ensure reward weights sum to approximately 1.0 (if rewards are defined)."""
        if self.rewards:
            total_weight = sum(reward.weight for reward in self.rewards)
            if not (0.99 <= total_weight <= 1.01):
                logging.warning(f"Reward weights do not sum to 1.0 (got {total_weight:.2f}). This may lead to unexpected reward scaling. Consider normalizing them.")
        return self

    @classmethod
    def load_from_yaml(cls, path: Path) -> 'ExperimentConfig':
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            instance = cls(**raw_config)
            instance.trainer.output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Loaded and validated configuration from {path}.")
            return instance
        except ValidationError as e:
            console.print(f"[bold red]Configuration Validation Error:[/bold red]\n{e}")
            raise ValueError(f"Invalid configuration in {path}.") from e
        except Exception as e:
            console.print(f"[bold red]Failed to load configuration from {path}:[/bold red] {e}")
            raise ValueError(f"Failed to load configuration from {path}.") from e


class TrainingArgs(BaseModel):
    train_dataset_path: Path = Field(..., description="Path to training dataset.")
    val_dataset_path: Optional[Path] = Field(None, description="Path to validation dataset.")
    dataset_prompt_key: str = Field("prompt", description="Key for prompt in dataset.")
    dataset_answer_key: str = Field("completion", description="Key for answer in dataset.")
    dataset_name: Optional[str] = Field(None, description="HuggingFace dataset name.")
    dataset_config: Optional[str] = Field(None, description="HuggingFace dataset config name.")
    dataset_train_split: str = Field("train", description="HuggingFace dataset train split.")
    dataset_val_split: str = Field("validation", description="HuggingFace dataset validation split.")
