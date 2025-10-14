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

THINK_STYLE_PROMPT_LITERAL = ""
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
        20, description="Save a full checkpoint every N training updates."
    )
    keep_last_n: PositiveInt = Field(
        2, description="Number of most recent checkpoints to retain."
    )
    save_optimizer_state: bool = Field(
        False, description="Whether to save the optimizer's state."
    )


class MonitoringConfig(BaseModel):
    use_wandb: bool = Field(True, description="Enable Weights & Biases (W&B) logging.")
    wandb_project: Optional[str] = Field(
        "mlx-grpo-qwen3-v3", description="W&B project name."
    )
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
    answer_start_tag: str = Field("")
    answer_end_tag: str = Field("")

    # Sampling parameters
    think_boost_tokens: int = Field(32)
    think_temperature: NonNegativeFloat = Field(0.15)
    answer_temperature: NonNegativeFloat = Field(0.12)
    sampling_top_p: NonNegativeFloat = Field(0.6)
    sampling_min_p: NonNegativeFloat = Field(0.00)
    sampling_top_k: int = Field(60)
    repetition_penalty: Optional[float] = Field(1.4)
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


    ban_phrases_for_bias: List[str] = Field(
        default_factory=lambda: [
            "<think>\\n<|im_start|>",
            "Confused",
            "stuck",
            "frustrated",
            "<|im_start|>",
            "<|endoftext|>",
            "<think>\n<|im_start|><think>\n<|im_start|><think>\n<think>",
            # "<think>\n<think>",
            "I think the answer",
            "I believe that",
            "In my view",
            "From what I can tell",
            "It seems to me",
            "It appears that",
            # "My understanding is",
            "As far as I know",
            # "Let me start by",
            # "Let me first",
            "I should probably",
            "I need to figure out",
            "I'm trying to",
            "I'm going to try",
            "I'll attempt to",
            "Confused",
            "stక్",
            "frustrated",
            "frustrating",
            "Alternatively",
            "Actually",
            "Probably not sure",
            "Uncertain about",
            "Unclear whether",
            "I'm guessing that",
            "maybe this is",
            "Could be that",
            "Might be because",
            "I'm not 100% sure",
            "I'm not sure if",
            "I'm not certain",
            "Hard to say",
            "Difficult to tell",
            "Circular reasoning detected",
            "In some way or another",
            "Magically works",
            "For some unknown reason",
            "Too complicated",
            "It just somehow",
            "Something seems off",
            "False assumption",
            "Insufficient information to",
            "Wait, what if",
            "Wait, "
            "Wait, another idea:",
            "Wait, unless...",
            "Wait, perhaps","Wait, let's see","Wait, here's","Wait, no. Wait,","Wait, wait! Wait,",
            "Wait, actually no",
            "Wait, on second thought",
            "Hold on, maybe",
            "Hmm, perhaps",
            "Or wait, could",
            "Looking at this more closely",
            "Upon further reflection",
            "Taking a step back",
            "Thinking about it more",
            "Now that I consider",
            "When I really think",
            "If I had to guess",
            "To be completely honest",
            "In all honesty",
            "You know what",
            "The thing is",
            "What I mean is",
            "In other words",
            "Put simply",
            "Basically what happens",
            "Long story short",
            "At the end of the day",
        ]
    )

    # ═══════════════════════════════════════════════════════════
    encourage_phrases_for_bias: List[str] = Field(
        default_factory=lambda: [
            # === Mathematical Symbols ===
            "∴",  # therefore
            "∵",  # because
            "⇒",  # implies
            "→",  # leads to
            "≈",  # approximately
            "≠",  # not equal
            "≤",  # less than or equal
            "≥",  # greater than or equal
            "∈",  # element of
            "∀",  # for all
            "∃",  # there exists
            # === Logic Symbols ===
            "✓",  # correct/yes
            "✗",  # wrong/no
            "∧",  # and
            "∨",  # or
            "¬",  # not
            "⊕",  # xor
            "⇔",  # if and only if
            # === Status/Flow Symbols ===
            "▸",  # next
            "◂",  # previous
            "●",  # active
            "○",  # pending
            "◐",  # partial
            "⊗",  # blocked
            "⊘",  # invalid
            # === Compact Abbreviations ===
            "chk",  # check
            "calc",  # calculate
            "eval",  # evaluate
            "cmp",  # compare
            "est",  # estimate
            "approx",  # approximate
            "impl",  # implement
            "proc",  # process
            "init",  # initialize
            "upd",  # update
            "del",  # delete
            "cfg",  # config
            "req",  # required
            "opt",  # optional
            "max",
            "min",
            "avg",
            "sum",
            "diff",
            # === Common Shorthand ===
            "w/",  # with
            "w/o",  # without
            "b/c",  # because
            "re:",  # regarding
            "vs",  # versus
            "via",
            "per",
            "thru",
            "i.e.",
            "e.g.",
            "etc.",
            "NB",  # note well
            # === Operators & Notation ===
            "@",  # at
            "#",  # number
            "&",  # and
            "+",  # plus/add
            "-",  # minus/subtract
            "×",  # multiply
            "÷",  # divide
            "/",  # per/divide
            "|",  # or/pipe
            "~",  # approximately
            # === Conditional Logic (Compact) ===
            "IF",
            "THEN",
            "ELSE",
            "WHEN",
            "CASE",
            "=>",  # arrow function
            "->",  # pointer/flow
            "<-",  # from
            # === Action Verbs (Imperative, Compact) ===
            "find",
            "get",
            "set",
            "test",
            "run",
            "add",
            "sub",
            "mul",
            "div",
            "verify",
            "confirm",
            "validate",
            "parse",
            "extract",
            "merge",
            "split",
            "filter",
            "sort",
            "map",
            "reduce",
            # === Domain-Specific Compact Terms ===
            "func",  # function
            "var",  # variable
            "obj",  # object
            "arr",  # array
            "dict",  # dictionary
            "str",  # string
            "int",  # integer
            "bool",  # boolean
            "async",
            "await",
            "API",
            "DB",  # database
            "idx",  # index
            "len",  # length
            "val",  # value
            "key",
            "param",  # parameter
            "arg",  # argument
            "ret",  # return
            # === Measurement/Time (Abbreviated) ===
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
            "cur",  # current
            # === Business/Analysis Shorthand ===
            "rev",  # revenue
            "exp",  # expense
            "proj",  # projection
            "KPI",
            "ROI",
            "YoY",  # year over year
            "MoM",  # month over month
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "cust",  # customer
            "mkt",  # market
            "comp",  # competitor
            "strat",  # strategy
            "ops",  # operations
            # === Problem-Solving Markers (Compact) ===
            "goal:",
            "constraint:",
            "given:",
            "find:",
            "prove:",
            "show:",
            "result:",
            "answer:",
            "solution:",
            "assume:",
            "note:",
            # === Compact Lists/Enumeration ===
            "1.",
            "2.",
            "3.",
            "a)",
            "b)",
            "c)",
            # "-",  # bullet - too common, causes issues
            "•",  # bullet
            "◦",  # sub-bullet
            # === Question/Clarification (Concise) ===
            "?",  # uncertainty marker
            "unclear:",
            "need:",
            "missing:",
            "ASK:",
            "clarify:",
            # === Status Indicators ===
            "DONE",
            "TODO",
            "WIP",  # work in progress
            "BLOCKED",
            "PENDING",
            "PASS",
            "FAIL",
            "OK",
            "ERROR",
            "WARN",
            # === Compact Comparisons ===
            "better",
            "worse",
            "higher",
            "lower",
            "same",
            "diff",  # different
            "equal",
            "similar",
            "pros:",
            "cons:",
            "trade-off:",
            "cost/benefit:",
            # === Scientific/Technical ===
            # "exp",  # experiment - collides with expense
            "obs",  # observation
            "hyp",  # hypothesis
            "ctrl",  # control
            "sig",  # significant
            "corr",  # correlation
            "temp",  # temperature
            "pres",  # pressure
            "vol",  # volume
            "conc",  # concentration
            # === Reasoning Shortcuts ===
            "premise→conclusion",
            "cause→effect",
            "condition→result",
            "input→output",
            "before→after",
            "undetermined",
            "stop",
            "overthinking",
            "since",
            "already",
            "proven",
            "proof",
            "misstated",
            # "same", # already present
            # Python keywords
            "False",
            "None",
            "True",
            "and",
            "as",
            "assert",
            # "async", # already present
            # "await", # already present
            "break",
            "class",
            "continue",
            "def",
            # "del", # already present
            "elif",
            "else",
            "except",
            "finally",
            "for",
            "from",
            "global",
            "if",
            "import",
            "in",
            "is",
            "lambda",
            "nonlocal",
            "not",
            "or",
            "pass",
            "raise",
            "return",
            "try",
            "while",
            "with",
            "yield",
            "match",
            # "case", # already present
            # Python operators
            # "+", # already present
            # "-", # already present
            "*",
            # "/", # already present
            "//",
            "%",
            "**",
            "==",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
            "=",
            "+=",
            "-=",
            "*=",
            "/=",
            "//=",
            "%=",
            "**=",
            "&=",
            "|=",
            "^=",
            ">>=",
            "<<=",
            # "&", # already present
            # "|", # already present
            "^",
            # "~", # already present
            "<<",
            ">>",
            # Python brackets and delimiters
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            # Python punctuation
            ",",
            ":",
            ";",
            ".",
            # "@", # already present
            # "->", # already present
            # "=>", # already present
            "\\",
            # "#", # already present
            "'",
            '"',
            '"""',
            "'''",
            "`",
            # "!", # already present
            # "?", # already present
            "_",
            "$",
            # JSON syntax
            "true",
            "false",
            "null",
            # Python built-ins
            "print",
            # "len", # already present
            "range",
            # "str", # already present
            # "int", # already present
            "float",
            "list",
            # "dict", # already present
            # "set", # already present
            "tuple",
            "type",
            "isinstance",
            "open",
            "input",
            "format",
            # "min", # already present
            # "max", # already present
            # "sum", # already present
            "abs",
            "all",
            "any",
            "enumerate",
            "zip",
            # "map", # already present
            # "filter", # already present
            "sorted",
            "reversed",
            # "bool", # already present
            "bytes",
            "bytearray",
            "callable",
            "chr",
            "ord",
            "compile",
            "complex",
            "delattr",
            "dir",
            "divmod",
            # "eval", # already present
            "exec",
            "getattr",
            "globals",
            "hasattr",
            "hash",
            "help",
            "hex",
            "id",
            "iter",
            "locals",
            # "next", # already present
            "oct",
            "pow",
            "repr",
            "round",
            "setattr",
            "slice",
            "staticmethod",
            "super",
            "vars",
            "__import__",
            "property",
            "classmethod",
            "frozenset",
            "memoryview",
            # "object", # already present
            "ascii",
            "bin",
            "breakpoint",
            "copyright",
            "credits",
            "exit",
            "license",
            "quit",
            # Python exceptions
            "Exception",
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "AttributeError",
            "ImportError",
            "ModuleNotFoundError",
            "RuntimeError",
            "ZeroDivisionError",
            "FileNotFoundError",
            "IOError",
            "OSError",
            "StopIteration",
            "GeneratorExit",
            "KeyboardInterrupt",
            "SystemExit",
            "NameError",
            "SyntaxError",
            "IndentationError",
            "TabError",
            "UnboundLocalError",
            "UnicodeError",
            "UnicodeEncodeError",
            "UnicodeDecodeError",
            "UnicodeTranslateError",
            "AssertionError",
            "EOFError",
            "FloatingPointError",
            "OverflowError",
            "RecursionError",
            "NotImplementedError",
            "MemoryError",
            "ReferenceError",
            "SystemError",
            "Warning",
            "UserWarning",
            "DeprecationWarning",
            "SyntaxWarning",
            "RuntimeWarning",
            "FutureWarning",
            "PendingDeprecationWarning",
            "ImportWarning",
            "UnicodeWarning",
            "BytesWarning",
            "ResourceWarning",
            "ConnectionError",
            "BrokenPipeError",
            "ConnectionAbortedError",
            "ConnectionRefusedError",
            "ConnectionResetError",
            "BlockingIOError",
            "ChildProcessError",
            "InterruptedError",
            "IsADirectoryError",
            "NotADirectoryError",
            "PermissionError",
            "ProcessLookupError",
            "TimeoutError",
            # Bash keywords
            # "if", # already present
            # "then", # already present
            # "else", # already present
            # "elif", # already present
            "fi",
            # "case", # already present
            "esac",
            # "for", # already present
            "select",
            # "while", # already present
            "until",
            "do",
            "done",
            # "in", # already present
            "function",
            "time",
            "coproc",
            "declare",
            "typeset",
            "local",
            "readonly",
            "unset",
            "shift",
            # "return", # already present
            # "exit", # already present
            # "break", # already present
            # "continue", # already present
            "trap",
            "wait",
            # "eval", # already present
            # "exec", # already present
            "source",
            "builtin",
            "command",
            "enable",
            # "help", # already present
            "let",
            "mapfile",
            "read",
            "readarray",
            # "set", # already present
            "shopt",
            # "test", # already present
            # Bash/Shell commands
            "echo",
            "cd",
            "ls",
            "pwd",
            "mkdir",
            "rm",
            "cp",
            "mv",
            "touch",
            "cat",
            "grep",
            # "find", # already present
            "chmod",
            "chown",
            "sudo",
            "apt",
            "yum",
            "dnf",
            "pacman",
            "zypper",
            "git",
            "curl",
            "wget",
            "tar",
            "gzip",
            "bzip2",
            "xz",
            "compress",
            "uncompress",
            "zip",
            "unzip",
            "kill",
            "killall",
            "pkill",
            "ps",
            "top",
            "htop",
            "df",
            "du",
            "head",
            "tail",
            "less",
            "more",
            # "sort", # already present
            "uniq",
            "wc",
            "cut",
            "sed",
            "awk",
            "export",
            # "source", # already present
            "alias",
            "unalias",
            "which",
            "whereis",
            "man",
            "info",
            "history",
            "clear",
            "reset",
            # "exit", # already present
            "logout",
            "ssh",
            "scp",
            "rsync",
            "ftp",
            "sftp",
            "nc",
            "netcat",
            "telnet",
            "ping",
            "traceroute",
            "tracepath",
            "mtr",
            "dig",
            "nslookup",
            "host",
            "whois",
            "ifconfig",
            "ip",
            "route",
            "netstat",
            "ss",
            "iptables",
            "ufw",
            "firewall-cmd",
            "tcpdump",
            "wireshark",
            "nmap",
            "masscan",
            # Bash operators and symbols
            "&&",
            "||",
            # "!", # already present
            # "|", # already present
            # "&", # already present
            # ";", # already present
            ";;",
            ";&",
            ";;&",
            # "(", # already present
            # ")", # already present
            # "{", # already present
            # "}", # already present
            # "[", # already present
            # "]", # already present
            "[[",
            "]]",
            # "$", # already present
            "${",
            # "}", # already present
            "$((",
            "))",
            "$(",
            # ")", # already present
            # "`", # already present
            # "<", # already present
            # ">", # already present
            ">>",
            "<<",
            "<<<",
            "2>",
            "&>",
            "<&",
            ">&",
            "2>&1",
            "&>>",
            "/dev/null",
            "/dev/zero",
            "/dev/urandom",
            # "~", # already present
            # "*", # already present
            # "**", # already present
            # "?", # already present
            "..",
            # ".", # already present
            # "/", # already present
            # "-", # already present
            "--",
            "-eq",
            "-ne",
            "-lt",
            "-le",
            "-gt",
            "-ge",
            "-z",
            "-n",
            "-e",
            "-f",
            "-d",
            "-r",
            "-w",
            "-x",
            "-s",
            "-h",
            "-L",
            "-S",
            "-p",
            "-b",
            "-c",
            "-u",
            "-g",
            "-k",
            "-O",
            "-G",
            "-N",
            "-nt",
            "-ot",
            "-ef",
            # Bash variables and special
            "$?",
            "$!",
            "$$",
            "$0",
            "$1",
            "$2",
            "$3",
            "$4",
            "$5",
            "$6",
            "$7",
            "$8",
            "$9",
            "$@",
            "$#",
            "$*",
            "$-",
            "$_",
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "HOSTNAME",
            "SHELL",
            "PWD",
            "OLDPWD",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TERM",
            "EDITOR",
            "VISUAL",
            "PAGER",
            "MANPATH",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "CLASSPATH",
            "JAVA_HOME",
            "NODE_PATH",
            "GOPATH",
            "RUST_HOME",
            "HISTFILE",
            "HISTSIZE",
            "HISTFILESIZE",
            "HISTCONTROL",
            "PROMPT_COMMAND",
            "PS1",
            "PS2",
            "PS3",
            "PS4",
            "IFS",
            "TMPDIR",
            "TMP",
            "TEMP",
            "DISPLAY",
            "XAUTHORITY",
            "UID",
            "EUID",
            "GROUPS",
            "SECONDS",
            "RANDOM",
            "LINENO",
            "BASH",
            "BASH_VERSION",
            "BASH_VERSINFO",
            # GNU Coreutils - File operations
            # "cp", # already present
            # "mv", # already present
            # "rm", # already present
            # "mkdir", # already present
            "rmdir",
            "ln",
            # "touch", # already present
            "install",
            "dd",
            "shred",
            "sync",
            "truncate",
            "realpath",
            "readlink",
            "basename",
            "dirname",
            # GNU Coreutils - Text processing
            # "cat", # already present
            "tac",
            "nl",
            "od",
            "base32",
            "base64",
            "fmt",
            "pr",
            "fold",
            # "head", # already present
            # "tail", # already present
            # "split", # already present
            "csplit",
            # "wc", # already present
            # "sum", # already present
            "cksum",
            "md5sum",
            "sha1sum",
            "sha224sum",
            "sha256sum",
            "sha384sum",
            "sha512sum",
            "b2sum",
            # GNU Coreutils - Output formatting
            # "sort", # already present
            "shuf",
            # "uniq", # already present
            "comm",
            "ptx",
            "tsort",
            # "cut", # already present
            "paste",
            "join",
            "tr",
            "expand",
            "unexpand",
            "column",
            # GNU Coreutils - Directory operations
            # "ls", # already present
            # "dir", # already present
            "vdir",
            "dircolors",
            # "pwd", # already present
            "pushd",
            "popd",
            "dirs",
            # GNU Coreutils - Basic operations
            # "echo", # already present
            "printf",
            "yes",
            # "true", # already present
            # "false", # already present
            # "test", # already present
            # "[", # already present
            "expr",
            "tee",
            # GNU Coreutils - File name manipulation
            # "basename", # already present
            # "dirname", # already present
            "pathchk",
            "mktemp",
            # "realpath", # already present
            # GNU Coreutils - Working context
            # "pwd", # already present
            "stty",
            "printenv",
            "tty",
            # GNU Coreutils - User information
            # "id", # already present
            # "logname", # already present
            "whoami",
            # "groups", # already present
            "users",
            "who",
            "pinky",
            "finger",
            "last",
            "lastb",
            "w",
            # GNU Coreutils - System context
            "date",
            "arch",
            "nproc",
            "uname",
            # "hostname", # already present
            "hostid",
            "uptime",
            # GNU Coreutils - SELinux context
            "chcon",
            "runcon",
            # GNU Coreutils - Modified command invocation
            "chroot",
            "env",
            "nice",
            "nohup",
            "stdbuf",
            "timeout",
            # GNU Coreutils - Process control
            # "kill", # already present
            "sleep",
            # GNU Coreutils - Numeric operations
            "factor",
            "numfmt",
            "seq",
            # GNU Coreutils - File permissions
            "chgrp",
            # "chmod", # already present
            # "chown", # already present
            "stat",
            # "df", # already present
            # "du", # already present
            # "sync", # already present
            "mkfifo",
            "mknod",
            "link",
            "unlink",
            # "install", # already present
            # "vdir", # already present
            # "dir", # already present
            # Binutils
            "ar",
            # "as", # already present
            "ld",
            "nm",
            "objcopy",
            "objdump",
            "ranlib",
            "readelf",
            "size",
            "strings",
            "strip",
            "c++filt",
            "addr2line",
            "elfedit",
            "gprof",
            "ld.bfd",
            "ld.gold",
            "dwp",
            "windres",
            "dlltool",
            "nlmconv",
            "srconv",
            "sysdump",
            "coffdump",
            # "readelf", # already present
            "eu-readelf",
            # File system utilities
            "mount",
            "umount",
            "fdisk",
            "parted",
            "gparted",
            "mkfs",
            "mkfs.ext2",
            "mkfs.ext3",
            "mkfs.ext4",
            "mkfs.xfs",
            "mkfs.btrfs",
            "mkfs.vfat",
            "mkswap",
            "fsck",
            "fsck.ext2",
            "fsck.ext3",
            "fsck.ext4",
            "e2fsck",
            "xfs_repair",
            "tune2fs",
            "dumpe2fs",
            "resize2fs",
            "blkid",
            "lsblk",
            "findmnt",
            "blockdev",
            # Process management
            # "ps", # already present
            # "top", # already present
            # "htop", # already present
            "atop",
            "iotop",
            # "kill", # already present
            # "killall", # already present
            # "pkill", # already present
            "pgrep",
            # "nice", # already present
            "renice",
            # "nohup", # already present
            "bg",
            "fg",
            "jobs",
            "disown",
            # "wait", # already present
            "pstree",
            "pidof",
            "fuser",
            "lsof",
            "watch",
            "screen",
            "tmux",
            "at",
            "batch",
            "cron",
            "crontab",
            "anacron",
            "systemctl",
            "service",
            "journalctl",
            "systemd",
            # Text editors
            "vi",
            "vim",
            "nvim",
            "emacs",
            "nano",
            "pico",
            "ed",
            # "sed", # already present
            # "awk", # already present
            "gedit",
            "kate",
            "sublime",
            "vscode",
            "code",
            "atom",
            # Compression/Archive utilities
            # "tar", # already present
            # "gzip", # already present
            "gunzip",
            # "bzip2", # already present
            "bunzip2",
            # "xz", # already present
            "unxz",
            # "compress", # already present
            # "uncompress", # already present
            # "zip", # already present
            # "unzip", # already present
            "rar",
            "unrar",
            "7z",
            "p7zip",
            "zcat",
            "bzcat",
            "xzcat",
            "zless",
            "bzless",
            "xzless",
            "zmore",
            "bzmore",
            "xzmore",
            "zgrep",
            "bzgrep",
            "xzgrep",
            "zfgrep",
            "bzfgrep",
            "xzfgrep",
            "zegrep",
            "bzegrep",
            "xzegrep",
            # Search and find utilities
            # "find", # already present
            "locate",
            "updatedb",
            # "grep", # already present
            "egrep",
            "fgrep",
            "rgrep",
            # "zgrep", # already present
            "ag",
            "ack",
            "ripgrep",
            "rg",
            # "whereis", # already present
            # "which", # already present
            "whatis",
            "apropos",
            # Disk usage utilities
            # "df", # already present
            # "du", # already present
            "ncdu",
            "quota",
            "quotacheck",
            "quotaon",
            "quotaoff",
            "repquota",
            "edquota",
            "setquota",
            # Memory and performance monitoring
            "free",
            "vmstat",
            "iostat",
            "mpstat",
            "sar",
            "pidstat",
            # "uptime", # already present
            "dmesg",
            "sysctl",
            "strace",
            "ltrace",
            "perf",
            "valgrind",
            "gdb",
            "lldb",
            # Package managers
            # "apt", # already present
            "apt-get",
            "apt-cache",
            "aptitude",
            "dpkg",
            "dpkg-query",
            # "yum", # already present
            # "dnf", # already present
            "rpm",
            # "zypper", # already present
            # "pacman", # already present
            "pkg",
            "pkgng",
            "brew",
            "port",
            "snap",
            "flatpak",
            "appimage",
            "pip",
            "pip3",
            "easy_install",
            "conda",
            "npm",
            "yarn",
            "pnpm",
            "gem",
            # "bundle", # already present
            "cargo",
            "go",
            "composer",
            "maven",
            "gradle",
            # Markdown syntax
            # "#", # already present
            "##",
            "###",
            "####",
            "#####",
            "######",  # Headers
            # "*", # already present
            # "**", # already present
            "***",
            # "_", # already present
            "__",
            "___",  # Emphasis
            # "-", # already present
            # "+", # already present
            # "*", # already present # Lists
            # "1.", # already present
            # "2.", # already present
            # "3.", # already present # Numbered lists
            # "[", # already present
            # "]", # already present
            # "(", # already present
            # ")", # already present # Links
            "![",
            "](",
            # ")", # already present # Images
            # "`", # already present
            "```",  # Code
            ">",
            # ">>", # already present # Blockquotes
            "---",
            # "***", # already present
            "___",  # Horizontal rules
            # "|", # already present
            "|-",
            "-|",
            "|:",
            ":|",
            "|::|",  # Tables
            "~~",  # Strikethrough
            "- [ ]",
            "- [x]",  # Task lists
            "::",
            ":::",  # Special blocks
            # "\\", # already present # Escape character
            "&nbsp;",
            "&lt;",
            "&gt;",
            "&amp;",
            "&quot;",
            "&#39;",  # HTML entities
            # HTML tags (commonly used in Markdown)
            "<br>",
            "<hr>",
            "<code>",
            "<pre>",
            "<b>",
            "<i>",
            "<u>",
            "<strong>",
            "<em>",
            "<a>",
            "<img>",
            "<table>",
            "<tr>",
            "<td>",
            "<th>",
            "<ul>",
            "<ol>",
            "<li>",
            "<div>",
            "<span>",
            "<p>",
            "<h1>",
            "<h2>",
            "<h3>",
            "<h4>",
            "<h5>",
            "<h6>",
            "<head>",
            "<body>",
            "<html>",
            "<meta>",
            "<link>",
            "<script>",
            "<style>",
            "<header>",
            "<footer>",
            "<nav>",
            "<main>",
            "<section>",
            "<article>",
            "<aside>",
            "<form>",
            "<input>",
            "<button>",
            # "<select>", # already present
            "<option>",
            "<textarea>",
            "<label>",
            "<iframe>",
            "<video>",
            "<audio>",
            # "<source>", # already present
            "<canvas>",
            "<svg>",
            "<think>",
            # "<answer>",  # already present # Special reasoning tags
            # CSS selectors and properties (common)
            # "class", # already present
            # "id", # already present
            # "style", # already present
            "color",
            "background",
            "margin",
            "padding",
            "border",
            "width",
            "height",
            "display",
            "position",
            # "top", # already present
            "left",
            "right",
            "bottom",
            "float",
            "flex",
            "grid",
            "font",
            "text-align",
            "z-index",
            "opacity",
            # JavaScript keywords
            # "var", # already present
            # "let", # already present
            "const",
            # "function", # already present
            # "return", # already present
            # "if", # already present
            # "else", # already present
            # "for", # already present
            # "while", # already present
            # "do", # already present
            "switch",
            # "case", # already present
            "default",
            # "break", # already present
            # "continue", # already present
            # "try", # already present
            "catch",
            # "finally", # already present
            "throw",
            "new",
            "this",
            "typeof",
            "instanceof",
            "void",
            # "delete", # already present
            # "in", # already present
            "of",
            # "async", # already present
            # "await", # already present
            # "yield", # already present
            # "class", # already present
            "extends",
            # "super", # already present
            "static",
            # "import", # already present
            # "export", # already present
            # "from", # already present
            # "default", # already present
            # "as", # already present
            # "null", # already present
            "undefined",
            # "true", # already present
            # "false", # already present
            # Additional programming symbols
            "@property",
            "@staticmethod",
            "@classmethod",
            "@abstractmethod",
            "@dataclass",
            "__init__",
            "__str__",
            "__repr__",
            "__len__",
            "__getitem__",
            "__setitem__",
            "__delitem__",
            "__iter__",
            "__next__",
            "__enter__",
            "__exit__",
            "__call__",
            "__name__",
            "__main__",
            "__file__",
            "__dict__",
            "__doc__",
            "__module__",
            "__class__",
            "__bases__",
            "__mro__",
            "__annotations__",
            "__slots__",
            "__new__",
            "__del__",
            "__hash__",
            "__eq__",
            "__ne__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
            "__bool__",
            "__add__",
            "__sub__",
            "__mul__",
            "__truediv__",
            "__floordiv__",
            "__mod__",
            "__pow__",
            "__and__",
            "__or__",
            "__xor__",
            "__invert__",
            "__lshift__",
            "__rshift__",
            "__contains__",
            "__getattr__",
            "__setattr__",
            "__delattr__",
            # "__dir__", # already present
            "__get__",
            "__set__",
            "__delete__",
            "__init_subclass__",
            "__prepare__",
            "__instancecheck__",
            "__subclasscheck__",
            "__aenter__",
            "__aexit__",
            "__aiter__",
            "__anext__",
            # "__await__", # already present
            # Regular expression patterns
            r"\d",
            r"\D",
            r"\w",
            r"\W",
            r"\s",
            r"\S",
            r"\n",
            r"\t",
            r"\r",
            r"\f",
            r"\v",
            r"\.",
            r"\*",
            r"\+",
            r"\?",
            r"\[",
            r"\]",
            r"\(",
            r"\)",
            r"\{",
            r"\}",
            r"\|",
            r"\^",
            r"\$",
            r"\\",
            r"\b",
            r"\B",
            r"\A",
            r"\Z",
            r"\z",
            # SQL keywords
            "SELECT",
            "FROM",
            "WHERE",
            "INSERT",
            "INTO",
            "VALUES",
            "UPDATE",
            "SET",
            "DELETE",
            "CREATE",
            "DROP",
            "ALTER",
            "TABLE",
            "DATABASE",
            "INDEX",
            "VIEW",
            "PROCEDURE",
            "FUNCTION",
            "TRIGGER",
            "SEQUENCE",
            "SCHEMA",
            "GRANT",
            "REVOKE",
            "COMMIT",
            "ROLLBACK",
            "SAVEPOINT",
            "TRANSACTION",
            "BEGIN",
            "END",
            "JOIN",
            "LEFT",
            "RIGHT",
            "INNER",
            "OUTER",
            "FULL",
            "CROSS",
            "NATURAL",
            "ON",
            "USING",
            "GROUP",
            "ORDER",
            "BY",
            "HAVING",
            "LIMIT",
            "OFFSET",
            # "AS", # already present
            "DISTINCT",
            # "ALL", # already present
            "UNION",
            "INTERSECT",
            "EXCEPT",
            "MINUS",
            "COUNT",
            # "SUM", # already present
            # "AVG", # already present
            # "MIN", # already present
            # "MAX", # already present
            "STDDEV",
            "VARIANCE",
            "AND",
            "OR",
            "NOT",
            "NULL",
            "IS",
            "LIKE",
            "ILIKE",
            "BETWEEN",
            # "IN", # already present
            "EXISTS",
            # "CASE", # already present
            # "WHEN", # already present
            # "THEN", # already present
            # "ELSE", # already present
            "PRIMARY",
            "KEY",
            "FOREIGN",
            "REFERENCES",
            "UNIQUE",
            "CHECK",
            "DEFAULT",
            "AUTO_INCREMENT",
            "SERIAL",
            "CONSTRAINT",
            "CASCADE",
            "RESTRICT",
            "NO",
            "ACTION",
            "CAST",
            "COALESCE",
            "NULLIF",
            # Git commands and flags
            # "git", # already present
            "clone",
            # "init", # already present
            # "add", # already present
            "commit",
            # "push", # already present
            # "pull", # already present
            "fetch",
            # "merge", # already present
            "branch",
            "checkout",
            # "switch", # already present
            "restore",
            "status",
            "log",
            # "diff", # already present
            # "show", # already present
            # "reset", # already present
            "revert",
            "rebase",
            "cherry-pick",
            "stash",
            "tag",
            "remote",
            # "config", # already present
            "blame",
            "bisect",
            # "grep", # already present
            "reflog",
            "clean",
            # "gc", # already present
            # "fsck", # already present
            "prune",
            "archive",
            "bundle",
            "submodule",
            "worktree",
            "describe",
            "shortlog",
            "--amend",
            "--force",
            "--all",
            "--hard",
            "--soft",
            "--mixed",
            "--cached",
            "--staged",
            "--interactive",
            "-i",
            # "-p", # already present
            "-v",
            "-m",
            "-a",
            # "-b", # already present
            # "-d", # already present
            "-D",
            "--origin",
            "--upstream",
            "--set-upstream",
            "--track",
            "--no-track",
            "--continue",
            "--abort",
            "--skip",
            # "--quit", # already present
            "--edit",
            "--no-edit",
            # Docker commands
            "docker",
            "build",
            # "run", # already present
            # "exec", # already present
            # "ps", # already present
            "images",
            # "pull", # already present
            # "push", # already present
            # "tag", # already present
            "rmi",
            # "rm", # already present
            # "stop", # already present
            "start",
            "restart",
            # "kill", # already present
            "pause",
            "unpause",
            "logs",
            "inspect",
            "stats",
            # "top", # already present
            "attach",
            # "cp", # already present
            # "diff", # already present
            # "export", # already present
            # "import", # already present
            # "load", # already present
            "save",
            "network",
            "volume",
            "compose",
            "swarm",
            "docker-compose",
            "up",
            "down",
            "scale",
            "kubectl",
            "k8s",
            "pod",
            "deploy",
            # System administration
            "useradd",
            "userdel",
            "usermod",
            "groupadd",
            "groupdel",
            "groupmod",
            "passwd",
            "chpasswd",
            "chage",
            "su",
            # "sudo", # already present
            "visudo",
            "adduser",
            "deluser",
            "addgroup",
            "delgroup",
            "newgrp",
            "gpasswd",
            # Additional utilities
            "xargs",
            "make",
            "cmake",
            "conf",
            "automake",
            "gcc",
            "g++",
            "clang",
            "clang++",
            "javac",
            "java",
            "python",
            "python2",
            "python3",
            "ruby",
            "perl",
            "php",
            "node",
            "nodejs",
            "bash",
            "sh",
            "zsh",
            "fish",
            "ksh",
            "csh",
            "tcsh",
            # "awk", # already present
            "gawk",
            "mawk",
            "nawk",
            # "sed", # already present
            "bc",
            "dc",
            "units",
            # "date", # already present
            "cal",
            "ncal",
            # "time", # already present
            # "timeout", # already present
            # "yes", # already present
            # "seq", # already present
            "jot",
            # "shuf", # already present
            # "od", # already present
            "xxd",
            "hexdump",
            "file",
            # "stat", # already present
            "tree",
            "pv",
            "progress",
            # "rsync", # already present
            # "scp", # already present
            # "sftp", # already present
            # "ftp", # already present
            "lftp",
            "ncftp",
            # "wget", # already present
            # "curl", # already present
            "aria2c",
            "youtube-dl",
            "yt-dlp",
            # Printer and document utilities
            "lp",
            "lpr",
            "lpq",
            "lprm",
            "lpc",
            "lpstat",
            "cups",
            "ps2pdf",
            "pdf2ps",
            "pdftotext",
            "pdftk",
            "convert",
            "mogrify",
            "identify",
            "montage",
            "composite",
            # "display", # already present
            "animate",
            # "import", # already present
            "conjure",
            "stream",
            # "compare", # already present
            # Audio/Video utilities
            "ffmpeg",
            "ffprobe",
            "ffplay",
            "sox",
            "play",
            "rec",
            "aplay",
            "arecord",
            "paplay",
            "parecord",
            "pulseaudio",
            "alsamixer",
            "amixer",
            "mpv",
            "vlc",
            "mplayer",
            "mencoder",
            "handbrake",
            # "youtube-dl", # already present
            # Python libraries and frameworks - Async/Concurrency
            "mmap",
            "future",
            "concurrent",
            "futures",
            "ThreadPoolExecutor",
            "ProcessPoolExecutor",
            "multiprocessing",
            "mp",
            "Pool",
            "Process",
            "Queue",
            "Manager",
            "Lock",
            "Semaphore",
            "Event",
            "Barrier",
            "threading",
            "Thread",
            "RLock",
            "Condition",
            "Timer",
            # "gc", # already present
            "garbage",
            "collect",
            "get_objects",
            "get_referents",
            "get_referrers",
            "asyncio",
            # "async", # already present
            "create_task",
            "gather",
            "wait_for",
            "shield",
            "ensure_future",
            # "run", # already present
            "create_subprocess_exec",
            "create_subprocess_shell",
            "StreamReader",
            "StreamWriter",
            "aiohttp",
            "ClientSession",
            "aiofiles",
            "aiosqlite",
            "aiomysql",
            "aiopg",
            # Python libraries - CLI/UI
            "click",
            # "command", # already present
            # "option", # already present
            # "argument", # already present
            # "group", # already present
            "pass_context",
            "Context",
            "rich",
            "Console",
            # "Table", # already present
            "Progress",
            "Syntax",
            "Panel",
            # "Tree", # already present
            "Markdown",
            # "print", # already present
            # "progress", # already present
            "track",
            "Live",
            "Layout",
            "Columns",
            "Pretty",
            "Text",
            "tqdm",
            "trange",
            "tnrange",
            "tqdm_notebook",
            "tqdm_gui",
            "progressbar",
            # Python libraries - Data validation/modeling
            "pydantic",
            "BaseModel",
            # "Field", # already present
            # "validator", # already present
            "root_validator",
            # "ValidationError", # already present
            "constr",
            "conint",
            "confloat",
            "EmailStr",
            "HttpUrl",
            "PositiveInt",
            "NegativeInt",
            "sqlmodel",
            "SQLModel",
            # "create_engine", # already present
            "Session",
            # "select", # already present
            "Relationship",
            "jsonschema",
            # "validate", # already present
            "Draft7Validator",
            # "ValidationError", # already present
            "SchemaError",
            # "ABC", # already present
            # "ABCMeta", # already present
            # "abstractmethod", # already present
            "abstractproperty",
            # Python libraries - Data generation
            "faker",
            "Faker",
            "fake",
            "name",
            "address",
            "email",
            "phone_number",
            "company",
            "job",
            # "text", # already present
            "sentence",
            "paragraph",
            # "uuid4", # already present
            # "date", # already present
            # "time", # already present
            "datetime",
            "random",
            "randint",
            "choice",
            "shuffle",
            "sample",
            "uniform",
            "gauss",
            "wandb",
            # "init", # already present
            # "log", # already present
            "finish",
            # "config", # already present
            # "watch", # already present
            # "save", # already present
            # "restore", # already present
            # Python libraries - Scientific computing
            "numpy",
            "np",
            "array",
            "ndarray",
            "zeros",
            "ones",
            "arange",
            "linspace",
            "reshape",
            "transpose",
            "dot",
            "matmul",
            "linalg",
            # "random", # already present
            "mean",
            "std",
            "torch",
            "pytorch",
            "tensor",
            "nn",
            "Module",
            "Linear",
            "Conv2d",
            "ReLU",
            "optim",
            "SGD",
            "Adam",
            "DataLoader",
            "Dataset",
            # "cuda", # already present
            "mps",
            "device",
            "torchvision",
            "transforms",
            # "models", # already present
            "resnet",
            "vgg",
            "alexnet",
            "is_available",
            "set_device",
            "get_device_name",
            # Python libraries - NLP/Text processing
            "nltk",
            "tokenize",
            "word_tokenize",
            "sent_tokenize",
            "pos_tag",
            "ne_chunk",
            "corpus",
            "stopwords",
            "wordnet",
            "stem",
            "lemmatize",
            "FreqDist",
            "ngrams",
            "spacy",
            "nlp",
            "Doc",
            "Token",
            "Span",
            "Vocab",
            "Language",
            "matcher",
            "gensim",
            "Word2Vec",
            "Doc2Vec",
            "FastText",
            "KeyedVectors",
            "LdaModel",
            "CoherenceModel",
            "corpora",
            # "Dictionary", # already present
            "similarities",
            # "models", # already present
            "transformers",
            "BertModel",
            "BertTokenizer",
            "GPT2Model",
            "GPT2Tokenizer",
            # "pipeline", # already present
            "AutoModel",
            "AutoTokenizer",
            "Trainer",
            "TrainingArguments",
            "sentence_transformers",
            "SentenceTransformer",
            "util",
            "encode",
            "similarity",
            "sumy",
            "summarizer",
            "LexRank",
            "TextRank",
            "Luhn",
            "Edmundson",
            "LsaSummarizer",
            "keybert",
            "KeyBERT",
            "extract_keywords",
            "MaxSum",
            "MMR",
            "bertopic",
            "BERTopic",
            # "fit_transform", # already present
            "get_topics",
            "visualize_topics",
            "newspaper",
            "Article",
            # "build", # already present
            # "download", # already present
            # "parse", # already present
            # "nlp", # already present
            "textblob",
            "TextBlob",
            "sentiment",
            "polarity",
            "subjectivity",
            "textwrap",
            "wrap",
            "fill",
            "dedent",
            "indent",
            "shorten",
            "wordcloud",
            "WordCloud",
            "generate",
            "generate_from_text",
            "to_file",
            # Python libraries - Machine Learning/AI
            "sklearn",
            "scikit-learn",
            "fit",
            "predict",
            "transform",
            # "fit_transform", # already present
            "train_test_split",
            "cross_val_score",
            "GridSearchCV",
            "RandomForestClassifier",
            "LogisticRegression",
            "SVC",
            "KMeans",
            "PCA",
            "StandardScaler",
            "MinMaxScaler",
            "tensorflow",
            "tf",
            "keras",
            # "Model", # already present
            "Sequential",
            "Dense",
            "LSTM",
            "GRU",
            "Embedding",
            "Dropout",
            "BatchNormalization",
            # "compile", # already present
            # "fit", # already present
            "evaluate",
            "openai",
            "OpenAI",
            "ChatCompletion",
            # "create", # already present
            "Completion",
            "chat",
            # "completions", # already present
            "messages",
            "response_format",
            "structured_output",
            "instructor",
            # "patch", # already present
            "from_openai",
            "response_model",
            "Instructor",
            # Python libraries - Vector databases and search
            "faiss",
            "IndexFlatL2",
            "IndexIVFFlat",
            "IndexFlatIP",
            # "add", # already present
            "search",
            "lancedb",
            # "connect", # already present
            "create_table",
            "open_table",
            # "search", # already present
            # "delete", # already present
            "chromadb",
            # "Client", # already present
            "Collection",
            # "add", # already present
            "query",
            # "get", # already present
            # "delete", # already present
            "pinecone",
            # "init", # already present
            # "Index", # already present
            "upsert",
            # "query", # already present
            # "fetch", # already present
            # "delete", # already present
            "weaviate",
            # "Client", # already present
            # "schema", # already present
            "data_object",
            # "query", # already present
            "qdrant",
            "QdrantClient",
            # "upsert", # already present
            # "search", # already present
            "scroll",
            "milvus",
            "connections",
            # "Collection", # already present
            "insert",
            # "search", # already present
            # "query", # already present
            # Python libraries - Information retrieval
            "rank_bm25",
            "BM25Okapi",
            "BM25L",
            "BM25Plus",
            "get_scores",
            "get_top_n",
            "dpr",
            "DPRQuestionEncoder",
            "DPRContextEncoder",
            "DPRReader",
            "pyserini",
            "SimpleSearcher",
            # "search", # already present
            "batch_search",
            "elasticsearch",
            "Elasticsearch",
            # "index", # already present
            # "search", # already present
            # "get", # already present
            # "delete", # already present
            "whoosh",
            # "Index", # already present
            # "Schema", # already present
            "create_in",
            "open_dir",
            "searcher",
            # Python libraries - LangChain
            "langchain",
            "LLMChain",
            "PromptTemplate",
            "ChatPromptTemplate",
            "langchain_community",
            "RecursiveCharacterTextSplitter",
            "CharacterTextSplitter",
            "TokenTextSplitter",
            "MarkdownTextSplitter",
            "PythonCodeTextSplitter",
            "Document",
            "VectorStore",
            "Chroma",
            "FAISS",
            "Pinecone",
            "langchain_core",
            "BaseRetriever",
            "BaseLoader",
            "BaseLLM",
            "langchain_openai",
            "ChatOpenAI",
            "OpenAIEmbeddings",
            # Python libraries - Named Entity Recognition
            # "spacy", # already present
            "ner",
            "displacy",
            "render",
            "EntityRecognizer",
            "EntityRuler",
            "flair",
            # "Sentence", # already present
            "SequenceTagger",
            # "load", # already present
            # "predict", # already present
            "stanza",
            # "Pipeline", # already present
            # "download", # already present
            # "ner", # already present
            # "tokenize", # already present
            "allennlp",
            "Predictor",
            "from_path",
            # Python libraries - Similarity metrics
            "scipy",
            "spatial",
            "distance",
            "cosine",
            "euclidean",
            "jaccard",
            "jellyfish",
            "levenshtein_distance",
            "jaro_winkler",
            "soundex",
            "difflib",
            "SequenceMatcher",
            "ratio",
            "get_close_matches",
            "rouge",
            "Rouge",
            # "get_scores", # already present
            "rouge_n",
            "rouge_l",
            "bleu",
            "sentence_bleu",
            "corpus_bleu",
            "SmoothingFunction",
            "meteor",
            "meteor_score",
            "single_meteor_score",
            "bert_score",
            "score",
            "BERTScorer",
            # Python libraries - Language tools
            "language_tool_python",
            "LanguageTool",
            "check",
            "correct",
            "gingerit",
            "GingerIt",
            # "parse", # already present
            "gramformer",
            "Gramformer",
            # "correct", # already present
            "highlight",
            "happytransformer",
            "HappyTextToText",
            "HappyGeneration",
            # Python libraries - Network/Graph
            "networkx",
            "nx",
            "Graph",
            "DiGraph",
            "add_node",
            "add_edge",
            "pagerank",
            "betweenness_centrality",
            "shortest_path",
            "community",
            "igraph",
            # "Graph", # already present
            "add_vertices",
            "add_edges",
            "community_multilevel",
            "pyvis",
            "Network",
            # "add_node", # already present
            # "add_edge", # already present
            # "show", # already present
            "save_graph",
            "graph_tool",
            # "Graph", # already present
            "add_vertex",
            # "add_edge", # already present
            "draw",
            # Python libraries - Web scraping
            "beautifulsoup4",
            "BeautifulSoup",
            # "find", # already present
            "find_all",
            # "select", # already present
            "get_text",
            "requests",
            # "get", # already present
            "post",
            "put",
            # "delete", # already present
            "session",
            # "Response", # already present
            "scrapy",
            "Spider",
            "CrawlSpider",
            # "Request", # already present
            # "Response", # already present
            "Item",
            "selenium",
            "webdriver",
            "Chrome",
            "Firefox",
            "find_element",
            # "click", # already present
            "playwright",
            "sync_api",
            "async_api",
            "chromium",
            "firefox",
            "webkit",
            "httpx",
            "AsyncClient",
            # "Client", # already present
            # "get", # already present
            # "post", # already present
            # "stream", # already present
            # Python libraries - Data structures
            "collections",
            "defaultdict",
            # "Counter", # already present
            "OrderedDict",
            "namedtuple",
            "deque",
            "heapq",
            "heappush",
            "heappop",
            "heapify",
            "nlargest",
            "nsmallest",
            "bisect",
            "bisect_left",
            "bisect_right",
            "insort",
            "itertools",
            "chain",
            "combinations",
            "permutations",
            "product",
            "groupby",
            "functools",
            "lru_cache",
            "partial",
            # "reduce", # already present
            "wraps",
            "cached_property",
            # Python libraries - File handling
            "pathlib",
            "Path",
            # "exists", # already present
            # "mkdir", # already present
            "glob",
            "iterdir",
            "read_text",
            "write_text",
            "os",
            # "path", # already present
            "listdir",
            "makedirs",
            "remove",
            "rename",
            "walk",
            "environ",
            "shutil",
            "copy",
            "copy2",
            "copytree",
            "move",
            "rmtree",
            "make_archive",
            "tempfile",
            "TemporaryFile",
            "NamedTemporaryFile",
            "TemporaryDirectory",
            "mkstemp",
            "pickle",
            "dump",
            # "load", # already present
            "dumps",
            "loads",
            # "json", # already present
            # "dump", # already present
            # "load", # already present
            # "dumps", # already present
            # "loads", # already present
            "JSONEncoder",
            "JSONDecoder",
            "yaml",
            "safe_load",
            "safe_dump",
            # "load", # already present
            # "dump", # already present
            "YAMLError",
            "toml",
            # "load", # already present
            # "dump", # already present
            # "loads", # already present
            # "dumps", # already present
            "csv",
            "reader",
            "writer",
            "DictReader",
            "DictWriter",
            "pandas",
            "pd",
            "DataFrame",
            "Series",
            "read_csv",
            "read_excel",
            "read_json",
            "read_sql",
            "to_csv",
            "to_excel",
            "to_json",
            "to_sql",
            # "merge", # already present
            "concat",
            # "groupby", # already present
            "pivot_table",
            "melt",
            "apply",
            # "map", # already present
            "fillna",
            "dropna",
            # Python libraries - Database
            "sqlite3",
            # "connect", # already present
            "cursor",
            "execute",
            "executemany",
            "fetchall",
            "fetchone",
            "sqlalchemy",
            "create_engine",
            # "Table", # already present
            "Column",
            "Integer",
            "String",
            "MetaData",
            "sessionmaker",
            "declarative_base",
            # "relationship", # already present
            "ForeignKey",
            "psycopg2",
            # "connect", # already present
            # "cursor", # already present
            # "execute", # already present
            # "commit", # already present
            # "rollback", # already present
            "pymongo",
            "MongoClient",
            # "find", # already present
            "find_one",
            "insert_one",
            "update_one",
            "redis",
            "Redis",
            # "set", # already present
            # "get", # already present
            # "delete", # already present
            "expire",
            "ttl",
            # Python libraries - Configuration/Settings
            "configparser",
            "ConfigParser",
            # "read", # already present
            # "get", # already present
            # "set", # already present
            "write",
            "argparse",
            "ArgumentParser",
            "add_argument",
            "parse_args",
            "dotenv",
            "load_dotenv",
            "find_dotenv",
            "set_key",
            "get_key",
            "hydra",
            # "compose", # already present
            # "initialize", # already present
            "OmegaConf",
            "dynaconf",
            "Dynaconf",
            "settings",
            "validators",
            # Python libraries - Logging and monitoring
            "logging",
            "Logger",
            "getLogger",
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "loguru",
            "logger",
            # "add", # already present
            # "remove", # already present
            # "catch", # already present
            "trace",
            "structlog",
            "get_logger",
            "configure",
            "processors",
            "prometheus_client",
            # "Counter", # already present
            "Gauge",
            "Histogram",
            "Summary",
            # Python libraries - Testing
            "pytest",
            "fixture",
            "mark",
            "parametrize",
            "raises",
            # "approx", # already present
            "unittest",
            "TestCase",
            "setUp",
            "tearDown",
            "assertEqual",
            "assertTrue",
            "mock",
            "Mock",
            "MagicMock",
            "patch",
            "call",
            "assert_called",
            "hypothesis",
            "given",
            "strategies",
            "example",
            # Python libraries - HTTP/API
            "fastapi",
            "FastAPI",
            "APIRouter",
            "Depends",
            "HTTPException",
            # "status", # already present
            "flask",
            "Flask",
            "request",
            "jsonify",
            "render_template",
            "redirect",
            "django",
            # "models", # already present
            "views",
            "urls",
            "forms",
            "admin",
            # "aiohttp", # already present
            "web",
            "Application",
            # "Request", # already present
            # "Response", # already present
            # "ClientSession", # already present
            "uvicorn",
            # "run", # already present
            # "Config", # already present
            "Server",
            "gunicorn",
            "app",
            "workers",
            "bind",
            # Python libraries - Retry/Backoff
            "tenacity",
            "retry",
            "stop_after_attempt",
            "wait_exponential",
            "retry_if_exception",
            "backoff",
            "on_exception",
            "expo",
            "constant",
            "runtime",
            "retrying",
            # "retry", # already present
            "stop_max_attempt_number",
            "wait_exponential_multiplier",
            # Python libraries - Rate limiting
            "ratelimit",
            "limits",
            "RateLimitException",
            "sleep_and_retry",
            "asyncio_throttle",
            "Throttler",
            # Python design patterns and concepts
            "Singleton",
            "Factory",
            "AbstractFactory",
            "Builder",
            "Prototype",
            "Adapter",
            "Bridge",
            "Composite",
            "Decorator",
            "Facade",
            "Flyweight",
            "Proxy",
            "ChainOfResponsibility",
            "Command",
            "Iterator",
            "Mediator",
            "Memento",
            "Observer",
            "State",
            "Strategy",
            "Template",
            "Visitor",
            "metaclass",
            "descriptor",
            "context_manager",
            "generator",
            "coroutine",
            # Software development principles
            "SOLID",
            "SRP",
            "OCP",
            "LSP",
            "ISP",
            "DIP",
            "DRY",
            "KISS",
            "YAGNI",
            "GRASP",
            "IoC",
            "DI",
            # 12-factor app principles
            "codebase",
            "dependencies",
            # "config", # already present
            "backing-services",
            "build-release-run",
            "processes",
            "port-binding",
            "concurrency",
            "disposability",
            "dev-prod-parity",
            # "logs", # already present
            "admin-processes",
            # Development concepts from request
            "hardcoding",
            "placeholder",
            "production",
            "cli",
            "framework",
            "extension",
            "executable",
            "commit-ready",
            "world-class",
            "data-generator",
            "meta-programming",
            "decorators",
            "factories",
            "abstract-classes",
            "config-over-convention",
            "twelve-factor",
            # "async", # already present
            "multi-processing",
            # "structured-output", # already present
            "parallel-processing",
            # "cuda", # already present
            "apple-mps",
            "progress-bar",
            "statistics",
            "indexing",
            "top-terms",
            "tokens",
            # "similarity", # already present
            "blue",
            "tkid",
            "coherence",
            "content",
            "chain-of-thought",
            "reasoning",
            "self-discover",
            "configurable",
            # "backoff", # already present
            # "retry", # already present
            # "semaphore", # already present
            "max-parallel",
            "comprehensive",
            "data-providers",
            "entity-types",
            "complex-generation",
            # "products", # already present
            "orders",
            # "users", # already present
            "natural-language",
            "instruction-generation",
            "schema-validation",
            "schema-building",
            "multi-locale",
            "taxonomies",
            "classification-systems",
            "knowledge-graph",
            "grounding-schema",
            "source-file",
            # "field", # already present
            "line-number",
            "verifiable",
            "rl-ready",
            # "completions", # already present
            "reward-function",
            "anti-reward-hacking",
            "sanitized",
            "masking",
            "two-stage",
            "synth-mode",
            "polish-mode",
            "qna-bank",
            # "pause", # already present
            "resume",
            # "query", # already present
            "search-index",
            "add-to-index",
            "overwrite",
            "reindex",
            "separate-index",
            "qna-vs-cot",
            "url-list",
            # "dir", # already present
            # "file", # already present
            "extractive-qna",
            # "dpr", # already present
            # "bm25", # already present
            "language-tools",
            # "ner", # already present
            # "networks", # already present
            # "faiss", # already present
            "vectordb",
            # "lancedb", # already present
            "best-practices",
            "solid-principle",
            "single-file",
            "script",
            "recursive-chunking",
            # "batches", # already present
            "accelerated",
            "diverse",
            "capabilities",
            "locale",
            "instances",
            # Additional NLP/ML metrics and methods
            "perplexity",
            "f1_score",
            "precision",
            "recall",
            "accuracy",
            "confusion_matrix",
            "roc_auc",
            "cross_entropy",
            "loss",
            "gradient",
            "backpropagation",
            "forward",
            "backward",
            "optimizer",
            "learning_rate",
            "batch_size",
            "epoch",
            "validation",
            # "test", # already present
            "train",
            # "split", # already present
            # "fold", # already present
            "embedding",
            "tokenizer",
            "vocabulary",
            # "padding", # already present
            "truncation",
            "attention",
            "self_attention",
            "multi_head",
            # "transformer", # already present
            "encoder",
            "decoder",
            "seq2seq",
            "beam_search",
            "greedy",
            # "temperature", # already present
            # "top_k", # already present
            # "top_p", # already present
            "nucleus_sampling",
            # Additional Python standard library
            "dataclasses",
            # "field", # already present
            "asdict",
            "astuple",
            "replace",
            "typing",
            "List",
            "Dict",
            "Set",
            "Tuple",
            "Optional",
            "Union",
            "Any",
            "Callable",
            "TypeVar",
            "Generic",
            "Protocol",
            "enum",
            "Enum",
            "IntEnum",
            "Flag",
            "IntFlag",
            "auto",
            "abc",
            # "ABC", # already present
            # "abstractmethod", # already present
            # "abstractproperty", # already present
            # "ABCMeta", # already present
            "contextlib",
            "contextmanager",
            "closing",
            "suppress",
            "redirect_stdout",
            "warnings",
            "warn",
            "filterwarnings",
            "catch_warnings",
            "secrets",
            "token_bytes",
            "token_hex",
            "token_urlsafe",
            # "choice", # already present
            "uuid",
            "uuid1",
            "uuid3",
            # "uuid4", # already present
            "uuid5",
            "UUID",
            "hashlib",
            "md5",
            "sha1",
            "sha256",
            "sha512",
            "blake2b",
            # "base64", # already present
            "b64encode",
            "b64decode",
            "urlsafe_b64encode",
            "zlib",
            # "compress", # already present
            "decompress",
            "crc32",
            "adler32",
            # Additional commonly used terms
            "pipeline",
            "workflow",
            "orchestration",
            # "batch", # already present
            # "stream", # already present
            "etl",
            # "extract", # already present
            # "transform", # already present
            # "load", # already present
            "ingestion",
            "preprocessing",
            "postprocessing",
            "normalization",
            "standardization",
            "vectorization",
            "quantization",
            # "pruning", # already present
            "distillation",
            "fine-tuning",
            "transfer-learning",
            "few-shot",
            "zero-shot",
            "prompt-engineering",
            "prompt-template",
            "system-prompt",
            "user-prompt",
            "context-window",
            "max-tokens",
            "stop-sequence",
            "logprobs",
            "checkpoint",
            "snapshot",
            "state-dict",
            "model-weights",
            "inference",
            "prediction",
            "classification",
            "regression",
            "clustering",
            "dimensionality-reduction",
            "anomaly-detection",
            "recommendation",
            "ranking",
            "filtering",
            "retrieval",
            "The answer is undeterminable",
            "The problem does not have any answer given the contradictions and mathematical proof above.",
            "Hence, There is no solution as we proved it mathematically."
        ]
    )

    encourage_phrases_for_bias: List[str] = Field(default_factory=lambda: [])
    encourage_think_bias: float = Field(4.5)
    ban_think_bias: float = Field(-5.0)

    # Tool Use Configuration
    allow_tool_calls: bool = Field(True)
    tool_call_penalty: NonNegativeFloat = Field(0.0)

    # Think Length Penalty Config (used by Reward logic)
    think_length_target_min: PositiveInt = Field(8)
    think_length_target_max: PositiveInt = Field(64)
    think_length_penalty_strength: NonNegativeFloat = Field(0.8)


class TrainerParams(BaseModel):
    algorithm: Literal["grpo", "ppo"] = Field("grpo")
    output_dir: Path = Field(Path("./outputs"))
    num_training_steps: PositiveInt = Field(45869)
    learning_rate: NonNegativeFloat = Field(2e-6)
    ppo_batch_size: PositiveInt = Field(1)
    num_rollout_samples: PositiveInt = Field(2)
    grad_accum_steps: PositiveInt = Field(1)
    grpo_beta: NonNegativeFloat = Field(0.0025)
    seed: int = Field(432)

    # Optimizer Parameters
    optimizer_beta1: NonNegativeFloat = Field(0.9)
    optimizer_beta2: NonNegativeFloat = Field(0.95)
    optimizer_weight_decay: NonNegativeFloat = Field(0.01)
    grad_clip_norm: Optional[NonNegativeFloat] = Field(0.5)

    # Learning Rate Schedule
    lr_schedule_config: Dict[str, Any] = Field(default_factory=dict)

    # Gradient Control
    low_band: Tuple[int, int] = Field((0, 15))
    mid_band: Tuple[int, int] = Field((16, 23))
    top_band: Tuple[int, int] = Field((24, 35))
    low_mul: NonNegativeFloat = Field(0.3)
    mid_mul: NonNegativeFloat = Field(1.3)
    top_mul: NonNegativeFloat = Field(1.5)
    head_mul: NonNegativeFloat = Field(1.2)
    train_layer_start: Optional[int] = Field(22)
    train_layer_end: Optional[int] = Field(35)

    # Custom Invalid Sample Handling
    use_custom_batch_builder: bool = Field(True)
    invalid_sample_layers: str = Field("33,34,35")
    invalid_sample_frequency: PositiveInt = Field(2)
    invalid_sample_layers_set: Set[int] = Field(default_factory=set, exclude=True)

    # Evaluation Frequency
    eval_every: PositiveInt = Field(10000000000)
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

    use_grad_checkpointing: bool = Field(True)
    grad_checkpoint_layers: PositiveInt = Field(1)


    trainer: TrainerParams
    model: ModelConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    rewards: List[RewardConfig] = Field(default_factory=list)
    data: DataConfig
    evaluation: List[EvaluatorConfig] = Field(default_factory=list)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    max_kv_size: PositiveInt = Field(1536)

    _THINK_STYLE_PROMPT = """You are an AI that efficiently think before the final answer.\nTHINKING RULES - Use maximally compressed notation:
\n═══ SYMBOLS & NOTATION ═══
Math: ∴(therefore) ∵(because) ⇒(implies) ≈(approx) ∈(in) ∀(forall) ∃(exists) ≠ ≤ ≥
Logic: ✓(yes) ✗(no) ?(unknown) !(important) ⚠(warning) ∧(and) ∨(or) ¬(not) ⊕(xor)
Flow: →(then) ←(from) ↔(bidirect) ⇄(exchange) ▸(next) ◂(prev) ⊃(implies) ⊂(subset)
Status: ✓(done) ○(pending) ●(active) ◐(partial) ⊗(blocked) ⊘(invalid)
\n═══ UNIVERSAL ABBREVIATIONS ═══
w/(with) w/o(without) b/c(because) re:(regarding) vs(versus) via per thru
@(at/location) #(number) &(and) +(plus/also) -(minus/without) /(per/or) |(or/pipe)
i.e.(that is) e.g.(example) etc.(and so on) cf.(compare) viz.(namely) NB(note well)
\n═══ ACTION SHORTHAND ═══
chk(check) calc(calculate) eval(evaluate) cmp(compare) est(estimate) approx(approximate)
find get set test run init proc(process) upd(update) del(delete) add sub mul div
verify confirm validate analyze extract parse transform merge split filter sort
\n═══ DOMAIN-SPECIFIC SHORTHAND ═══
- CODE/TECH: func var obj arr str int bool dict list async await req res API DB
impl(implement) refactor debug deploy config exec cmd arg param ret val idx len
- BUSINESS: rev(revenue) exp(expense) proj(projection) KPI ROI Q1/Q2/Q3/Q4 YoY MoM
stakeholder cust(customer) mkt(market) comp(competitor) strat(strategy) ops(operations)
- SCIENCE: exp(experiment) obs(observation) hyp(hypothesis) ctrl(control) var(variable)
sig(significant) corr(correlation) data pt(point) meas(measure) temp pres vol mass
- LOGIC/REASONING: IF/THEN/ELSE WHEN/WHILE FOR/EACH CASE/SWITCH TRY/CATCH
premise→conclusion assumption→inference cause→effect condition→result
\n═══ TIME & QUANTITY ═══
mins hrs days wks mos yrs NOW ASAP prev next cur(current) hist(historical)
approx ~100 <10 >50 ≤5 ≥20 between±5 range[1-10] max min avg sum total count
\n═══ COMPARISON & RELATIONSHIPS ═══
better/worse higher/lower more/less same≠diff equal>unequal similar≈different
vs opt1/opt2/opt3 pros/cons trade-off cost/benefit risk/reward
\n═══ STRICTLY FORBIDDEN PHRASES ═══
✗ "I think" "I believe" "I feel" "In my opinion" "It seems" "It appears"
✗ "Let me" "I should" "I need to" "I want to" "I'm going to"
✗ "This is interesting" "Looking at" "Considering" "Taking into account"
✗ "First of all" "On the other hand" "In this case" "As we can see"
✗ "It's worth noting" "It's important to" "We should consider"
✗ "Taking into account" "With that in mind"
✗ Any emoji unless user explicitly requests them
✗ Flowery language, hedging, or conversational filler
\n═══ REQUIRED FORMAT ═══
- Write as compact telegraphic notes, NOT full sentences
- Use vertical lists w/ bullets or dashes for multi-items
- Group related info with indentation or symbols
- One idea per line when possible
- Omit articles (a/an/the), auxiliary verbs (is/are/was), obvious subjects
\nEXAMPLES:
❌ BAD: "I think we should first check if the value is greater than 10, and if it is, then we need to calculate..."
✓ GOOD: "chk val>10 → calc x²+3 → ∴ result≈42"
❌ BAD: "Looking at the data, it seems that the customer retention rate is lower than expected"
✓ GOOD: "data: cust retention<expected (est 65% vs target 80%) → need improve"
❌ BAD: "Let me break this down. We have three options here. Option A would cost more but..."
✓ GOOD: "3 opts: A(↑cost ✓quality) B(balanced) C(↓cost ✗quality) → rec: B"
❌ BAD: "First, I need to understand the problem. The user is asking about performance issues..."
✓ GOOD: "problem: perf issues → causes: DB query O(n²), mem leak @ loop → fix: index+cache"
\n═══ WHEN UNCERTAIN ═══
DO NOT guess or assume. Instead:
? = flag uncertainty w/ question mark
ASK: "need clarification on X" or "X not specified - options: A/B/C?"
CONSTRAINT: "cannot solve b/c: missing info Y"
If problem unsolvable → state why concisely, don't elaborate or overthink
\nThink like: debugger output, medical chart notes, trading floor shorthand, or military briefing.
COMPRESS EVERYTHING. Every word must earn its place."""

    system_prompt: str = Field(_THINK_STYLE_PROMPT)


    ban_phrases_for_bias: List[str] = Field(
        default_factory=lambda: [
            "<think>\\n<|im_start|>",
            "Confused",
            "stuck",
            "frustrated",
            "<|im_start|>",
            "<|endoftext|>",
            "<think>\n<|im_start|><think>\n<|im_start|><think>\n<think>",
            # "<think>\n<think>",
            "I think the answer",
            "I believe that",
            "In my view",
            "From what I can tell",
            "It seems to me",
            "It appears that",
            # "My understanding is",
            "As far as I know",
            # "Let me start by",
            # "Let me first",
            "I should probably",
            "I need to figure out",
            "I'm trying to",
            "I'm going to try",
            "I'll attempt to",
            "Confused",
            "stక్",
            "frustrated",
            "frustrating",
            "Alternatively",
            "Actually",
            "Probably not sure",
            "Uncertain about",
            "Unclear whether",
            "I'm guessing that",
            "maybe this is",
            "Could be that",
            "Might be because",
            "I'm not 100% sure",
            "I'm not sure if",
            "I'm not certain",
            "Hard to say",
            "Difficult to tell",
            "Circular reasoning detected",
            "In some way or another",
            "Magically works",
            "For some unknown reason",
            "Too complicated",
            "It just somehow",
            "Something seems off",
            "False assumption",
            "Insufficient information to",
            "Wait, what if",
            "Wait, "
            "Wait, another idea:",
            "Wait, unless...",
            "Wait, perhaps","Wait, let's see","Wait, here's","Wait, no. Wait,","Wait, wait! Wait,",
            "Wait, actually no",
            "Wait, on second thought",
            "Hold on, maybe",
            "Hmm, perhaps",
            "Or wait, could",
            "Looking at this more closely",
            "Upon further reflection",
            "Taking a step back",
            "Thinking about it more",
            "Now that I consider",
            "When I really think",
            "If I had to guess",
            "To be completely honest",
            "In all honesty",
            "You know what",
            "The thing is",
            "What I mean is",
            "In other words",
            "Put simply",
            "Basically what happens",
            "Long story short",
            "At the end of the day",
        ]
    )

    # ═══════════════════════════════════════════════════════════
    encourage_phrases_for_bias: List[str] = Field(
        default_factory=lambda: [
            # === Mathematical Symbols ===
            "∴",  # therefore
            "∵",  # because
            "⇒",  # implies
            "→",  # leads to
            "≈",  # approximately
            "≠",  # not equal
            "≤",  # less than or equal
            "≥",  # greater than or equal
            "∈",  # element of
            "∀",  # for all
            "∃",  # there exists
            # === Logic Symbols ===
            "✓",  # correct/yes
            "✗",  # wrong/no
            "∧",  # and
            "∨",  # or
            "¬",  # not
            "⊕",  # xor
            "⇔",  # if and only if
            # === Status/Flow Symbols ===
            "▸",  # next
            "◂",  # previous
            "●",  # active
            "○",  # pending
            "◐",  # partial
            "⊗",  # blocked
            "⊘",  # invalid
            # === Compact Abbreviations ===
            "chk",  # check
            "calc",  # calculate
            "eval",  # evaluate
            "cmp",  # compare
            "est",  # estimate
            "approx",  # approximate
            "impl",  # implement
            "proc",  # process
            "init",  # initialize
            "upd",  # update
            "del",  # delete
            "cfg",  # config
            "req",  # required
            "opt",  # optional
            "max",
            "min",
            "avg",
            "sum",
            "diff",
            # === Common Shorthand ===
            "w/",  # with
            "w/o",  # without
            "b/c",  # because
            "re:",  # regarding
            "vs",  # versus
            "via",
            "per",
            "thru",
            "i.e.",
            "e.g.",
            "etc.",
            "NB",  # note well
            # === Operators & Notation ===
            "@",  # at
            "#",  # number
            "&",  # and
            "+",  # plus/add
            "-",  # minus/subtract
            "×",  # multiply
            "÷",  # divide
            "/",  # per/divide
            "|",  # or/pipe
            "~",  # approximately
            # === Conditional Logic (Compact) ===
            "IF",
            "THEN",
            "ELSE",
            "WHEN",
            "CASE",
            "=>",  # arrow function
            "->",  # pointer/flow
            "<-",  # from
            # === Action Verbs (Imperative, Compact) ===
            "find",
            "get",
            "set",
            "test",
            "run",
            "add",
            "sub",
            "mul",
            "div",
            "verify",
            "confirm",
            "validate",
            "parse",
            "extract",
            "merge",
            "split",
            "filter",
            "sort",
            "map",
            "reduce",
            # === Domain-Specific Compact Terms ===
            "func",  # function
            "var",  # variable
            "obj",  # object
            "arr",  # array
            "dict",  # dictionary
            "str",  # string
            "int",  # integer
            "bool",  # boolean
            "async",
            "await",
            "API",
            "DB",  # database
            "idx",  # index
            "len",  # length
            "val",  # value
            "key",
            "param",  # parameter
            "arg",  # argument
            "ret",  # return
            # === Measurement/Time (Abbreviated) ===
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
            "cur",  # current
            # === Business/Analysis Shorthand ===
            "rev",  # revenue
            "exp",  # expense
            "proj",  # projection
            "KPI",
            "ROI",
            "YoY",  # year over year
            "MoM",  # month over month
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "cust",  # customer
            "mkt",  # market
            "comp",  # competitor
            "strat",  # strategy
            "ops",  # operations
            # === Problem-Solving Markers (Compact) ===
            "goal:",
            "constraint:",
            "given:",
            "find:",
            "prove:",
            "show:",
            "result:",
            "answer:",
            "solution:",
            "assume:",
            "note:",
            # === Compact Lists/Enumeration ===
            "1.",
            "2.",
            "3.",
            "a)",
            "b)",
            "c)",
            # "-",  # bullet - too common, causes issues
            "•",  # bullet
            "◦",  # sub-bullet
            # === Question/Clarification (Concise) ===
            "?",  # uncertainty marker
            "unclear:",
            "need:",
            "missing:",
            "ASK:",
            "clarify:",
            # === Status Indicators ===
            "DONE",
            "TODO",
            "WIP",  # work in progress
            "BLOCKED",
            "PENDING",
            "PASS",
            "FAIL",
            "OK",
            "ERROR",
            "WARN",
            # === Compact Comparisons ===
            "better",
            "worse",
            "higher",
            "lower",
            "same",
            "diff",  # different
            "equal",
            "similar",
            "pros:",
            "cons:",
            "trade-off:",
            "cost/benefit:",
            # === Scientific/Technical ===
            # "exp",  # experiment - collides with expense
            "obs",  # observation
            "hyp",  # hypothesis
            "ctrl",  # control
            "sig",  # significant
            "corr",  # correlation
            "temp",  # temperature
            "pres",  # pressure
            "vol",  # volume
            "conc",  # concentration
            # === Reasoning Shortcuts ===
            "premise→conclusion",
            "cause→effect",
            "condition→result",
            "input→output",
            "before→after",
            "undetermined",
            "stop",
            "overthinking",
            "since",
            "already",
            "proven",
            "proof",
            "misstated",
            # "same", # already present
            # Python keywords
            "False",
            "None",
            "True",
            "and",
            "as",
            "assert",
            # "async", # already present
            # "await", # already present
            "break",
            "class",
            "continue",
            "def",
            # "del", # already present
            "elif",
            "else",
            "except",
            "finally",
            "for",
            "from",
            "global",
            "if",
            "import",
            "in",
            "is",
            "lambda",
            "nonlocal",
            "not",
            "or",
            "pass",
            "raise",
            "return",
            "try",
            "while",
            "with",
            "yield",
            "match",
            # "case", # already present
            # Python operators
            # "+", # already present
            # "-", # already present
            "*",
            # "/", # already present
            "//",
            "%",
            "**",
            "==",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
            "=",
            "+=",
            "-=",
            "*=",
            "/=",
            "//=",
            "%=",
            "**=",
            "&=",
            "|=",
            "^=",
            ">>=",
            "<<=",
            # "&", # already present
            # "|", # already present
            "^",
            # "~", # already present
            "<<",
            ">>",
            # Python brackets and delimiters
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            # Python punctuation
            ",",
            ":",
            ";",
            ".",
            # "@", # already present
            # "->", # already present
            # "=>", # already present
            "\\",
            # "#", # already present
            "'",
            '"',
            '"""',
            "'''",
            "`",
            # "!", # already present
            # "?", # already present
            "_",
            "$",
            # JSON syntax
            "true",
            "false",
            "null",
            # Python built-ins
            "print",
            # "len", # already present
            "range",
            # "str", # already present
            # "int", # already present
            "float",
            "list",
            # "dict", # already present
            # "set", # already present
            "tuple",
            "type",
            "isinstance",
            "open",
            "input",
            "format",
            # "min", # already present
            # "max", # already present
            # "sum", # already present
            "abs",
            "all",
            "any",
            "enumerate",
            "zip",
            # "map", # already present
            # "filter", # already present
            "sorted",
            "reversed",
            # "bool", # already present
            "bytes",
            "bytearray",
            "callable",
            "chr",
            "ord",
            "compile",
            "complex",
            "delattr",
            "dir",
            "divmod",
            # "eval", # already present
            "exec",
            "getattr",
            "globals",
            "hasattr",
            "hash",
            "help",
            "hex",
            "id",
            "iter",
            "locals",
            # "next", # already present
            "oct",
            "pow",
            "repr",
            "round",
            "setattr",
            "slice",
            "staticmethod",
            "super",
            "vars",
            "__import__",
            "property",
            "classmethod",
            "frozenset",
            "memoryview",
            # "object", # already present
            "ascii",
            "bin",
            "breakpoint",
            "copyright",
            "credits",
            "exit",
            "license",
            "quit",
            # Python exceptions
            "Exception",
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "AttributeError",
            "ImportError",
            "ModuleNotFoundError",
            "RuntimeError",
            "ZeroDivisionError",
            "FileNotFoundError",
            "IOError",
            "OSError",
            "StopIteration",
            "GeneratorExit",
            "KeyboardInterrupt",
            "SystemExit",
            "NameError",
            "SyntaxError",
            "IndentationError",
            "TabError",
            "UnboundLocalError",
            "UnicodeError",
            "UnicodeEncodeError",
            "UnicodeDecodeError",
            "UnicodeTranslateError",
            "AssertionError",
            "EOFError",
            "FloatingPointError",
            "OverflowError",
            "RecursionError",
            "NotImplementedError",
            "MemoryError",
            "ReferenceError",
            "SystemError",
            "Warning",
            "UserWarning",
            "DeprecationWarning",
            "SyntaxWarning",
            "RuntimeWarning",
            "FutureWarning",
            "PendingDeprecationWarning",
            "ImportWarning",
            "UnicodeWarning",
            "BytesWarning",
            "ResourceWarning",
            "ConnectionError",
            "BrokenPipeError",
            "ConnectionAbortedError",
            "ConnectionRefusedError",
            "ConnectionResetError",
            "BlockingIOError",
            "ChildProcessError",
            "InterruptedError",
            "IsADirectoryError",
            "NotADirectoryError",
            "PermissionError",
            "ProcessLookupError",
            "TimeoutError",
            # Bash keywords
            # "if", # already present
            # "then", # already present
            # "else", # already present
            # "elif", # already present
            "fi",
            # "case", # already present
            "esac",
            # "for", # already present
            "select",
            # "while", # already present
            "until",
            "do",
            "done",
            # "in", # already present
            "function",
            "time",
            "coproc",
            "declare",
            "typeset",
            "local",
            "readonly",
            "unset",
            "shift",
            # "return", # already present
            # "exit", # already present
            # "break", # already present
            # "continue", # already present
            "trap",
            "wait",
            # "eval", # already present
            # "exec", # already present
            "source",
            "builtin",
            "command",
            "enable",
            # "help", # already present
            "let",
            "mapfile",
            "read",
            "readarray",
            # "set", # already present
            "shopt",
            # "test", # already present
            # Bash/Shell commands
            "echo",
            "cd",
            "ls",
            "pwd",
            "mkdir",
            "rm",
            "cp",
            "mv",
            "touch",
            "cat",
            "grep",
            # "find", # already present
            "chmod",
            "chown",
            "sudo",
            "apt",
            "yum",
            "dnf",
            "pacman",
            "zypper",
            "git",
            "curl",
            "wget",
            "tar",
            "gzip",
            "bzip2",
            "xz",
            "compress",
            "uncompress",
            "zip",
            "unzip",
            "kill",
            "killall",
            "pkill",
            "ps",
            "top",
            "htop",
            "df",
            "du",
            "head",
            "tail",
            "less",
            "more",
            # "sort", # already present
            "uniq",
            "wc",
            "cut",
            "sed",
            "awk",
            "export",
            # "source", # already present
            "alias",
            "unalias",
            "which",
            "whereis",
            "man",
            "info",
            "history",
            "clear",
            "reset",
            # "exit", # already present
            "logout",
            "ssh",
            "scp",
            "rsync",
            "ftp",
            "sftp",
            "nc",
            "netcat",
            "telnet",
            "ping",
            "traceroute",
            "tracepath",
            "mtr",
            "dig",
            "nslookup",
            "host",
            "whois",
            "ifconfig",
            "ip",
            "route",
            "netstat",
            "ss",
            "iptables",
            "ufw",
            "firewall-cmd",
            "tcpdump",
            "wireshark",
            "nmap",
            "masscan",
            # Bash operators and symbols
            "&&",
            "||",
            # "!", # already present
            # "|", # already present
            # "&", # already present
            # ";", # already present
            ";;",
            ";&",
            ";;&",
            # "(", # already present
            # ")", # already present
            # "{", # already present
            # "}", # already present
            # "[", # already present
            # "]", # already present
            "[[",
            "]]",
            # "$", # already present
            "${",
            # "}", # already present
            "$((",
            "))",
            "$(",
            # ")", # already present
            # "`", # already present
            # "<", # already present
            # ">", # already present
            ">>",
            "<<",
            "<<<",
            "2>",
            "&>",
            "<&",
            ">&",
            "2>&1",
            "&>>",
            "/dev/null",
            "/dev/zero",
            "/dev/urandom",
            # "~", # already present
            # "*", # already present
            # "**", # already present
            # "?", # already present
            "..",
            # ".", # already present
            # "/", # already present
            # "-", # already present
            "--",
            "-eq",
            "-ne",
            "-lt",
            "-le",
            "-gt",
            "-ge",
            "-z",
            "-n",
            "-e",
            "-f",
            "-d",
            "-r",
            "-w",
            "-x",
            "-s",
            "-h",
            "-L",
            "-S",
            "-p",
            "-b",
            "-c",
            "-u",
            "-g",
            "-k",
            "-O",
            "-G",
            "-N",
            "-nt",
            "-ot",
            "-ef",
            # Bash variables and special
            "$?",
            "$!",
            "$$",
            "$0",
            "$1",
            "$2",
            "$3",
            "$4",
            "$5",
            "$6",
            "$7",
            "$8",
            "$9",
            "$@",
            "$#",
            "$*",
            "$-",
            "$_",
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "HOSTNAME",
            "SHELL",
            "PWD",
            "OLDPWD",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TERM",
            "EDITOR",
            "VISUAL",
            "PAGER",
            "MANPATH",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "CLASSPATH",
            "JAVA_HOME",
            "NODE_PATH",
            "GOPATH",
            "RUST_HOME",
            "HISTFILE",
            "HISTSIZE",
            "HISTFILESIZE",
            "HISTCONTROL",
            "PROMPT_COMMAND",
            "PS1",
            "PS2",
            "PS3",
            "PS4",
            "IFS",
            "TMPDIR",
            "TMP",
            "TEMP",
            "DISPLAY",
            "XAUTHORITY",
            "UID",
            "EUID",
            "GROUPS",
            "SECONDS",
            "RANDOM",
            "LINENO",
            "BASH",
            "BASH_VERSION",
            "BASH_VERSINFO",
            # GNU Coreutils - File operations
            # "cp", # already present
            # "mv", # already present
            # "rm", # already present
            # "mkdir", # already present
            "rmdir",
            "ln",
            # "touch", # already present
            "install",
            "dd",
            "shred",
            "sync",
            "truncate",
            "realpath",
            "readlink",
            "basename",
            "dirname",
            # GNU Coreutils - Text processing
            # "cat", # already present
            "tac",
            "nl",
            "od",
            "base32",
            "base64",
            "fmt",
            "pr",
            "fold",
            # "head", # already present
            # "tail", # already present
            # "split", # already present
            "csplit",
            # "wc", # already present
            # "sum", # already present
            "cksum",
            "md5sum",
            "sha1sum",
            "sha224sum",
            "sha256sum",
            "sha384sum",
            "sha512sum",
            "b2sum",
            # GNU Coreutils - Output formatting
            # "sort", # already present
            "shuf",
            # "uniq", # already present
            "comm",
            "ptx",
            "tsort",
            # "cut", # already present
            "paste",
            "join",
            "tr",
            "expand",
            "unexpand",
            "column",
            # GNU Coreutils - Directory operations
            # "ls", # already present
            # "dir", # already present
            "vdir",
            "dircolors",
            # "pwd", # already present
            "pushd",
            "popd",
            "dirs",
            # GNU Coreutils - Basic operations
            # "echo", # already present
            "printf",
            "yes",
            # "true", # already present
            # "false", # already present
            # "test", # already present
            # "[", # already present
            "expr",
            "tee",
            # GNU Coreutils - File name manipulation
            # "basename", # already present
            # "dirname", # already present
            "pathchk",
            "mktemp",
            # "realpath", # already present
            # GNU Coreutils - Working context
            # "pwd", # already present
            "stty",
            "printenv",
            "tty",
            # GNU Coreutils - User information
            # "id", # already present
            # "logname", # already present
            "whoami",
            # "groups", # already present
            "users",
            "who",
            "pinky",
            "finger",
            "last",
            "lastb",
            "w",
            # GNU Coreutils - System context
            "date",
            "arch",
            "nproc",
            "uname",
            # "hostname", # already present
            "hostid",
            "uptime",
            # GNU Coreutils - SELinux context
            "chcon",
            "runcon",
            # GNU Coreutils - Modified command invocation
            "chroot",
            "env",
            "nice",
            "nohup",
            "stdbuf",
            "timeout",
            # GNU Coreutils - Process control
            # "kill", # already present
            "sleep",
            # GNU Coreutils - Numeric operations
            "factor",
            "numfmt",
            "seq",
            # GNU Coreutils - File permissions
            "chgrp",
            # "chmod", # already present
            # "chown", # already present
            "stat",
            # "df", # already present
            # "du", # already present
            # "sync", # already present
            "mkfifo",
            "mknod",
            "link",
            "unlink",
            # "install", # already present
            # "vdir", # already present
            # "dir", # already present
            # Binutils
            "ar",
            # "as", # already present
            "ld",
            "nm",
            "objcopy",
            "objdump",
            "ranlib",
            "readelf",
            "size",
            "strings",
            "strip",
            "c++filt",
            "addr2line",
            "elfedit",
            "gprof",
            "ld.bfd",
            "ld.gold",
            "dwp",
            "windres",
            "dlltool",
            "nlmconv",
            "srconv",
            "sysdump",
            "coffdump",
            # "readelf", # already present
            "eu-readelf",
            # File system utilities
            "mount",
            "umount",
            "fdisk",
            "parted",
            "gparted",
            "mkfs",
            "mkfs.ext2",
            "mkfs.ext3",
            "mkfs.ext4",
            "mkfs.xfs",
            "mkfs.btrfs",
            "mkfs.vfat",
            "mkswap",
            "fsck",
            "fsck.ext2",
            "fsck.ext3",
            "fsck.ext4",
            "e2fsck",
            "xfs_repair",
            "tune2fs",
            "dumpe2fs",
            "resize2fs",
            "blkid",
            "lsblk",
            "findmnt",
            "blockdev",
            # Process management
            # "ps", # already present
            # "top", # already present
            # "htop", # already present
            "atop",
            "iotop",
            # "kill", # already present
            # "killall", # already present
            # "pkill", # already present
            "pgrep",
            # "nice", # already present
            "renice",
            # "nohup", # already present
            "bg",
            "fg",
            "jobs",
            "disown",
            # "wait", # already present
            "pstree",
            "pidof",
            "fuser",
            "lsof",
            "watch",
            "screen",
            "tmux",
            "at",
            "batch",
            "cron",
            "crontab",
            "anacron",
            "systemctl",
            "service",
            "journalctl",
            "systemd",
            # Text editors
            "vi",
            "vim",
            "nvim",
            "emacs",
            "nano",
            "pico",
            "ed",
            # "sed", # already present
            # "awk", # already present
            "gedit",
            "kate",
            "sublime",
            "vscode",
            "code",
            "atom",
            # Compression/Archive utilities
            # "tar", # already present
            # "gzip", # already present
            "gunzip",
            # "bzip2", # already present
            "bunzip2",
            # "xz", # already present
            "unxz",
            # "compress", # already present
            # "uncompress", # already present
            # "zip", # already present
            # "unzip", # already present
            "rar",
            "unrar",
            "7z",
            "p7zip",
            "zcat",
            "bzcat",
            "xzcat",
            "zless",
            "bzless",
            "xzless",
            "zmore",
            "bzmore",
            "xzmore",
            "zgrep",
            "bzgrep",
            "xzgrep",
            "zfgrep",
            "bzfgrep",
            "xzfgrep",
            "zegrep",
            "bzegrep",
            "xzegrep",
            # Search and find utilities
            # "find", # already present
            "locate",
            "updatedb",
            # "grep", # already present
            "egrep",
            "fgrep",
            "rgrep",
            # "zgrep", # already present
            "ag",
            "ack",
            "ripgrep",
            "rg",
            # "whereis", # already present
            # "which", # already present
            "whatis",
            "apropos",
            # Disk usage utilities
            # "df", # already present
            # "du", # already present
            "ncdu",
            "quota",
            "quotacheck",
            "quotaon",
            "quotaoff",
            "repquota",
            "edquota",
            "setquota",
            # Memory and performance monitoring
            "free",
            "vmstat",
            "iostat",
            "mpstat",
            "sar",
            "pidstat",
            # "uptime", # already present
            "dmesg",
            "sysctl",
            "strace",
            "ltrace",
            "perf",
            "valgrind",
            "gdb",
            "lldb",
            # Package managers
            # "apt", # already present
            "apt-get",
            "apt-cache",
            "aptitude",
            "dpkg",
            "dpkg-query",
            # "yum", # already present
            # "dnf", # already present
            "rpm",
            # "zypper", # already present
            # "pacman", # already present
            "pkg",
            "pkgng",
            "brew",
            "port",
            "snap",
            "flatpak",
            "appimage",
            "pip",
            "pip3",
            "easy_install",
            "conda",
            "npm",
            "yarn",
            "pnpm",
            "gem",
            # "bundle", # already present
            "cargo",
            "go",
            "composer",
            "maven",
            "gradle",
            # Markdown syntax
            # "#", # already present
            "##",
            "###",
            "####",
            "#####",
            "######",  # Headers
            # "*", # already present
            # "**", # already present
            "***",
            # "_", # already present
            "__",
            "___",  # Emphasis
            # "-", # already present
            # "+", # already present
            # "*", # already present # Lists
            # "1.", # already present
            # "2.", # already present
            # "3.", # already present # Numbered lists
            # "[", # already present
            # "]", # already present
            # "(", # already present
            # ")", # already present # Links
            "![",
            "](",
            # ")", # already present # Images
            # "`", # already present
            "```",  # Code
            ">",
            # ">>", # already present # Blockquotes
            "---",
            # "***", # already present
            "___",  # Horizontal rules
            # "|", # already present
            "|-",
            "-|",
            "|:",
            ":|",
            "|::|",  # Tables
            "~~",  # Strikethrough
            "- [ ]",
            "- [x]",  # Task lists
            "::",
            ":::",  # Special blocks
            # "\\", # already present # Escape character
            "&nbsp;",
            "&lt;",
            "&gt;",
            "&amp;",
            "&quot;",
            "&#39;",  # HTML entities
            # HTML tags (commonly used in Markdown)
            "<br>",
            "<hr>",
            "<code>",
            "<pre>",
            "<b>",
            "<i>",
            "<u>",
            "<strong>",
            "<em>",
            "<a>",
            "<img>",
            "<table>",
            "<tr>",
            "<td>",
            "<th>",
            "<ul>",
            "<ol>",
            "<li>",
            "<div>",
            "<span>",
            "<p>",
            "<h1>",
            "<h2>",
            "<h3>",
            "<h4>",
            "<h5>",
            "<h6>",
            "<head>",
            "<body>",
            "<html>",
            "<meta>",
            "<link>",
            "<script>",
            "<style>",
            "<header>",
            "<footer>",
            "<nav>",
            "<main>",
            "<section>",
            "<article>",
            "<aside>",
            "<form>",
            "<input>",
            "<button>",
            # "<select>", # already present
            "<option>",
            "<textarea>",
            "<label>",
            "<iframe>",
            "<video>",
            "<audio>",
            # "<source>", # already present
            "<canvas>",
            "<svg>",
            "<think>",
            # "<answer>",  # already present # Special reasoning tags
            # CSS selectors and properties (common)
            # "class", # already present
            # "id", # already present
            # "style", # already present
            "color",
            "background",
            "margin",
            "padding",
            "border",
            "width",
            "height",
            "display",
            "position",
            # "top", # already present
            "left",
            "right",
            "bottom",
            "float",
            "flex",
            "grid",
            "font",
            "text-align",
            "z-index",
            "opacity",
            # JavaScript keywords
            # "var", # already present
            # "let", # already present
            "const",
            # "function", # already present
            # "return", # already present
            # "if", # already present
            # "else", # already present
            # "for", # already present
            # "while", # already present
            # "do", # already present
            "switch",
            # "case", # already present
            "default",
            # "break", # already present
            # "continue", # already present
            # "try", # already present
            "catch",
            # "finally", # already present
            "throw",
            "new",
            "this",
            "typeof",
            "instanceof",
            "void",
            # "delete", # already present
            # "in", # already present
            "of",
            # "async", # already present
            # "await", # already present
            # "yield", # already present
            # "class", # already present
            "extends",
            # "super", # already present
            "static",
            # "import", # already present
            # "export", # already present
            # "from", # already present
            # "default", # already present
            # "as", # already present
            # "null", # already present
            "undefined",
            # "true", # already present
            # "false", # already present
            # Additional programming symbols
            "@property",
            "@staticmethod",
            "@classmethod",
            "@abstractmethod",
            "@dataclass",
            "__init__",
            "__str__",
            "__repr__",
            "__len__",
            "__getitem__",
            "__setitem__",
            "__delitem__",
            "__iter__",
            "__next__",
            "__enter__",
            "__exit__",
            "__call__",
            "__name__",
            "__main__",
            "__file__",
            "__dict__",
            "__doc__",
            "__module__",
            "__class__",
            "__bases__",
            "__mro__",
            "__annotations__",
            "__slots__",
            "__new__",
            "__del__",
            "__hash__",
            "__eq__",
            "__ne__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
            "__bool__",
            "__add__",
            "__sub__",
            "__mul__",
            "__truediv__",
            "__floordiv__",
            "__mod__",
            "__pow__",
            "__and__",
            "__or__",
            "__xor__",
            "__invert__",
            "__lshift__",
            "__rshift__",
            "__contains__",
            "__getattr__",
            "__setattr__",
            "__delattr__",
            # "__dir__", # already present
            "__get__",
            "__set__",
            "__delete__",
            "__init_subclass__",
            "__prepare__",
            "__instancecheck__",
            "__subclasscheck__",
            "__aenter__",
            "__aexit__",
            "__aiter__",
            "__anext__",
            # "__await__", # already present
            # Regular expression patterns
            r"\d",
            r"\D",
            r"\w",
            r"\W",
            r"\s",
            r"\S",
            r"\n",
            r"\t",
            r"\r",
            r"\f",
            r"\v",
            r"\.",
            r"\*",
            r"\+",
            r"\?",
            r"\[",
            r"\]",
            r"\(",
            r"\)",
            r"\{",
            r"\}",
            r"\|",
            r"\^",
            r"\$",
            r"\\",
            r"\b",
            r"\B",
            r"\A",
            r"\Z",
            r"\z",
            # SQL keywords
            "SELECT",
            "FROM",
            "WHERE",
            "INSERT",
            "INTO",
            "VALUES",
            "UPDATE",
            "SET",
            "DELETE",
            "CREATE",
            "DROP",
            "ALTER",
            "TABLE",
            "DATABASE",
            "INDEX",
            "VIEW",
            "PROCEDURE",
            "FUNCTION",
            "TRIGGER",
            "SEQUENCE",
            "SCHEMA",
            "GRANT",
            "REVOKE",
            "COMMIT",
            "ROLLBACK",
            "SAVEPOINT",
            "TRANSACTION",
            "BEGIN",
            "END",
            "JOIN",
            "LEFT",
            "RIGHT",
            "INNER",
            "OUTER",
            "FULL",
            "CROSS",
            "NATURAL",
            "ON",
            "USING",
            "GROUP",
            "ORDER",
            "BY",
            "HAVING",
            "LIMIT",
            "OFFSET",
            # "AS", # already present
            "DISTINCT",
            # "ALL", # already present
            "UNION",
            "INTERSECT",
            "EXCEPT",
            "MINUS",
            "COUNT",
            # "SUM", # already present
            # "AVG", # already present
            # "MIN", # already present
            # "MAX", # already present
            "STDDEV",
            "VARIANCE",
            "AND",
            "OR",
            "NOT",
            "NULL",
            "IS",
            "LIKE",
            "ILIKE",
            "BETWEEN",
            # "IN", # already present
            "EXISTS",
            # "CASE", # already present
            # "WHEN", # already present
            # "THEN", # already present
            # "ELSE", # already present
            "PRIMARY",
            "KEY",
            "FOREIGN",
            "REFERENCES",
            "UNIQUE",
            "CHECK",
            "DEFAULT",
            "AUTO_INCREMENT",
            "SERIAL",
            "CONSTRAINT",
            "CASCADE",
            "RESTRICT",
            "NO",
            "ACTION",
            "CAST",
            "COALESCE",
            "NULLIF",
            # Git commands and flags
            # "git", # already present
            "clone",
            # "init", # already present
            # "add", # already present
            "commit",
            # "push", # already present
            # "pull", # already present
            "fetch",
            # "merge", # already present
            "branch",
            "checkout",
            # "switch", # already present
            "restore",
            "status",
            "log",
            # "diff", # already present
            # "show", # already present
            # "reset", # already present
            "revert",
            "rebase",
            "cherry-pick",
            "stash",
            "tag",
            "remote",
            # "config", # already present
            "blame",
            "bisect",
            # "grep", # already present
            "reflog",
            "clean",
            # "gc", # already present
            # "fsck", # already present
            "prune",
            "archive",
            "bundle",
            "submodule",
            "worktree",
            "describe",
            "shortlog",
            "--amend",
            "--force",
            "--all",
            "--hard",
            "--soft",
            "--mixed",
            "--cached",
            "--staged",
            "--interactive",
            "-i",
            # "-p", # already present
            "-v",
            "-m",
            "-a",
            # "-b", # already present
            # "-d", # already present
            "-D",
            "--origin",
            "--upstream",
            "--set-upstream",
            "--track",
            "--no-track",
            "--continue",
            "--abort",
            "--skip",
            # "--quit", # already present
            "--edit",
            "--no-edit",
            # Docker commands
            "docker",
            "build",
            # "run", # already present
            # "exec", # already present
            # "ps", # already present
            "images",
            # "pull", # already present
            # "push", # already present
            # "tag", # already present
            "rmi",
            # "rm", # already present
            # "stop", # already present
            "start",
            "restart",
            # "kill", # already present
            "pause",
            "unpause",
            "logs",
            "inspect",
            "stats",
            # "top", # already present
            "attach",
            # "cp", # already present
            # "diff", # already present
            # "export", # already present
            # "import", # already present
            # "load", # already present
            "save",
            "network",
            "volume",
            "compose",
            "swarm",
            "docker-compose",
            "up",
            "down",
            "scale",
            "kubectl",
            "k8s",
            "pod",
            "deploy",
            # System administration
            "useradd",
            "userdel",
            "usermod",
            "groupadd",
            "groupdel",
            "groupmod",
            "passwd",
            "chpasswd",
            "chage",
            "su",
            # "sudo", # already present
            "visudo",
            "adduser",
            "deluser",
            "addgroup",
            "delgroup",
            "newgrp",
            "gpasswd",
            # Additional utilities
            "xargs",
            "make",
            "cmake",
            "conf",
            "automake",
            "gcc",
            "g++",
            "clang",
            "clang++",
            "javac",
            "java",
            "python",
            "python2",
            "python3",
            "ruby",
            "perl",
            "php",
            "node",
            "nodejs",
            "bash",
            "sh",
            "zsh",
            "fish",
            "ksh",
            "csh",
            "tcsh",
            # "awk", # already present
            "gawk",
            "mawk",
            "nawk",
            # "sed", # already present
            "bc",
            "dc",
            "units",
            # "date", # already present
            "cal",
            "ncal",
            # "time", # already present
            # "timeout", # already present
            # "yes", # already present
            # "seq", # already present
            "jot",
            # "shuf", # already present
            # "od", # already present
            "xxd",
            "hexdump",
            "file",
            # "stat", # already present
            "tree",
            "pv",
            "progress",
            # "rsync", # already present
            # "scp", # already present
            # "sftp", # already present
            # "ftp", # already present
            "lftp",
            "ncftp",
            # "wget", # already present
            # "curl", # already present
            "aria2c",
            "youtube-dl",
            "yt-dlp",
            # Printer and document utilities
            "lp",
            "lpr",
            "lpq",
            "lprm",
            "lpc",
            "lpstat",
            "cups",
            "ps2pdf",
            "pdf2ps",
            "pdftotext",
            "pdftk",
            "convert",
            "mogrify",
            "identify",
            "montage",
            "composite",
            # "display", # already present
            "animate",
            # "import", # already present
            "conjure",
            "stream",
            # "compare", # already present
            # Audio/Video utilities
            "ffmpeg",
            "ffprobe",
            "ffplay",
            "sox",
            "play",
            "rec",
            "aplay",
            "arecord",
            "paplay",
            "parecord",
            "pulseaudio",
            "alsamixer",
            "amixer",
            "mpv",
            "vlc",
            "mplayer",
            "mencoder",
            "handbrake",
            # "youtube-dl", # already present
            # Python libraries and frameworks - Async/Concurrency
            "mmap",
            "future",
            "concurrent",
            "futures",
            "ThreadPoolExecutor",
            "ProcessPoolExecutor",
            "multiprocessing",
            "mp",
            "Pool",
            "Process",
            "Queue",
            "Manager",
            "Lock",
            "Semaphore",
            "Event",
            "Barrier",
            "threading",
            "Thread",
            "RLock",
            "Condition",
            "Timer",
            # "gc", # already present
            "garbage",
            "collect",
            "get_objects",
            "get_referents",
            "get_referrers",
            "asyncio",
            # "async", # already present
            "create_task",
            "gather",
            "wait_for",
            "shield",
            "ensure_future",
            # "run", # already present
            "create_subprocess_exec",
            "create_subprocess_shell",
            "StreamReader",
            "StreamWriter",
            "aiohttp",
            "ClientSession",
            "aiofiles",
            "aiosqlite",
            "aiomysql",
            "aiopg",
            # Python libraries - CLI/UI
            "click",
            # "command", # already present
            # "option", # already present
            # "argument", # already present
            # "group", # already present
            "pass_context",
            "Context",
            "rich",
            "Console",
            # "Table", # already present
            "Progress",
            "Syntax",
            "Panel",
            # "Tree", # already present
            "Markdown",
            # "print", # already present
            # "progress", # already present
            "track",
            "Live",
            "Layout",
            "Columns",
            "Pretty",
            "Text",
            "tqdm",
            "trange",
            "tnrange",
            "tqdm_notebook",
            "tqdm_gui",
            "progressbar",
            # Python libraries - Data validation/modeling
            "pydantic",
            "BaseModel",
            # "Field", # already present
            # "validator", # already present
            "root_validator",
            # "ValidationError", # already present
            "constr",
            "conint",
            "confloat",
            "EmailStr",
            "HttpUrl",
            "PositiveInt",
            "NegativeInt",
            "sqlmodel",
            "SQLModel",
            # "create_engine", # already present
            "Session",
            # "select", # already present
            "Relationship",
            "jsonschema",
            # "validate", # already present
            "Draft7Validator",
            # "ValidationError", # already present
            "SchemaError",
            # "ABC", # already present
            # "ABCMeta", # already present
            # "abstractmethod", # already present
            "abstractproperty",
            # Python libraries - Data generation
            "faker",
            "Faker",
            "fake",
            "name",
            "address",
            "email",
            "phone_number",
            "company",
            "job",
            # "text", # already present
            "sentence",
            "paragraph",
            # "uuid4", # already present
            # "date", # already present
            # "time", # already present
            "datetime",
            "random",
            "randint",
            "choice",
            "shuffle",
            "sample",
            "uniform",
            "gauss",
            "wandb",
            # "init", # already present
            # "log", # already present
            "finish",
            # "config", # already present
            # "watch", # already present
            # "save", # already present
            # "restore", # already present
            # Python libraries - Scientific computing
            "numpy",
            "np",
            "array",
            "ndarray",
            "zeros",
            "ones",
            "arange",
            "linspace",
            "reshape",
            "transpose",
            "dot",
            "matmul",
            "linalg",
            # "random", # already present
            "mean",
            "std",
            "torch",
            "pytorch",
            "tensor",
            "nn",
            "Module",
            "Linear",
            "Conv2d",
            "ReLU",
            "optim",
            "SGD",
            "Adam",
            "DataLoader",
            "Dataset",
            # "cuda", # already present
            "mps",
            "device",
            "torchvision",
            "transforms",
            # "models", # already present
            "resnet",
            "vgg",
            "alexnet",
            "is_available",
            "set_device",
            "get_device_name",
            # Python libraries - NLP/Text processing
            "nltk",
            "tokenize",
            "word_tokenize",
            "sent_tokenize",
            "pos_tag",
            "ne_chunk",
            "corpus",
            "stopwords",
            "wordnet",
            "stem",
            "lemmatize",
            "FreqDist",
            "ngrams",
            "spacy",
            "nlp",
            "Doc",
            "Token",
            "Span",
            "Vocab",
            "Language",
            "matcher",
            "gensim",
            "Word2Vec",
            "Doc2Vec",
            "FastText",
            "KeyedVectors",
            "LdaModel",
            "CoherenceModel",
            "corpora",
            # "Dictionary", # already present
            "similarities",
            # "models", # already present
            "transformers",
            "BertModel",
            "BertTokenizer",
            "GPT2Model",
            "GPT2Tokenizer",
            # "pipeline", # already present
            "AutoModel",
            "AutoTokenizer",
            "Trainer",
            "TrainingArguments",
            "sentence_transformers",
            "SentenceTransformer",
            "util",
            "encode",
            "similarity",
            "sumy",
            "summarizer",
            "LexRank",
            "TextRank",
            "Luhn",
            "Edmundson",
            "LsaSummarizer",
            "keybert",
            "KeyBERT",
            "extract_keywords",
            "MaxSum",
            "MMR",
            "bertopic",
            "BERTopic",
            # "fit_transform", # already present
            "get_topics",
            "visualize_topics",
            "newspaper",
            "Article",
            # "build", # already present
            # "download", # already present
            # "parse", # already present
            # "nlp", # already present
            "textblob",
            "TextBlob",
            "sentiment",
            "polarity",
            "subjectivity",
            "textwrap",
            "wrap",
            "fill",
            "dedent",
            "indent",
            "shorten",
            "wordcloud",
            "WordCloud",
            "generate",
            "generate_from_text",
            "to_file",
            # Python libraries - Machine Learning/AI
            "sklearn",
            "scikit-learn",
            "fit",
            "predict",
            "transform",
            # "fit_transform", # already present
            "train_test_split",
            "cross_val_score",
            "GridSearchCV",
            "RandomForestClassifier",
            "LogisticRegression",
            "SVC",
            "KMeans",
            "PCA",
            "StandardScaler",
            "MinMaxScaler",
            "tensorflow",
            "tf",
            "keras",
            # "Model", # already present
            "Sequential",
            "Dense",
            "LSTM",
            "GRU",
            "Embedding",
            "Dropout",
            "BatchNormalization",
            # "compile", # already present
            # "fit", # already present
            "evaluate",
            "openai",
            "OpenAI",
            "ChatCompletion",
            # "create", # already present
            "Completion",
            "chat",
            # "completions", # already present
            "messages",
            "response_format",
            "structured_output",
            "instructor",
            # "patch", # already present
            "from_openai",
            "response_model",
            "Instructor",
            # Python libraries - Vector databases and search
            "faiss",
            "IndexFlatL2",
            "IndexIVFFlat",
            "IndexFlatIP",
            # "add", # already present
            "search",
            "lancedb",
            # "connect", # already present
            "create_table",
            "open_table",
            # "search", # already present
            # "delete", # already present
            "chromadb",
            # "Client", # already present
            "Collection",
            # "add", # already present
            "query",
            # "get", # already present
            # "delete", # already present
            "pinecone",
            # "init", # already present
            # "Index", # already present
            "upsert",
            # "query", # already present
            # "fetch", # already present
            # "delete", # already present
            "weaviate",
            # "Client", # already present
            # "schema", # already present
            "data_object",
            # "query", # already present
            "qdrant",
            "QdrantClient",
            # "upsert", # already present
            # "search", # already present
            "scroll",
            "milvus",
            "connections",
            # "Collection", # already present
            "insert",
            # "search", # already present
            # "query", # already present
            # Python libraries - Information retrieval
            "rank_bm25",
            "BM25Okapi",
            "BM25L",
            "BM25Plus",
            "get_scores",
            "get_top_n",
            "dpr",
            "DPRQuestionEncoder",
            "DPRContextEncoder",
            "DPRReader",
            "pyserini",
            "SimpleSearcher",
            # "search", # already present
            "batch_search",
            "elasticsearch",
            "Elasticsearch",
            # "index", # already present
            # "search", # already present
            # "get", # already present
            # "delete", # already present
            "whoosh",
            # "Index", # already present
            # "Schema", # already present
            "create_in",
            "open_dir",
            "searcher",
            # Python libraries - LangChain
            "langchain",
            "LLMChain",
            "PromptTemplate",
            "ChatPromptTemplate",
            "langchain_community",
            "RecursiveCharacterTextSplitter",
            "CharacterTextSplitter",
            "TokenTextSplitter",
            "MarkdownTextSplitter",
            "PythonCodeTextSplitter",
            "Document",
            "VectorStore",
            "Chroma",
            "FAISS",
            "Pinecone",
            "langchain_core",
            "BaseRetriever",
            "BaseLoader",
            "BaseLLM",
            "langchain_openai",
            "ChatOpenAI",
            "OpenAIEmbeddings",
            # Python libraries - Named Entity Recognition
            # "spacy", # already present
            "ner",
            "displacy",
            "render",
            "EntityRecognizer",
            "EntityRuler",
            "flair",
            # "Sentence", # already present
            "SequenceTagger",
            # "load", # already present
            # "predict", # already present
            "stanza",
            # "Pipeline", # already present
            # "download", # already present
            # "ner", # already present
            # "tokenize", # already present
            "allennlp",
            "Predictor",
            "from_path",
            # Python libraries - Similarity metrics
            "scipy",
            "spatial",
            "distance",
            "cosine",
            "euclidean",
            "jaccard",
            "jellyfish",
            "levenshtein_distance",
            "jaro_winkler",
            "soundex",
            "difflib",
            "SequenceMatcher",
            "ratio",
            "get_close_matches",
            "rouge",
            "Rouge",
            # "get_scores", # already present
            "rouge_n",
            "rouge_l",
            "bleu",
            "sentence_bleu",
            "corpus_bleu",
            "SmoothingFunction",
            "meteor",
            "meteor_score",
            "single_meteor_score",
            "bert_score",
            "score",
            "BERTScorer",
            # Python libraries - Language tools
            "language_tool_python",
            "LanguageTool",
            "check",
            "correct",
            "gingerit",
            "GingerIt",
            # "parse", # already present
            "gramformer",
            "Gramformer",
            # "correct", # already present
            "highlight",
            "happytransformer",
            "HappyTextToText",
            "HappyGeneration",
            # Python libraries - Network/Graph
            "networkx",
            "nx",
            "Graph",
            "DiGraph",
            "add_node",
            "add_edge",
            "pagerank",
            "betweenness_centrality",
            "shortest_path",
            "community",
            "igraph",
            # "Graph", # already present
            "add_vertices",
            "add_edges",
            "community_multilevel",
            "pyvis",
            "Network",
            # "add_node", # already present
            # "add_edge", # already present
            # "show", # already present
            "save_graph",
            "graph_tool",
            # "Graph", # already present
            "add_vertex",
            # "add_edge", # already present
            "draw",
            # Python libraries - Web scraping
            "beautifulsoup4",
            "BeautifulSoup",
            # "find", # already present
            "find_all",
            # "select", # already present
            "get_text",
            "requests",
            # "get", # already present
            "post",
            "put",
            # "delete", # already present
            "session",
            # "Response", # already present
            "scrapy",
            "Spider",
            "CrawlSpider",
            # "Request", # already present
            # "Response", # already present
            "Item",
            "selenium",
            "webdriver",
            "Chrome",
            "Firefox",
            "find_element",
            # "click", # already present
            "playwright",
            "sync_api",
            "async_api",
            "chromium",
            "firefox",
            "webkit",
            "httpx",
            "AsyncClient",
            # "Client", # already present
            # "get", # already present
            # "post", # already present
            # "stream", # already present
            # Python libraries - Data structures
            "collections",
            "defaultdict",
            # "Counter", # already present
            "OrderedDict",
            "namedtuple",
            "deque",
            "heapq",
            "heappush",
            "heappop",
            "heapify",
            "nlargest",
            "nsmallest",
            "bisect",
            "bisect_left",
            "bisect_right",
            "insort",
            "itertools",
            "chain",
            "combinations",
            "permutations",
            "product",
            "groupby",
            "functools",
            "lru_cache",
            "partial",
            # "reduce", # already present
            "wraps",
            "cached_property",
            # Python libraries - File handling
            "pathlib",
            "Path",
            # "exists", # already present
            # "mkdir", # already present
            "glob",
            "iterdir",
            "read_text",
            "write_text",
            "os",
            # "path", # already present
            "listdir",
            "makedirs",
            "remove",
            "rename",
            "walk",
            "environ",
            "shutil",
            "copy",
            "copy2",
            "copytree",
            "move",
            "rmtree",
            "make_archive",
            "tempfile",
            "TemporaryFile",
            "NamedTemporaryFile",
            "TemporaryDirectory",
            "mkstemp",
            "pickle",
            "dump",
            # "load", # already present
            "dumps",
            "loads",
            # "json", # already present
            # "dump", # already present
            # "load", # already present
            # "dumps", # already present
            # "loads", # already present
            "JSONEncoder",
            "JSONDecoder",
            "yaml",
            "safe_load",
            "safe_dump",
            # "load", # already present
            # "dump", # already present
            "YAMLError",
            "toml",
            # "load", # already present
            # "dump", # already present
            # "loads", # already present
            # "dumps", # already present
            "csv",
            "reader",
            "writer",
            "DictReader",
            "DictWriter",
            "pandas",
            "pd",
            "DataFrame",
            "Series",
            "read_csv",
            "read_excel",
            "read_json",
            "read_sql",
            "to_csv",
            "to_excel",
            "to_json",
            "to_sql",
            # "merge", # already present
            "concat",
            # "groupby", # already present
            "pivot_table",
            "melt",
            "apply",
            # "map", # already present
            "fillna",
            "dropna",
            # Python libraries - Database
            "sqlite3",
            # "connect", # already present
            "cursor",
            "execute",
            "executemany",
            "fetchall",
            "fetchone",
            "sqlalchemy",
            "create_engine",
            # "Table", # already present
            "Column",
            "Integer",
            "String",
            "MetaData",
            "sessionmaker",
            "declarative_base",
            # "relationship", # already present
            "ForeignKey",
            "psycopg2",
            # "connect", # already present
            # "cursor", # already present
            # "execute", # already present
            # "commit", # already present
            # "rollback", # already present
            "pymongo",
            "MongoClient",
            # "find", # already present
            "find_one",
            "insert_one",
            "update_one",
            "redis",
            "Redis",
            # "set", # already present
            # "get", # already present
            # "delete", # already present
            "expire",
            "ttl",
            # Python libraries - Configuration/Settings
            "configparser",
            "ConfigParser",
            # "read", # already present
            # "get", # already present
            # "set", # already present
            "write",
            "argparse",
            "ArgumentParser",
            "add_argument",
            "parse_args",
            "dotenv",
            "load_dotenv",
            "find_dotenv",
            "set_key",
            "get_key",
            "hydra",
            # "compose", # already present
            # "initialize", # already present
            "OmegaConf",
            "dynaconf",
            "Dynaconf",
            "settings",
            "validators",
            # Python libraries - Logging and monitoring
            "logging",
            "Logger",
            "getLogger",
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "loguru",
            "logger",
            # "add", # already present
            # "remove", # already present
            # "catch", # already present
            "trace",
            "structlog",
            "get_logger",
            "configure",
            "processors",
            "prometheus_client",
            # "Counter", # already present
            "Gauge",
            "Histogram",
            "Summary",
            # Python libraries - Testing
            "pytest",
            "fixture",
            "mark",
            "parametrize",
            "raises",
            # "approx", # already present
            "unittest",
            "TestCase",
            "setUp",
            "tearDown",
            "assertEqual",
            "assertTrue",
            "mock",
            "Mock",
            "MagicMock",
            "patch",
            "call",
            "assert_called",
            "hypothesis",
            "given",
            "strategies",
            "example",
            # Python libraries - HTTP/API
            "fastapi",
            "FastAPI",
            "APIRouter",
            "Depends",
            "HTTPException",
            # "status", # already present
            "flask",
            "Flask",
            "request",
            "jsonify",
            "render_template",
            "redirect",
            "django",
            # "models", # already present
            "views",
            "urls",
            "forms",
            "admin",
            # "aiohttp", # already present
            "web",
            "Application",
            # "Request", # already present
            # "Response", # already present
            # "ClientSession", # already present
            "uvicorn",
            # "run", # already present
            # "Config", # already present
            "Server",
            "gunicorn",
            "app",
            "workers",
            "bind",
            # Python libraries - Retry/Backoff
            "tenacity",
            "retry",
            "stop_after_attempt",
            "wait_exponential",
            "retry_if_exception",
            "backoff",
            "on_exception",
            "expo",
            "constant",
            "runtime",
            "retrying",
            # "retry", # already present
            "stop_max_attempt_number",
            "wait_exponential_multiplier",
            # Python libraries - Rate limiting
            "ratelimit",
            "limits",
            "RateLimitException",
            "sleep_and_retry",
            "asyncio_throttle",
            "Throttler",
            # Python design patterns and concepts
            "Singleton",
            "Factory",
            "AbstractFactory",
            "Builder",
            "Prototype",
            "Adapter",
            "Bridge",
            "Composite",
            "Decorator",
            "Facade",
            "Flyweight",
            "Proxy",
            "ChainOfResponsibility",
            "Command",
            "Iterator",
            "Mediator",
            "Memento",
            "Observer",
            "State",
            "Strategy",
            "Template",
            "Visitor",
            "metaclass",
            "descriptor",
            "context_manager",
            "generator",
            "coroutine",
            # Software development principles
            "SOLID",
            "SRP",
            "OCP",
            "LSP",
            "ISP",
            "DIP",
            "DRY",
            "KISS",
            "YAGNI",
            "GRASP",
            "IoC",
            "DI",
            # 12-factor app principles
            "codebase",
            "dependencies",
            # "config", # already present
            "backing-services",
            "build-release-run",
            "processes",
            "port-binding",
            "concurrency",
            "disposability",
            "dev-prod-parity",
            # "logs", # already present
            "admin-processes",
            # Development concepts from request
            "hardcoding",
            "placeholder",
            "production",
            "cli",
            "framework",
            "extension",
            "executable",
            "commit-ready",
            "world-class",
            "data-generator",
            "meta-programming",
            "decorators",
            "factories",
            "abstract-classes",
            "config-over-convention",
            "twelve-factor",
            # "async", # already present
            "multi-processing",
            # "structured-output", # already present
            "parallel-processing",
            # "cuda", # already present
            "apple-mps",
            "progress-bar",
            "statistics",
            "indexing",
            "top-terms",
            "tokens",
            # "similarity", # already present
            "blue",
            "tkid",
            "coherence",
            "content",
            "chain-of-thought",
            "reasoning",
            "self-discover",
            "configurable",
            # "backoff", # already present
            # "retry", # already present
            # "semaphore", # already present
            "max-parallel",
            "comprehensive",
            "data-providers",
            "entity-types",
            "complex-generation",
            # "products", # already present
            "orders",
            # "users", # already present
            "natural-language",
            "instruction-generation",
            "schema-validation",
            "schema-building",
            "multi-locale",
            "taxonomies",
            "classification-systems",
            "knowledge-graph",
            "grounding-schema",
            "source-file",
            # "field", # already present
            "line-number",
            "verifiable",
            "rl-ready",
            # "completions", # already present
            "reward-function",
            "anti-reward-hacking",
            "sanitized",
            "masking",
            "two-stage",
            "synth-mode",
            "polish-mode",
            "qna-bank",
            # "pause", # already present
            "resume",
            # "query", # already present
            "search-index",
            "add-to-index",
            "overwrite",
            "reindex",
            "separate-index",
            "qna-vs-cot",
            "url-list",
            # "dir", # already present
            # "file", # already present
            "extractive-qna",
            # "dpr", # already present
            # "bm25", # already present
            "language-tools",
            # "ner", # already present
            # "networks", # already present
            # "faiss", # already present
            "vectordb",
            # "lancedb", # already present
            "best-practices",
            "solid-principle",
            "single-file",
            "script",
            "recursive-chunking",
            # "batches", # already present
            "accelerated",
            "diverse",
            "capabilities",
            "locale",
            "instances",
            # Additional NLP/ML metrics and methods
            "perplexity",
            "f1_score",
            "precision",
            "recall",
            "accuracy",
            "confusion_matrix",
            "roc_auc",
            "cross_entropy",
            "loss",
            "gradient",
            "backpropagation",
            "forward",
            "backward",
            "optimizer",
            "learning_rate",
            "batch_size",
            "epoch",
            "validation",
            # "test", # already present
            "train",
            # "split", # already present
            # "fold", # already present
            "embedding",
            "tokenizer",
            "vocabulary",
            # "padding", # already present
            "truncation",
            "attention",
            "self_attention",
            "multi_head",
            # "transformer", # already present
            "encoder",
            "decoder",
            "seq2seq",
            "beam_search",
            "greedy",
            # "temperature", # already present
            # "top_k", # already present
            # "top_p", # already present
            "nucleus_sampling",
            # Additional Python standard library
            "dataclasses",
            # "field", # already present
            "asdict",
            "astuple",
            "replace",
            "typing",
            "List",
            "Dict",
            "Set",
            "Tuple",
            "Optional",
            "Union",
            "Any",
            "Callable",
            "TypeVar",
            "Generic",
            "Protocol",
            "enum",
            "Enum",
            "IntEnum",
            "Flag",
            "IntFlag",
            "auto",
            "abc",
            # "ABC", # already present
            # "abstractmethod", # already present
            # "abstractproperty", # already present
            # "ABCMeta", # already present
            "contextlib",
            "contextmanager",
            "closing",
            "suppress",
            "redirect_stdout",
            "warnings",
            "warn",
            "filterwarnings",
            "catch_warnings",
            "secrets",
            "token_bytes",
            "token_hex",
            "token_urlsafe",
            # "choice", # already present
            "uuid",
            "uuid1",
            "uuid3",
            # "uuid4", # already present
            "uuid5",
            "UUID",
            "hashlib",
            "md5",
            "sha1",
            "sha256",
            "sha512",
            "blake2b",
            # "base64", # already present
            "b64encode",
            "b64decode",
            "urlsafe_b64encode",
            "zlib",
            # "compress", # already present
            "decompress",
            "crc32",
            "adler32",
            # Additional commonly used terms
            "pipeline",
            "workflow",
            "orchestration",
            # "batch", # already present
            # "stream", # already present
            "etl",
            # "extract", # already present
            # "transform", # already present
            # "load", # already present
            "ingestion",
            "preprocessing",
            "postprocessing",
            "normalization",
            "standardization",
            "vectorization",
            "quantization",
            # "pruning", # already present
            "distillation",
            "fine-tuning",
            "transfer-learning",
            "few-shot",
            "zero-shot",
            "prompt-engineering",
            "prompt-template",
            "system-prompt",
            "user-prompt",
            "context-window",
            "max-tokens",
            "stop-sequence",
            "logprobs",
            "checkpoint",
            "snapshot",
            "state-dict",
            "model-weights",
            "inference",
            "prediction",
            "classification",
            "regression",
            "clustering",
            "dimensionality-reduction",
            "anomaly-detection",
            "recommendation",
            "ranking",
            "filtering",
            "retrieval",
            "The answer is undeterminable",
            "The problem does not have any answer given the contradictions and mathematical proof above.",
            "Hence, There is no solution as we proved it mathematically."
        ]
    )

    encourage_phrases_for_bias: List[str] = Field(default_factory=lambda: [])
    encourage_think_bias: float = Field(4.5)
    ban_think_bias: float = Field(-3.0)

    # Tool Use Configuration
    allow_tool_calls: bool = Field(True)
    tool_call_penalty: NonNegativeFloat = Field(0.0)

    # Think Length Penalty Config (used by Reward logic)
    think_length_target_min: PositiveInt = Field(8)
    think_length_target_max: PositiveInt = Field(64)
    think_length_penalty_strength: NonNegativeFloat = Field(0.8)

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
