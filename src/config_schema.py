"""
Pydantic schema for training_config.yaml.

Validated at load time in train.py — any missing or invalid field
fails immediately before any GPU memory is allocated.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    base_model_id: str
    max_seq_length: int
    trust_remote_code: bool = False

    @field_validator("max_seq_length")
    @classmethod
    def validate_seq_length(cls, v: int) -> int:
        if not (128 <= v <= 2048):
            raise ValueError(f"max_seq_length must be between 128 and 2048, got {v}")
        return v


class QuantizationConfig(BaseModel):
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str
    bnb_4bit_use_double_quant: bool

    @field_validator("bnb_4bit_quant_type")
    @classmethod
    def validate_quant_type(cls, v: str) -> str:
        if v not in ("nf4", "fp4"):
            raise ValueError(f"bnb_4bit_quant_type must be 'nf4' or 'fp4', got '{v}'")
        return v

    @field_validator("bnb_4bit_compute_dtype")
    @classmethod
    def validate_compute_dtype(cls, v: str) -> str:
        if v not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"bnb_4bit_compute_dtype must be bfloat16/float16/float32, got '{v}'")
        return v


class LoRAConfig(BaseModel):
    r: int
    lora_alpha: int
    target_modules: List[str]
    lora_dropout: float
    bias: str
    task_type: str

    @field_validator("r")
    @classmethod
    def validate_rank(cls, v: int) -> int:
        if v not in (4, 8, 16, 32, 64):
            raise ValueError(f"LoRA rank must be one of [4, 8, 16, 32, 64], got {v}")
        return v

    @field_validator("bias")
    @classmethod
    def validate_bias(cls, v: str) -> str:
        if v not in ("none", "all", "lora_only"):
            raise ValueError(f"bias must be 'none', 'all', or 'lora_only', got '{v}'")
        return v

    @field_validator("lora_dropout")
    @classmethod
    def validate_dropout(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError(f"lora_dropout must be in [0, 1), got {v}")
        return v


class TrainingConfig(BaseModel):
    output_dir: str
    num_train_epochs: float
    max_steps: Optional[int] = -1
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    weight_decay: float
    optim: str
    lr_scheduler_type: str
    warmup_ratio: float
    max_grad_norm: float
    logging_steps: int
    save_steps: int
    eval_steps: int
    evaluation_strategy: str
    load_best_model_at_end: bool
    metric_for_best_model: str
    fp16: bool
    bf16: bool
    group_by_length: bool
    packing: bool
    report_to: str

    @field_validator("learning_rate")
    @classmethod
    def validate_lr(cls, v: float) -> float:
        if not (1e-6 <= v <= 1e-2):
            raise ValueError(
                f"Learning rate {v} is outside safe LoRA range [1e-6, 1e-2]. "
                "Typical values: 2e-4 (standard), 1e-4 (conservative)."
            )
        return v

    @field_validator("per_device_train_batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v > 4:
            raise ValueError(
                f"per_device_train_batch_size={v} will OOM on T4. Keep it at 1 or 2."
            )
        return v

    @model_validator(mode="after")
    def check_fp_flags(self) -> "TrainingConfig":
        if self.fp16 and self.bf16:
            raise ValueError("fp16 and bf16 cannot both be True. Use bf16=true for Mistral.")
        return self


class DataConfig(BaseModel):
    dataset_name: str
    train_config: str
    eval_config: str
    train_split: str
    eval_split: str
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


class HubConfig(BaseModel):
    push_to_hub: bool
    hub_model_id: str
    hub_strategy: str


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class MediTuneConfig(BaseModel):
    model: ModelConfig
    quantization: QuantizationConfig
    lora: LoRAConfig
    training: TrainingConfig
    data: DataConfig
    hub: HubConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str = "configs/training_config.yaml") -> MediTuneConfig:
    """Load and validate the training config from YAML.

    Raises:
        pydantic.ValidationError: if any field is missing or violates a constraint.
        FileNotFoundError: if the config file does not exist.
    """
    import yaml  # local import so this module stays importable without pyyaml at test time

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Pydantic will raise a detailed ValidationError if anything is wrong.
    return MediTuneConfig(**raw)