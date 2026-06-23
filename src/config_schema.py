from pydantic import BaseModel, validator
from typing import List, Optional

class ModelConfig(BaseModel):
    base_model_id: str
    max_seq_length: int
    trust_remote_code: bool = False

    @validator("max_seq_length")
    def validate_seq_length(cls, v):
        assert 128 <= v <= 2048, "max_seq_length must be between 128 and 2048"
        return v

class QuantizationConfig(BaseModel):
    load_in_4bit: bool
    bnb_4bit_quant_type: str       # "nf4" or "fp4"
    bnb_4bit_compute_dtype: str    # "bfloat16" or "float16"
    bnb_4bit_use_double_quant: bool

    @validator("bnb_4bit_quant_type")
    def validate_quant_type(cls, v):
        assert v in ["nf4", "fp4"], "quant_type must be 'nf4' or 'fp4'"
        return v

class LoRAConfig(BaseModel):
    r: int
    lora_alpha: int
    target_modules: List[str]
    lora_dropout: float
    bias: str
    task_type: str

    @validator("r")
    def validate_rank(cls, v):
        assert v in [4, 8, 16, 32, 64], "LoRA rank must be a power of 2 between 4 and 64"
        return v

    @validator("bias")
    def validate_bias(cls, v):
        assert v in ["none", "all", "lora_only"]
        return v

class TrainingConfig(BaseModel):
    output_dir: str
    num_train_epochs: float
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

    @validator("learning_rate")
    def validate_lr(cls, v):
        assert 1e-6 <= v <= 1e-2, "Learning rate out of safe range for LoRA"
        return v

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

class MediTuneConfig(BaseModel):
    model: ModelConfig
    quantization: QuantizationConfig
    lora: LoRAConfig
    training: TrainingConfig
    data: DataConfig
    hub: HubConfig
