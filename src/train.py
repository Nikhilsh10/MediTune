import os
import yaml
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config_schema import MediTuneConfig

def load_config(path: str) -> MediTuneConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return MediTuneConfig(**raw)

def setup_wandb(config: MediTuneConfig, run_name: str):
    wandb.init(
        project="meditune-medical-qa",
        name=run_name,
        config={
            "model": config.model.base_model_id,
            "lora_r": config.lora.r,
            "lora_alpha": config.lora.lora_alpha,
            "learning_rate": config.training.learning_rate,
            "epochs": config.training.num_train_epochs,
        }
    )

def load_model_and_tokenizer(config: MediTuneConfig):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.quantization.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config.model.trust_remote_code,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # Required for gradient checkpointing
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def apply_lora(model, config: MediTuneConfig):
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def load_datasets_from_jsonl(config: MediTuneConfig):
    from datasets import load_dataset as hf_load
    # Load from the repo root
    train_ds = hf_load("json", data_files={"train": "data/train.jsonl"}, split="train")
    eval_ds = hf_load("json", data_files={"eval": "data/eval.jsonl"}, split="eval")
    return train_ds, eval_ds

def train(config_path: str = "configs/training_config.yaml"):
    config = load_config(config_path)
    
    # Setup W&B
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        setup_wandb(config, run_name=f"meditune-r{config.lora.r}-lr{config.training.learning_rate}")
    else:
        print("WANDB_API_KEY not found. W&B logging disabled.")
    
    # Load model
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Apply LoRA
    print("Applying LoRA adapters...")
    model = apply_lora(model, config)
    
    # Load data
    print("Loading datasets...")
    train_dataset, eval_dataset = load_datasets_from_jsonl(config)
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_checkpointing=config.training.gradient_checkpointing,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        optim=config.training.optim,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        eval_strategy=config.training.evaluation_strategy,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        group_by_length=config.training.group_by_length,
        packing=config.training.packing,
        report_to=config.training.report_to if "WANDB_API_KEY" in os.environ else "none",
        push_to_hub=config.hub.push_to_hub,
        hub_model_id=config.hub.hub_model_id,
        hub_strategy=config.hub.hub_strategy,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=config.model.max_seq_length,
        dataset_text_field="text",
        dataset_kwargs={"skip_prepare_dataset": False},
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model()
    
    if "WANDB_API_KEY" in os.environ:
        wandb.finish()
    print("Done.")

if __name__ == "__main__":
    train()
