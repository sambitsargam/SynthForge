

import os
from dataclasses import dataclass
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from dataset import EnhancedSFTDataset, SyntheticDataGenerator
from utils.constants import model2template

@dataclass
class LoraTrainingArguments:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: int

def train_lora(
    model_id: str, context_length: int, training_args: LoraTrainingArguments, data_file: str
):
    assert model_id in model2template, f"model_id {model_id} not supported"

    lora_config = LoraConfig(
        r=training_args.lora_rank,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        task_type="CAUSAL_LM",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    training_args = SFTConfig(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=20,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        num_train_epochs=training_args.num_train_epochs,
        max_seq_length=context_length,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=os.environ["HF_TOKEN"],
    )

    synthetic_generator = SyntheticDataGenerator(model_id)
    dataset = EnhancedSFTDataset(
        file=data_file,
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
        synthetic_generator=synthetic_generator,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
    )

    trainer.train()
    trainer.save_model("outputs")
    os.system("rm -rf outputs/checkpoint-*")

    print("Training Completed.")
