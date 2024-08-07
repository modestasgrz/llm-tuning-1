from src.get_models import get_models
from src.load_datasets import load_data_from_json, prepare_datasets, get_data_collator

from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, get_scheduler
import torch

def train(
    model_id: str = "EleutherAI/gpt-neo-125M",
    lora_rank: int = 32,
    lora_alpha: int = 24,
    lora_dropout: float = 0.3,
    num_train_epochs: int = 100,
    output_dir: str = "results",
    per_device_train_batch_size: int = 10,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 0,
    learning_rate: float = 1e-4,
    logging_dir: str = "./logs",
    logging_steps: int = 10,
    push_to_hub: bool = False,
    hub_model_id: str = None,
    hub_token: str = None
) -> float:

    tokenizer, model = get_models(model_id=model_id)

    training_samples = load_data_from_json('data/companies_dataset.json')
    eval_samples = load_data_from_json('data/companies_dataset_eval.json')

    tokenized_train_dataset, tokenized_eval_dataset = prepare_datasets(
        training_samples=training_samples,
        eval_samples=eval_samples,
        tokenizer=tokenizer
    )

    data_collator = get_data_collator(tokenizer)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout
    )

    gptneo_w_lora = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_token=hub_token
    )


    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    trainer = Trainer(
        model=gptneo_w_lora,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None)
    )
    
    trainer.train()
    final_loss = next(entry["eval_loss"] for entry in reversed(trainer.state.log_history) if "eval_loss" in entry)

    return final_loss

if __name__ == "__main__":

    with open("hub_token.txt", "r") as f:
        hub_token = f.read()

    train(
        output_dir="results/gpt-neo_125m",
        push_to_hub=True,
        hub_model_id="modestasgrz/domains-gpt-neo-125m",
        hub_token=hub_token
    )