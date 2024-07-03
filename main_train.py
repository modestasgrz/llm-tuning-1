from src.get_models import get_models
from src.load_datasets import load_data_from_json, prepare_datasets, get_data_collator

from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, get_scheduler
import torch

def train(
    model_id: str = "EleutherAI/gpt-neo-125M",
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    num_train_epochs: int = 100,
    output_dir: str = "results",
    per_device_train_batch_size: int = 10,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 0,
    learning_rate: float = 1e-4,
    logging_dir: str = "./logs",
    logging_steps: int = 10
):

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
        logging_steps=logging_steps
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

    # def get_cosine_scheduler(num_training_steps, optimizer):
    #     return get_scheduler(
    #         name="cosine",
    #         optimizer=optimizer,
    #         num_warmup_steps=0,
    #         num_training_steps=num_training_steps
    #     )

    # trainer.create_scheduler = get_cosine_scheduler
    # trainer.lr_scheduler = get_cosine_scheduler(training_args.num_train_epochs * (len(training_samples) + len(eval_samples)), trainer.optimizer)

    # model_output_dir = "results/model/gpt-neo_1-2b"
    # trainer.save_model(model_output_dir)

    trainer.train()

if __name__ == "__main__":

    train(
        output_dir="results/gpt-neo_1-2b"
    )