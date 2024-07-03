from src.get_models import get_models
from src.load_datasets import load_data_from_json, prepare_datasets, get_data_collator

from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, get_scheduler
import torch

tokenizer, model = get_models()

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
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

gptneo_w_lora = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="results",
    evaluation_strategy="steps",
    num_train_epochs=100,
    per_device_train_batch_size=10,
    gradient_accumulation_steps=1,
    warmup_steps=0,
    learning_rate=1e-4,
    logging_dir="./logs",
    logging_steps=10
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

model_output_dir = "results/model/gpt-neo_1-2b"
trainer.save_model(model_output_dir)  # This saves the model, tokenizer, and args

trainer.train()