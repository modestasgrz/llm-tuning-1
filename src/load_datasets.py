import json
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, GPT2Tokenizer

from typing import Dict, Any, Tuple, List

def load_data_from_json(file_path):

  with open(file_path, 'r') as f:
      data = json.load(f)

  samples = []

  for entry in data.values():
      description = entry['description']
      domains = ' '.join(entry['domains'].values())
      samples.append(f"{description}\nSuggested domains: {domains}")

  return samples

def prepare_datasets(
    training_samples: Dict[str, Any],
    eval_samples: Dict[str, Any],
    tokenizer: GPT2Tokenizer
) -> Tuple[Dataset, Dataset]:
    
    def _tokenize_function(
        examples: List[str],
    ):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    
    train_dataset = Dataset.from_dict({"text": training_samples})
    tokenized_train_dataset = train_dataset.map(_tokenize_function, batched=True)

    eval_dataset = Dataset.from_dict({"text": eval_samples})
    tokenized_eval_dataset = eval_dataset.map(_tokenize_function, batched=True)

    return tokenized_train_dataset, tokenized_eval_dataset
    

def get_data_collator(
    tokenizer: GPT2Tokenizer
) -> DataCollatorForLanguageModeling:

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    return data_collator

if __name__ == "__main__":

    training_samples = load_data_from_json('data/companies_dataset.json')
    eval_samples = load_data_from_json('data/companies_dataset_eval.json')

    print(training_samples[0])
    print("")
    print(eval_samples[0])