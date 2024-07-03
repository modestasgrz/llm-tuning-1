from src.get_models import get_models
from src.load_datasets import load_data_from_json, prepare_datasets, get_data_collator
import json

tokenizer, gptneo_w_lora = get_models(saved_model_dir_path="results/gpt-neo_1-2b/checkpoint-500")

def generate_domain_lora(company_description: str) -> str:

    inputs = tokenizer(company_description, return_tensors="pt")
    outputs = gptneo_w_lora.generate(**inputs, max_new_tokens=50)
    plain_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    domains = str.strip(plain_output.split("Suggested domains:")[1]).split(" ")[:3]
    
    return domains

with open("data/companies_dataset_test.json", "r") as f:
    company_descriptions_dict = json.load(f)

for num, company_description in company_descriptions_dict.items():

  domains = generate_domain_lora(company_description)
  company_descriptions_dict[num] = {
      "description": company_description,
      "domains": {}
  }
  for i, domain in enumerate(domains):
    company_descriptions_dict[num]["domains"][str(i + 1)] = domain

print(json.dumps(company_descriptions_dict, indent=4))