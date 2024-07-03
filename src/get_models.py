from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from typing import Tuple

def get_models(
    model_id: str = "EleutherAI/gpt-neo-125M",
    saved_model_dir_path: str = None
) -> Tuple[GPT2Tokenizer, GPTNeoForCausalLM]:
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    if saved_model_dir_path:
        model = GPTNeoForCausalLM.from_pretrained(saved_model_dir_path)
    else:
        model = GPTNeoForCausalLM.from_pretrained(model_id)

    return tokenizer, model

if __name__ == "__main__":

    tokenizer, model = get_models()

    inputs = tokenizer("Hello my name is", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))