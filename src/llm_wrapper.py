from transformers import AutoModelForCausalLM, AutoTokenizer 
from conf import *

model = AutoModelForCausalLM.from_pretrained(MODEL_1)
tokenizer = AutoTokenizer.from_pretrained(MODEL_1)

def call_llm(prompt, model=model, tokenizer=tokenizer):
  max_len = 2048
  if len(prompt) > max_len:
    print(f"Warning: prompt is too long {len(prompt)}, truncating to {max_len}")
  tokens = tokenizer.encode(prompt[-1024:], return_tensors="pt")
  generated_ids = model.generate(tokens, pad_token_id=tokenizer.eos_token_id, max_new_tokens=100)
  output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
  return output[len(prompt):]