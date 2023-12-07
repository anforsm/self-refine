from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch
import re
from conf import *

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_1)
tokenizer = AutoTokenizer.from_pretrained(MODEL_1)

def call_llm(prompt, model=model, tokenizer=tokenizer, pipeline=None, extract_code=False):
  if pipeline is not None:
    #print("--------------")
    #print(prompt)
    #print("-------------")
    #print(type(prompt))
    #print(len(prompt))
    res = pipeline(
      prompt,
      do_sample=True,
      temperature=0.01,
      top_k=5,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
      max_length=len(prompt) + 300,
      pad_token_id=tokenizer.eos_token_id,
    )
    s = res[0]["generated_text"][len(prompt):]
    del res
    if extract_code:
      if "```" in s:
        start = "```"
        result = s[s.find(start) + len(start):s.rfind(start)]
        return result, s
    return s, None
  exit()
  max_len = 4096
  if len(prompt) > max_len:
    print(f"Warning: prompt is too long {len(prompt)}, truncating to {max_len}")
  #print(prompt)
  tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
  #print(tokens.shape)
  generated_ids = model.generate(
    tokens, 
    pad_token_id=tokenizer.eos_token_id, 
    max_new_tokens=2300, 
    temperature=0.7, 
    do_sample=True
  )
  print(generated_ids.shape)
  output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
  print("=======")
  print(output)
  print("=======")
  return output[len(prompt):]
