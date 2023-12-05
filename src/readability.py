from openai_wrapper import call_openai
from llm_wrapper import call_llm

import pandas as pd
from tqdm import tqdm
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch

from readability_prompts import COUNT_VAR_PROMPT, PROMPT_CRITIQUE, PROMPT_FIX

def self_refine(code, model, tokenizer):
  code = code.replace("\n\n", "\n")
  code_prompt = PROMPT_CRITIQUE.format(code=code)
  #feedback = call_openai(code_prompt)
  feedback = call_llm(code_prompt, model, tokenizer)
  fix_code_prompt = PROMPT_FIX.format(code=code, suggestion=feedback)
  new_code = call_llm(fix_code_prompt, model, tokenizer)
  #new_code = call_openai(fix_code_prompt)
  return feedback, new_code
  

def main(model_name, model, tokenizer):
  programs = pd.read_json("data/code_samples/codenet-python-test-1k.jsonl", lines=True, orient="records")
  results = []
  processed_programs = set()
  num_sampes = len(programs)
  for i, row in tqdm(programs.iterrows(), total=num_sampes):
    id = row["submission_id_v0"]
    if id in processed_programs:
      continue
    if i >= num_sampes:
      break
    

    processed_programs.add(id)
    code = row["input"]

    result = []
    for it in range(3):
      feedback, new_code = self_refine(code, model, tokenizer)

      result.append({
        "old_code": code,
        "feedback": feedback,
        "new_code": new_code,
        "it": it,
      })

      code = new_code

    results.append({
      "id": id,
      "log": result,
    })
  
  json.dump(results, open(f"results/{model_name.replace('/', '_')}results.json", "w"), indent=2)

if __name__ == "__main__":
  # get modle name
  device = "cuda" if torch.cuda.is_available() else "cpu"
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default="EleutherAI/gpt-neo-1.3B")
  args = parser.parse_args()
  model_name = args.model
  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
  model = model.to(device)
  tokenizer = AutoTokenizer.from_pretrained(args.model)
  main(model_name, model, tokenizer)
