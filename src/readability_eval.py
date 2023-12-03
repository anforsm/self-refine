import tokenize
from io import BytesIO
import ast

from openai_wrapper import call_openai
import pandas as pd
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


from readability_prompts import COUNT_VAR_PROMPT, PROMPT_CRITIQUE, PROMPT_FIX


def count_comments(code):
  comment_count = 0
  total_lines = len([l for l in code.splitlines() if l.strip()])

  tokens = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
  for token in tokens:
    if token.type == tokenize.COMMENT:
      comment_count += 1
  return comment_count, comment_count / total_lines

def count_functions(code):
  tree = ast.parse(code)
  return len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])

def count_variables(code):
  prompt = COUNT_VAR_PROMPT.format(code=code)
  result = call_openai(prompt)
  result = result.strip().splitlines()
  num_vars = len(result)
  num_random_vars = len([r for r in result if r.endswith("- random")])
  num_meaningful_vars = num_vars - num_random_vars

  return num_meaningful_vars, num_meaningful_vars / num_vars, result

def run_eval():
  with open("results_long.json", "r") as f:
    results = json.load(f)
  
  remove_indices = []
  for i, result in tqdm(enumerate(results), total=len(results)):
    try:
      old_code = None
      for it in range(len(result["log"])):
        if old_code is None:
          old_code = result["log"][it]["old_code"]
          old_comment_count, old_comment_density = count_comments(old_code)
          old_num_functions = count_functions(old_code)
          old_num_meaningful_vars, old_var_density, old_vars = count_variables(old_code)
    
        new_code = result["log"][it]["new_code"]
        comment_count, comment_density = count_comments(new_code)
        num_functions = count_functions(new_code)
        num_meaningful_vars, var_density, vars = count_variables(new_code)

        result["log"][it].update({
          "comment_count": comment_count,
          "comment_density": comment_density,
          "num_functions": num_functions,
          "num_meaningful_vars": num_meaningful_vars,
          "var_density": var_density,
          "vars": vars,
          "old_comment_count": old_comment_count,
          "old_comment_density": old_comment_density,
          "old_num_functions": old_num_functions,
          "old_num_meaningful_vars": old_num_meaningful_vars,
          "old_var_density": old_var_density,
          "old_vars": old_vars,
        })

        old_code = new_code
        old_comment_count = comment_count
        old_comment_density = comment_density
        old_num_functions = num_functions
        old_num_meaningful_vars = num_meaningful_vars
        old_var_density = var_density
        old_vars = vars
    except:
      remove_indices.append(i)
  
  for i in remove_indices[::-1]:
    results.pop(i)
  
  json.dump(results, open("results_evaluated.json", "w"), indent=2)

def draw_graphs():
  results = json.load(open("results_evaluated.json", "r"))
  comment_ratios = []
  var_ratios = []
  func_nums = []
  comment_ratios.append(sum([r["log"][0]["old_comment_density"] for r in results]) / len(results))
  var_ratios.append(sum([r["log"][0]["old_var_density"] for r in results]) / len(results))
  func_nums.append(sum([r["log"][0]["old_num_functions"] for r in results]) / len(results))
  for it in range(len(results[0]["log"])):
    comment_ratios.append(sum([r["log"][it]["comment_density"] for r in results]) / len(results))
    var_ratios.append(sum([r["log"][it]["var_density"] for r in results]) / len(results))
    func_nums.append(sum([r["log"][it]["num_functions"] for r in results]) / len(results))
  
  plt.plot(comment_ratios)
  plt.plot(var_ratios)
  plt.plot(func_nums)
  plt.legend(["Comment ratio", "Variable ratio", "Function number"])
  plt.show()

def calculate_stats():
  with open("results_evaluated.json", "r") as f:
    results = json.load(f)

  avg_comment_ratio = 0
  avg_func_num = 0
  avg_var_ratio = 0

  old_avg_comment_ratio = 0
  old_avg_func_num = 0
  old_avg_var_ratio = 0

  for i, result in enumerate(results):
    old_avg_comment_ratio += result["log"][0]["old_comment_density"]
    old_avg_func_num += result["log"][0]["old_num_functions"]
    old_avg_var_ratio += result["log"][0]["old_var_density"]
    
    avg_comment_ratio += result["log"][-1]["old_comment_density"]
    avg_func_num += result["log"][-1]["old_num_functions"]
    avg_var_ratio += result["log"][-1]["old_var_density"]
    

  avg_comment_ratio /= len(results)
  avg_func_num /= len(results)
  avg_var_ratio /= len(results)
  
  old_avg_comment_ratio /= len(results)
  old_avg_func_num /= len(results)
  old_avg_var_ratio /= len(results)
  
  print(f"Average comment ratio: {old_avg_comment_ratio} -> {avg_comment_ratio}")
  print(f"Average function number: {old_avg_func_num} -> {avg_func_num}")
  print(f"Average variable ratio: {old_avg_var_ratio} -> {avg_var_ratio}")





def main():
  run_eval()
  calculate_stats()
  draw_graphs()


if __name__ == "__main__":
  main()