import json

name = "results/gpt-4_evaluated.json"

results = json.load(open(name, "r"))

print(len(results))