import json
import matplotlib.pyplot as plt
import argparse
import numpy as np

friendly_name_dict = {
  "gpt-3.5": "GPT-3.5 Turbo",
  "gpt-4": "GPT-4 Turbo",

  "meta-llama/Llama-2-7b-chat-hf": "Llama 2 Chat 7B",
  "TheBloke/Llama-2-7B-Chat-AWQ": "Llama 2 Chat 7B (AWQ)",

  "TheBloke/Llama-2-13B-Chat-AWQ": "Llama 2 Chat 13B (AWQ)",

  "TheBloke/CodeLlama-13B-Instruct-fp16": "CodeLlama 13B Instruct (fp16)",
}


def model_progress(model_names):
    file_names = [f"results/{model_name.replace('/', '_')}_evaluated.json" for model_name in model_names]
    model_to_progress = {}

    for file_name in file_names:
        with open(file_name, "r") as f:
            results = json.load(f)
        
        progress_per_iteration = {
            "1": [],
            "2": [],
            "3": [],
        }

        for result in results:
            for it in range(3):
                progress_per_iteration[str(it + 1)].append(
                    0 if result["log"][it]["failed"] else 1
                )
        
        total_progress_per_iteration = {}
        for it in range(3):
            total_progress_per_iteration[str(it + 1)] = sum(progress_per_iteration[str(it + 1)])
        
        model_to_progress[file_name] = total_progress_per_iteration
    
    # plot
    plt.figure(figsize=(10, 5))
    for model_name, total_progress_per_iteration in model_to_progress.items():
        plt.plot(
            list(total_progress_per_iteration.keys()), 
            list(total_progress_per_iteration.values()),
            "*-",
        )
        
    plt.legend([friendly_name_dict[model_name] for model_name in model_names])
    plt.xlabel("Iteration")
    plt.ylabel("Number of Successful Runs")
    plt.title("Model Progress")
    
    plt.show()


def comparison_plot(model_names, aspect):
    file_names = [f"results/{model_name.replace('/', '_')}_evaluated.json" for model_name in model_names]

    model_to_average = {}
    model_to_std = {}
  
    for file_name in file_names:
        with open(file_name, "r") as f:
            results = json.load(f)
    
        aspect_per_iteration = {
            "0": [],
            "1": [],
            "2": [],
            "3": [],
        }
        for result in results:
            aspect_per_iteration["0"].append(result["log"][0][f"old_{aspect}"])
            for it in range(3):
                if not result["log"][it]["failed"]:
                    aspect_per_iteration[str(it + 1)].append(result["log"][it][aspect])
    
        std_per_iteration = {}
        average_per_iteration = {}
        for it in range(4):
            average_per_iteration[str(it)] = sum(aspect_per_iteration[str(it)]) / len(aspect_per_iteration[str(it)])

            print(file_name)
            print(len(aspect_per_iteration[str(it)]))
            print(np.std(aspect_per_iteration[str(it)]))
            std_per_iteration[str(it)] = np.std(aspect_per_iteration[str(it)]) * 0.1
    
        model_to_average[file_name] = average_per_iteration
        model_to_std[file_name] = std_per_iteration
  
    # plot
    plt.figure(figsize=(10, 5))
    for model_name, average_per_iteration in model_to_average.items():
        std_per_iteration = model_to_std[model_name]
        plt.errorbar(
            list(average_per_iteration.keys()), 
            list(average_per_iteration.values()),
            yerr=list(std_per_iteration.values()), 
            fmt="*-",
            capsize=4,
        )

    plt.legend([friendly_name_dict[model_name] for model_name in model_names])
    plt.xlabel("Iteration")
    plt.ylabel(f"Average {aspect.replace('_', ' ').title()}")
    plt.title(f"{aspect.replace('_', ' ').title()} Comparison")

    plt.show()



if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument("--models", nargs="+", type=str)
  args = argparser.parse_args()
  model_list = args.models

  # comment_density
  # var_density
  # num_functions

  #comparison_plot(model_list, "var_density")
  model_progress(model_list)