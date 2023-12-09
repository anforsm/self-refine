import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd

friendly_name_dict = {
  "gpt-3.5": "GPT-3.5 Turbo",
  "gpt-4": "GPT-4 Turbo",

  "meta-llama/Llama-2-7b-chat-hf": "Llama 2 7B Chat",
  "TheBloke/Llama-2-7B-Chat-fp16": "Llama 2 7B Chat (fp16)",
  "TheBloke/Llama-2-7B-Chat-AWQ": "Llama 2 7B Chat (AWQ)",

  "meta-llama/Llama-2-13b-chat-hf": "Llama 2 13B Chat",
  "TheBloke/Llama-2-13B-Chat-fp16": "Llama 2 13B Chat (fp16)",
  "TheBloke/Llama-2-13B-Chat-AWQ": "Llama 2 13B Chat (AWQ)",

  "codellama/CodeLlama-13b-Instruct-hf": "CodeLlama 13B Instruct",
  "mistralai/Mistral-7B-Instruct-v0.1": "Mistral 7B Instruct",
  "lmsys/vicuna-13b-v1.5": "Vicuna 13B",
}

style_dict = {
    "gpt-4": "ro-",
    "gpt-3.5": "mo-",
    
    "meta-llama/Llama-2-7b-chat-hf": "c*-",
    "TheBloke/Llama-2-7B-Chat-fp16": "c*--",
    "TheBloke/Llama-2-7B-Chat-AWQ": "c*:",

    "meta-llama/Llama-2-13b-chat-hf": "bs-",
    "TheBloke/Llama-2-13B-Chat-fp16": "bs--",
    "TheBloke/Llama-2-13B-Chat-AWQ": "bs:",

    "codellama/CodeLlama-13b-Instruct-hf": "gp-",
    "mistralai/Mistral-7B-Instruct-v0.1": "yp-",
    "lmsys/vicuna-13b-v1.5": "kp-",
}

filename_to_model_name = {}
for model_name in friendly_name_dict.keys():
    filename_to_model_name[f"results/{model_name.replace('/', '_')}_evaluated.json"] = model_name

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
            [0] + list(total_progress_per_iteration.keys()), 
            [50] + list(total_progress_per_iteration.values()),
            style_dict[filename_to_model_name[model_name]],
        )
        
    plt.legend([friendly_name_dict[model_name] for model_name in model_names])
    plt.xlabel("Iteration")
    plt.ylabel("Number of Successful Parses")
    plt.title("Parsability of Model Output")
    
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
            try:
                aspect_per_iteration["0"].append(result["log"][0][f"old_{aspect}"])
            except:
                print(result["id"])
                print(file_name)
                continue
            for it in range(3):
                if not result["log"][it]["failed"]:
                    aspect_per_iteration[str(it + 1)].append(result["log"][it][aspect])
    
        std_per_iteration = {}
        average_per_iteration = {}
        for it in range(4):
            average_per_iteration[str(it)] = sum(aspect_per_iteration[str(it)]) / len(aspect_per_iteration[str(it)])

            std_per_iteration[str(it)] = np.std(aspect_per_iteration[str(it)]) * 0.1
    
        model_to_average[file_name] = average_per_iteration
        model_to_std[file_name] = std_per_iteration

    # create dataframe with results for each model and iteration 0 (base) and iteration 3 (final)
    # do it by first creating a dictionary
    dict_to_df = {
        "model": [],
        "base": [],
        "final": [],
    }
    for model_name, average_per_iteration in model_to_average.items():
        dict_to_df["model"].append(friendly_name_dict[filename_to_model_name[model_name]])
        dict_to_df["base"].append(average_per_iteration["1"])
        dict_to_df["final"].append(average_per_iteration["3"])
    
    # then create the dataframe
    df = pd.DataFrame(dict_to_df)
    # save the dataframe to a csv
    df.to_csv(f"results/{aspect}.csv", index=False)
    
    # plot
    # plt.figure(figsize=(10, 5))
    # for model_name, average_per_iteration in model_to_average.items():
    #     std_per_iteration = model_to_std[model_name]
    #     plt.errorbar(
    #         list(average_per_iteration.keys()), 
    #         list(average_per_iteration.values()),
    #         #yerr=list(std_per_iteration.values()), 
    #         fmt=style_dict[filename_to_model_name[model_name]],
    #         capsize=4,
    #     )

    # plt.legend([friendly_name_dict[model_name] for model_name in model_names], loc="upper left")
    # plt.xlabel("Refinement Iteration")
    # plt.ylabel(f"Average {task_to_friendly_name[aspect]}")
    # plt.ylim(bottom=0)
    # if aspect == "comment_density":
    #     plt.ylim(top=0.4)
    # if aspect == "var_density":
    #     plt.ylim(top=0.8)
    # if aspect == "num_functions":
    #     plt.ylim(top=1)
    # plt.title(f"{task_to_friendly_name[aspect]}")

    # plt.show()

    
    # print the results
    for model_name, average_per_iteration in model_to_average.items():
        std_per_iteration = model_to_std[model_name]
        print(f"Model: {friendly_name_dict[filename_to_model_name[model_name]]}")
        print(f"Final Iteration: {average_per_iteration[str(3)]:.3f}")
        print()


task_to_friendly_name = {
    "comment_density": "Comment Density",
    "var_density": "Density of Appropriately Named Variables",
    "num_functions": "Number of Functions",
}

if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument("--models", nargs="+", type=str)
  args = argparser.parse_args()
  model_list = args.models

  # comment_density
  # var_density
  # num_functions

  model_progress(model_list)
  comparison_plot(model_list, "comment_density")
  comparison_plot(model_list, "var_density")
  comparison_plot(model_list, "num_functions")