import json
from collections import defaultdict

def count_correct_per_category(file_path):
    counts = defaultdict(int)
    totals = defaultdict(int)

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # extract prefix (category) from id
            category = data["id"].split("_")[0]
            
            totals[category] += 1
            if data.get("correct", False):
                counts[category] += 1

    return counts, totals


def print_results(name, counts, totals):
    print(f"\n{name}")
    for cat in sorted(totals.keys()):
        print(f"{cat} → {counts[cat]} / {totals[cat]}")
 

def get_bucket(id_str):
    # extract numeric part (after underscore)
    num = int(id_str.split("_")[1])
    
    if 0 <= num <= 19:
        return "100"
    elif 20 <= num <= 29:
        return "200"
    elif 30 <= num <= 39:
        return "300"
    elif 40 <= num <= 49:
        return "500"
    else:
        return "other"  # safety fallback


def count_correct_per_bucket(file_path):
    counts = defaultdict(int)
    totals = defaultdict(int)

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            bucket = get_bucket(data["id"])
            
            totals[bucket] += 1
            if data.get("correct", False):
                counts[bucket] += 1

    return counts, totals


def print_results2(name, counts, totals):
    print(f"\n{name}")
    for bucket in ["100", "200", "300", "500", "other"]:
        if bucket in totals:
            print(f"{bucket} → {counts[bucket]} / {totals[bucket]}")


# paths
fileslist3 = ["Results/Qwen/Experiment_3.jsonl", "Results/Llama/Experiment_3.jsonl", "Results/Gemma/Experiment_3.jsonl","Results/Gemma3/Experiment_3.jsonl", "Results/Haiku/Experiment_3.jsonl", "Results/Deepseek/Experiment_3.jsonl"]
fileslist2 = ["Results/Qwen/Experiment_2.jsonl", "Results/Llama/Experiment_2.jsonl", "Results/Gemma/Experiment_2.jsonl", "Results/Gemma3/Experiment_2.jsonl", "Results/Haiku/Experiment_2.jsonl", "Results/Deepseek/Experiment_2.jsonl"]

for x in fileslist2:
    # compute
    q_counts, q_totals = count_correct_per_bucket(x)

    # print
    print_results2(x.split("/")[1], q_counts, q_totals)

    # compute
    q_counts, q_totals = count_correct_per_category(x)

    # print
    print_results(x.split("/")[1], q_counts, q_totals)

for x in fileslist3:
    # compute
    q_counts, q_totals = count_correct_per_bucket(x)

    # print
    print_results2(x.split("/")[1], q_counts, q_totals)

    # compute
    q_counts, q_totals = count_correct_per_category(x)

    # print
    print_results(x.split("/")[1], q_counts, q_totals)





