import argparse
import json
from collections import defaultdict
from pathlib import Path


TYPE_ORDER = ["ls", "mm", "ms", "re", "sb", "sc", "sd", "tc", "to", "vs"]
# Change this path to summarize any compatible Experiment 2 or Experiment 3 result JSONL.
RESULT_FILE = Path("Results/New_Qwen/New_Experiment_3_with_think.jsonl")
MODEL_LABEL = "Qwen"


def is_correct(record):
    if "correct" in record:
        return bool(record["correct"])
    return record.get("predicted") == record.get("ground_truth")


def load_results(filepath):
    totals = defaultdict(int)
    correct = defaultdict(int)

    with open(filepath, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                result_type = record["id"].split("_", 1)[0]
            except (json.JSONDecodeError, KeyError, AttributeError) as error:
                raise ValueError(
                    f"Invalid result record in {filepath} at line {line_number}"
                ) from error

            totals[result_type] += 1
            correct[result_type] += is_correct(record)

    return totals, correct


def print_summary(totals, correct, model_label):
    result_types = TYPE_ORDER + sorted(set(totals) - set(TYPE_ORDER))
    all_total = 0
    all_correct = 0

    print(f"{'type':<8} {model_label}")
    for result_type in result_types:
        total = totals[result_type]
        correct_count = correct[result_type]
        all_total += total
        all_correct += correct_count
        accuracy = correct_count / total if total else 0.0
        print(f"{result_type:<8} {accuracy:.3f} ({total})")

    all_accuracy = all_correct / all_total if all_total else 0.0
    print(f"{'ALL':<8} {all_accuracy:.3f} ({all_total})")


def main():
    parser = argparse.ArgumentParser(
        description="Print Experiment 2 accuracy by dataset type from a JSONL result file."
    )
    parser.add_argument(
        "result_file",
        nargs="?",
        type=Path,
        default=RESULT_FILE,
        help=f"Path to a result JSONL file (default: {RESULT_FILE})",
    )
    parser.add_argument(
        "--label",
        default=MODEL_LABEL,
        help="Column label to display for the model (default: Qwen)",
    )
    args = parser.parse_args()

    if not args.result_file.is_file():
        parser.error(f"Result file not found: {args.result_file}")

    totals, correct = load_results(args.result_file)
    print("----- EXP 3 Paraphrased Synthetic with Reasoning ------\n")
    print_summary(totals, correct, args.label)


if __name__ == "__main__":
    main()
