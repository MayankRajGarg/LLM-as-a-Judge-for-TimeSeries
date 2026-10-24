import numpy as np
import json
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from Utils.stats import percent_change, compute_zscore

np.random.seed(42)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_FILE = str(PROJECT_ROOT / "Data" / "synthetic_multianomaly.jsonl")

N_PER_LENGTH = 25  # 25 samples per series length tier = 100 total
SERIES_LENGTHS = [100, 200, 300, 500]

# Minimum spacing between anomalies scales with series length so density is roughly consistent across tiers
MIN_SPACING = {100: 8, 200: 8, 300: 10, 500: 15}

def save_case(case):
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(case) + "\n")

def generate_multi_anomaly(case_id, series_length):
    """
    Linear trend with 1-10 distinct spikes of varying magnitude.
    Spikes are spaced at least MIN_SPACING[series_length] indices apart
    within safe bounds [10, series_length-10] to ensure each anomaly
    is unambiguous and individually identifiable.

    Series length varies across tiers (100, 200, 300, 500) to test
    whether model navigational accuracy degrades with longer input.

    Ground truth: anomaly count + list of {index, percent_change, z_score}.
    Evaluation: exact count accuracy + F1 on detected indices (+-2 tolerance).
    """
    t = np.arange(series_length)
    y = 0.3 * t + np.random.normal(0, 1, series_length)

    n_anomalies = np.random.randint(1, 11)  # 1 to 10 inclusive
    spacing = MIN_SPACING[series_length]

    # Space anomalies at least `spacing` indices apart within safe bounds
    candidates = list(range(10, series_length - 10))
    chosen = []
    for _ in range(n_anomalies):
        valid = [c for c in candidates if all(abs(c - x) >= spacing for x in chosen)]
        if not valid:
            break
        idx = int(np.random.choice(valid))
        chosen.append(idx)

    chosen = sorted(chosen)
    base_std = float(np.std(y))
    anomaly_records = []

    for idx in chosen:
        magnitude = np.random.uniform(3.5, 5.5)
        y[idx] += magnitude * base_std
        pct = percent_change(y[idx - 1], y[idx])
        z = compute_zscore(y, idx)
        anomaly_records.append({
            "index": int(idx),
            "percent_change": round(float(pct), 2),
            "z_score": round(float(z), 2)
        })

    return {
        "id": f"multi_{case_id}",
        "meta": {
            "type": "multi_anomaly",
            "series_length": series_length,
            "n_anomalies": len(anomaly_records)
        },
        "series": [{"t": int(i), "value": float(y[i])} for i in range(len(y))],
        "question": (
            "How many anomalies are present in this series? "
            "List the index of each anomaly and quantify the percentage change at each one."
        ),
        "ground_truth": {
            "anomaly_count": len(anomaly_records),
            "anomalies": anomaly_records
        }
    }

def main():
    pathlib.Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    open(OUTPUT_FILE, "w").close()

    case_id = 0
    for series_length in SERIES_LENGTHS:
        print(f"\n--- Generating {N_PER_LENGTH} samples with series_length={series_length} ---")
        for _ in range(N_PER_LENGTH):
            case = generate_multi_anomaly(case_id, series_length)
            save_case(case)
            print(f"  multi_{case_id} | anomalies: {case['ground_truth']['anomaly_count']} "
                  f"| indices: {[a['index'] for a in case['ground_truth']['anomalies']]}")
            case_id += 1

    print(f"\nDone. {case_id} total samples written to {OUTPUT_FILE}")

    # Summary by series length and anomaly count distribution
    with open(OUTPUT_FILE) as f:
        cases = [json.loads(l) for l in f]

    print("\nAnomaly count distribution per series length:")
    for sl in SERIES_LENGTHS:
        subset = [c for c in cases if c["meta"]["series_length"] == sl]
        counts = [c["ground_truth"]["anomaly_count"] for c in subset]
        dist = {n: counts.count(n) for n in range(1, 11) if counts.count(n) > 0}
        print(f"  length={sl}: {dist}")

if __name__ == "__main__":
    main()