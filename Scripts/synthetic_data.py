import numpy as np
import json
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from Utils.stats import percent_change, compute_zscore, slope

np.random.seed(67)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_FILE  = str(PROJECT_ROOT / "Data" / "synthetic.jsonl")
LENGTHS_SCHEDULE = [100] * 20 + [200] * 10 + [300] * 10 + [500] * 10  # 50 per type

NEW_TYPES = ["trend_comparison", "temporal_ordering", "shape_classification"]

TYPE_TO_PREFIX = {
    "linear_spike":             "ls",
    "seasonal_drop":            "sd",
    "structural_break":         "sb",
    "multi_metric_consistency": "mm",
    "relative_extremum":        "re",
    "mean_shift_query":         "ms",
    "volatility_shift":         "vs",
    # ── NEW ──────────────────────────────
    "trend_comparison":         "tc",
    "temporal_ordering":        "to",
    "shape_classification":     "sc",
}

# ── Original 7 generators ─────────────────────────────────────────────────────

def generate_linear_spike(case_id, series_length=100):
    t = np.arange(series_length)
    y = 0.5 * t + np.random.normal(0, 1, series_length)
    spike_idx = np.random.randint(int(series_length * 0.2), int(series_length * 0.8))
    y[spike_idx] += 5 * np.std(y)
    pct = percent_change(y[spike_idx - 1], y[spike_idx])
    z   = compute_zscore(y, spike_idx)
    return {
        "id":   f"ls_{case_id}",
        "meta": {"type": "linear_spike", "series_length": series_length},
        "series": [{"t": int(i), "value": float(y[i])} for i in range(series_length)],
        "question": "Is there an anomaly? If yes, identify when and quantify the change.",
        "ground_truth": {
            "anomaly_index":  int(spike_idx),
            "percent_change": round(float(pct), 2),
            "z_score":        round(float(z), 2),
        },
    }

def generate_seasonal_drop(case_id, series_length=100):
    t = np.arange(series_length)
    y = np.sin(2 * np.pi * t / 20) + np.random.normal(0, 0.2, series_length)
    drop_idx = np.random.randint(int(series_length * 0.2), int(series_length * 0.8))
    y[drop_idx] -= 3
    pct = percent_change(y[drop_idx - 1], y[drop_idx])
    z   = compute_zscore(y, drop_idx)
    return {
        "id":   f"sd_{case_id}",
        "meta": {"type": "seasonal_drop", "series_length": series_length},
        "series": [{"t": int(i), "value": float(y[i])} for i in range(series_length)],
        "question": "Is there an anomaly? If yes, identify when and quantify the change.",
        "ground_truth": {
            "anomaly_index":  int(drop_idx),
            "percent_change": round(float(pct), 2),
            "z_score":        round(float(z), 2),
        },
    }

def generate_structural_break(case_id, series_length=100):
    break_idx = series_length // 2
    y = np.ones(series_length) * 10
    y[break_idx:] += 10
    y += np.random.normal(0, 0.5, series_length)
    mean_before = float(np.mean(y[:break_idx]))
    mean_after  = float(np.mean(y[break_idx:]))
    pct_shift   = percent_change(mean_before, mean_after)
    return {
        "id":   f"sb_{case_id}",
        "meta": {"type": "structural_break", "series_length": series_length},
        "series": [{"t": int(i), "value": float(y[i])} for i in range(series_length)],
        "question": "Is there a structural change? If yes, quantify the shift.",
        "ground_truth": {
            "break_index":   break_idx,
            "mean_before":   round(mean_before, 2),
            "mean_after":    round(mean_after, 2),
            "percent_shift": round(pct_shift, 2),
        },
    }

def generate_relative_extremum(case_id, series_length=100):
    t = np.arange(series_length)
    y = 0.2 * t + np.random.normal(0, 1, series_length)
    lo = int(series_length * 0.2)
    hi = int(series_length * 0.8)
    spike_indices  = sorted(np.random.choice(range(lo, hi), 3, replace=False))
    spike_magnitudes = []
    for idx in spike_indices:
        mag = np.random.uniform(3, 6)
        y[idx] += mag
        spike_magnitudes.append(mag)
    best_pos   = int(np.argmax(spike_magnitudes))
    best_index = spike_indices[best_pos]
    best_mag   = spike_magnitudes[best_pos]
    return {
        "id":   f"re_{case_id}",
        "meta": {"type": "relative_extremum", "series_length": series_length},
        "series": [{"t": int(i), "value": float(y[i])} for i in range(series_length)],
        "question": "Identify the largest spike in the series and justify your answer with numeric evidence.",
        "ground_truth": {
            "spike_indices":            [int(i) for i in spike_indices],
            "largest_spike_index":      int(best_index),
            "largest_spike_magnitude":  round(float(best_mag), 2),
        },
    }

def generate_multi_metric(case_id, series_length=100):
    t = np.arange(series_length)
    y = 0.3 * t + np.random.normal(0, 1, series_length)
    spike_idx = np.random.randint(int(series_length * 0.3), int(series_length * 0.7))
    y[spike_idx] += 5 * np.std(y)
    y[spike_idx+1:] += np.random.normal(0, 2, series_length - spike_idx - 1)
    pct        = percent_change(y[spike_idx - 1], y[spike_idx])
    z          = compute_zscore(y, spike_idx)
    var_before = np.var(y[:spike_idx])
    var_after  = np.var(y[spike_idx+1:])
    return {
        "id":   f"mm_{case_id}",
        "meta": {"type": "multi_metric_consistency", "series_length": series_length},
        "series": [{"t": int(i), "value": float(y[i])} for i in range(series_length)],
        "question": (
            "Analyze the series. Is there an anomaly? "
            "If yes, quantify the spike and describe any change in variance before and after it."
        ),
        "ground_truth": {
            "anomaly_index":   int(spike_idx),
            "percent_change":  round(float(pct), 2),
            "z_score":         round(float(z), 2),
            "variance_before": round(float(var_before), 2),
            "variance_after":  round(float(var_after), 2),
        },
    }

def generate_mean_shift_query(case_id, series_length=100):
    t    = np.arange(series_length)
    base = 0.1 * t + np.random.normal(0, 0.5, series_length)
    shift = np.random.uniform(10, 20)
    mid   = series_length // 2
    base[mid:] += shift
    mean_first  = float(np.mean(base[:mid]))
    mean_second = float(np.mean(base[mid:]))
    med_first   = float(np.median(base[:mid]))
    med_second  = float(np.median(base[mid:]))
    pct_diff    = percent_change(mean_first, mean_second)
    return {
        "id":   f"ms_{case_id}",
        "meta": {"type": "mean_shift_query", "series_length": series_length},
        "series": [{"t": int(i), "value": float(base[i])} for i in range(series_length)],
        "question": (
            f"Compare the mean of the first half of the series (t=0 to t={mid - 1}) "
            f"with the mean of the second half (t={mid} to t={series_length - 1}). "
            f"Quantify the difference and state which half has a higher mean."
        ),
        "ground_truth": {
            "mean_first_half":    round(mean_first, 2),
            "mean_second_half":   round(mean_second, 2),
            "median_first_half":  round(med_first, 2),
            "median_second_half": round(med_second, 2),
            "percent_difference": round(float(pct_diff), 2),
            "higher_half":        "second",
        },
    }

def generate_volatility_shift(case_id, series_length=100):
    break_idx  = np.random.randint(int(series_length * 0.3), int(series_length * 0.7))
    low_std    = np.random.uniform(0.5, 1.5)
    high_std   = np.random.uniform(4.0, 7.0)
    if np.random.rand() > 0.5:
        std_before, std_after = low_std, high_std
        higher_segment = "second"
    else:
        std_before, std_after = high_std, low_std
        higher_segment = "first"
    t     = np.arange(series_length)
    base  = 0.2 * t
    noise = np.concatenate([
        np.random.normal(0, std_before, break_idx),
        np.random.normal(0, std_after,  series_length - break_idx),
    ])
    y = base + noise
    return {
        "id":   f"vs_{case_id}",
        "meta": {"type": "volatility_shift", "series_length": series_length},
        "series": [{"t": int(i), "value": float(y[i])} for i in range(series_length)],
        "question": (
            "Does the volatility of the series change at any point? "
            "If yes, identify approximately when and compare the variability before and after."
        ),
        "ground_truth": {
            "break_index":               int(break_idx),
            "std_before":                round(float(np.std(y[:break_idx])), 2),
            "std_after":                 round(float(np.std(y[break_idx:])), 2),
            "higher_volatility_segment": higher_segment,
        },
    }


# ── NEW TYPE 8: trend_comparison ──────────────────────────────────────────────

def generate_trend_comparison(case_id, series_length=100):
    """
    Two-phase linear trend with meaningfully different slopes in each half.
    Model must estimate per-step slope in each half and name the steeper one.
    GT: slope_first_half, slope_second_half, steeper_half.
    """
    mid = series_length // 2
    slope_first  = np.random.uniform(-0.5, 2.0)
    slope_second = np.random.uniform(-0.5, 2.0)
    # Guarantee a detectable difference between the two halves
    while abs(slope_first - slope_second) < 0.4:
        slope_second = np.random.uniform(-0.5, 2.0)

    y = np.zeros(series_length)
    y[:mid]  = slope_first  * np.arange(mid) + np.random.normal(0, 0.3, mid)
    y[mid:]  = (y[mid - 1]
                + slope_second * np.arange(1, series_length - mid + 1)
                + np.random.normal(0, 0.3, series_length - mid))

    fit_first  = float(np.polyfit(np.arange(mid), y[:mid], 1)[0])
    fit_second = float(np.polyfit(np.arange(series_length - mid), y[mid:], 1)[0])
    steeper    = "first" if abs(fit_first) > abs(fit_second) else "second"

    return {
        "id":   f"tc_{case_id}",
        "meta": {"type": "trend_comparison", "series_length": series_length},
        "series": [{"t": int(i), "value": float(y[i])} for i in range(series_length)],
        "question": (
            f"Compare the trend in the first half of the series (t=0 to t={mid-1}) "
            f"with the second half (t={mid} to t={series_length-1}). "
            f"Estimate the slope (units per step) in each half and state which half "
            f"has a steeper trend in absolute terms."
        ),
        "ground_truth": {
            "slope_first_half":  round(fit_first, 3),
            "slope_second_half": round(fit_second, 3),
            "steeper_half":      steeper,
        },
    }





# ── NEW TYPE 9: temporal_ordering ────────────────────────────────────────────

def generate_temporal_ordering(case_id, series_length=100):
    """
    TSAQA Temporal Relationship category.

    The full series is split into 3 equal non-overlapping segments and
    shuffled into a random permutation. The model receives the segments
    in shuffled order (each labelled Segment A/B/C) and must reconstruct
    the original left-to-right sequence, justifying with numeric evidence
    (e.g. mean level, trend direction, value range).

    The series is designed so segments are clearly distinguishable:
    each third has a distinct mean level (low → mid → high or permuted),
    so the correct ordering is deterministic from the data.

    GT: correct_order (list of segment labels in temporal order, e.g. ["B","A","C"]),
        segment_means (dict of label → mean value for evaluation).
    """
    seg_len = series_length // 3
    # Three segments with clearly separated mean levels
    levels = sorted(np.random.uniform(0, 30, 3))   # always low < mid < high
    noise_std = np.random.uniform(0.3, 1.0)

    segments_in_order = []
    for level in levels:
        seg = level + np.random.normal(0, noise_std, seg_len)
        # Optional: add a small local trend within each segment for richer signal
        seg += np.random.uniform(-0.05, 0.05) * np.arange(seg_len)
        segments_in_order.append(seg)

    # Shuffle into a random permutation
    perm = list(np.random.permutation(3))           # e.g. [2, 0, 1]
    labels = ["A", "B", "C"]
    shuffled = [segments_in_order[i] for i in perm]

    # correct_order: which label comes 1st, 2nd, 3rd in real time
    # perm[i] = original position of shuffled[i], so inverse_perm[orig] = shuffled_label
    inverse_perm = [0] * 3
    for shuffled_pos, orig_pos in enumerate(perm):
        inverse_perm[orig_pos] = shuffled_pos
    correct_order = [labels[inverse_perm[orig]] for orig in range(3)]  # e.g. ["B","C","A"]

    segment_means = {
        labels[i]: round(float(np.mean(shuffled[i])), 2)
        for i in range(3)
    }

    # Build the series presented to the model: segments in shuffled order, relabelled t
    presented_series = []
    t_offset = 0
    segment_data = {}
    for label, seg in zip(labels, shuffled):
        pts = [{"t": int(t_offset + j), "value": float(seg[j])} for j in range(len(seg))]
        segment_data[label] = {
            "t_start": t_offset,
            "t_end":   t_offset + len(seg) - 1,
            "points":  pts,
        }
        presented_series.extend(pts)
        t_offset += len(seg)

    return {
        "id":   f"to_{case_id}",
        "meta": {
            "type":         "temporal_ordering",
            "series_length": series_length,
            "seg_len":       seg_len,
            "permutation":   [int(x) for x in perm],
        },
        "segments": segment_data,   # A/B/C with t-ranges, for prompt construction
        "series":   presented_series,
        "question": (
            "Three segments (A, B, C) were extracted from a single time series "
            "but are presented in shuffled order. Each segment covers an equal number "
            f"of time steps (length {seg_len}). "
            "Determine the correct temporal ordering of the segments from earliest to latest. "
            "Justify your answer using the mean level, trend direction, or value range of each segment. "
            "Return the correct order as a sequence of three labels, e.g. 'B, A, C'."
        ),
        "ground_truth": {
            "correct_order":  correct_order,   # e.g. ["B", "C", "A"]
            "segment_means":  segment_means,
        },
    }


# ── NEW TYPE 10: shape_classification ────────────────────────────────────────

def generate_shape_classification(case_id, series_length=100):
    """
    TSAQA Classification category.

    Generates a series with one of four canonical temporal shapes:
      - 'monotone_increase': steady upward trend throughout
      - 'monotone_decrease': steady downward trend throughout
      - 'concave' (hill):    rises then falls, single interior peak
      - 'convex' (valley):   falls then rises, single interior trough

    All shapes have light noise added. The model must identify which
    shape best describes the overall pattern and justify with the
    approximate location of any turning point and the direction of change.

    GT: shape (str), turning_point_index (int or None for monotone),
        value_at_start (float), value_at_end (float), value_at_turning_point (float or None).
    """
    shape = np.random.choice(["monotone_increase", "monotone_decrease", "concave", "convex"])
    t = np.arange(series_length)
    noise_std = np.random.uniform(0.5, 1.5)

    if shape == "monotone_increase":
        slope_val = np.random.uniform(0.3, 1.0)
        y = slope_val * t + np.random.normal(0, noise_std, series_length)
        turning_point_index = None
        value_at_turning_point = None

    elif shape == "monotone_decrease":
        slope_val = np.random.uniform(0.3, 1.0)
        y = -slope_val * t + np.random.normal(0, noise_std, series_length)
        turning_point_index = None
        value_at_turning_point = None

    elif shape == "concave":   # hill: up then down
        peak = np.random.randint(int(series_length * 0.35), int(series_length * 0.65))
        amplitude = np.random.uniform(10, 25)
        # Piecewise linear up then down
        y = np.where(t <= peak,
                     amplitude * t / peak,
                     amplitude * (series_length - 1 - t) / (series_length - 1 - peak))
        y += np.random.normal(0, noise_std, series_length)
        turning_point_index = int(peak)
        value_at_turning_point = round(float(y[peak]), 2)

    else:   # convex: valley, down then up
        trough = np.random.randint(int(series_length * 0.35), int(series_length * 0.65))
        amplitude = np.random.uniform(10, 25)
        y = np.where(t <= trough,
                     -amplitude * t / trough,
                     -amplitude * (series_length - 1 - t) / (series_length - 1 - trough))
        y += np.random.normal(0, noise_std, series_length)
        turning_point_index = int(trough)
        value_at_turning_point = round(float(y[trough]), 2)

    return {
        "id":   f"sc_{case_id}",
        "meta": {"type": "shape_classification", "series_length": series_length},
        "series": [{"t": int(i), "value": float(y[i])} for i in range(series_length)],
        "question": (
            "Classify the overall shape of this time series. "
            "Choose the best description from: "
            "'monotone_increase' (rises throughout), "
            "'monotone_decrease' (falls throughout), "
            "'concave' (rises then falls, single peak), or "
            "'convex' (falls then rises, single trough). "
            "Justify your answer by referencing the approximate index and value of any "
            "turning point, and the values at the start and end of the series."
        ),
        "ground_truth": {
            "shape":                  str(shape),
            "turning_point_index":    turning_point_index,
            "value_at_turning_point": value_at_turning_point,
            "value_at_start":         round(float(y[0]), 2),
            "value_at_end":           round(float(y[-1]), 2),
        },
    }


# ── Registry ──────────────────────────────────────────────────────────────────

GENERATORS = {
    "linear_spike":             generate_linear_spike,
    "seasonal_drop":            generate_seasonal_drop,
    "structural_break":         generate_structural_break,
    "multi_metric_consistency": generate_multi_metric,
    "relative_extremum":        generate_relative_extremum,
    "mean_shift_query":         generate_mean_shift_query,
    "volatility_shift":         generate_volatility_shift,
    # ── NEW ──
    "trend_comparison":         generate_trend_comparison,
    "temporal_ordering":        generate_temporal_ordering,
    "shape_classification":     generate_shape_classification,
}


# ── Helpers (unchanged) ───────────────────────────────────────────────────────

def get_existing_ids():
    existing = set()
    if pathlib.Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    case = json.loads(line)
                    existing.add(case.get("id", ""))
    return existing

def get_next_case_num(type_str, existing_ids):
    prefix = TYPE_TO_PREFIX[type_str]
    num = 0
    while f"{prefix}_{num}" in existing_ids:
        num += 1
    return num


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pathlib.Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    existing_ids = get_existing_ids()
    if existing_ids:
        print(f"Found {len(existing_ids)} existing entries")
    added = 0

    with open(OUTPUT_FILE, "a") as f:
        for type_str in NEW_TYPES:
            gen_fn   = GENERATORS[type_str]
            case_num = get_next_case_num(type_str, existing_ids)
            for length in LENGTHS_SCHEDULE:
                case_id = f"{TYPE_TO_PREFIX[type_str]}_{case_num}"
                if case_id not in existing_ids:
                    case = gen_fn(case_num, series_length=length)
                    f.write(json.dumps(case) + "\n")
                    existing_ids.add(case_id)
                    print(f"  {case_id} | type={type_str} | length={length}")
                    added += 1
                case_num += 1

    print(f"\nAdded {added} new entries.")
    with open(OUTPUT_FILE) as f:
        total = sum(1 for l in f if l.strip())
    print(f"Total entries: {total}")

    # Distribution summary
    with open(OUTPUT_FILE) as f:
        cases = [json.loads(l) for l in f if l.strip()]
    from collections import Counter
    type_counts = Counter(c["meta"]["type"] for c in cases)
    print("\nEntries per query type:")
    for t, n in sorted(type_counts.items()):
        print(f"  {t:<30} {n}")


if __name__ == "__main__":
    main()