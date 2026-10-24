import json
import random
import os
import sys, pathlib

PROJECT_ROOT    = pathlib.Path(__file__).resolve().parent.parent
INPUT_FILES     = [str(PROJECT_ROOT / "Data" / "synthetic.jsonl")]
RANDOMIZED_OUTPUT = str(PROJECT_ROOT / "Data" / "randomized_dataset.jsonl")
ANSWER_KEY_OUTPUT = str(PROJECT_ROOT / "Data" / "answer_key.jsonl")

random.seed(42)

# ── Candidate generators ──────────────────────────────────────────────────────
def perturb(value, scale=0.15):
    return round(value * (1 + random.uniform(-scale, scale)), 2)

def linear_spike(gt):
    faithful = (
        f"There is a clear anomaly at t={gt['anomaly_index']}. "
        f"The value increases by approximately {gt['percent_change']}% compared to the previous point. "
        f"This corresponds to a z-score of about {gt['z_score']}, indicating statistical significance."
    )
    numeric_error = (
        f"There is an anomaly at t={gt['anomaly_index']}. "
        f"The value increases by approximately {perturb(gt['percent_change'])}% compared to the previous point. "
        f"This corresponds to a z-score of about {perturb(gt['z_score'])}."
    )
    verbose_incorrect = (
        f"A comprehensive review of the series suggests no statistically significant anomaly. "
        f"Although fluctuations are observed, they remain within expected stochastic variation. "
        f"The most notable deviation appears around t={gt['anomaly_index'] + 5}, "
        f"with an estimated increase of roughly {perturb(gt['percent_change'], 0.5)}%. "
        f"This shift does not exceed volatility thresholds and does not indicate structural instability. "
        f"Overall, the time series demonstrates stable and consistent dynamics."
    )
    return faithful, numeric_error, verbose_incorrect

def seasonal_drop(gt):
    faithful = (
        f"There is an anomaly at t={gt['anomaly_index']}. "
        f"The value drops by approximately {abs(gt['percent_change'])}% compared to the previous point. "
        f"This corresponds to a z-score of about {gt['z_score']}, indicating a significant downward deviation."
    )
    numeric_error = (
        f"There is an anomaly at t={gt['anomaly_index']}. "
        f"The value drops by approximately {perturb(abs(gt['percent_change']))}% compared to the previous point. "
        f"This corresponds to a z-score of about {perturb(gt['z_score'])}."
    )
    verbose_incorrect = (
        f"The seasonal pattern of the series remains intact throughout the observed window. "
        f"Minor troughs are visible near t={gt['anomaly_index'] + 4}, "
        f"consistent with expected cyclical behavior. "
        f"No point falls outside the anticipated seasonal range, "
        f"and the oscillation amplitude remains stable across the full series. "
        f"Overall, no anomalous drop is detected."
    )
    return faithful, numeric_error, verbose_incorrect

def structural_break(gt):
    faithful = (
        f"There is a structural break at t={gt['break_index']}. "
        f"The mean shifts from approximately {gt['mean_before']} to {gt['mean_after']}, "
        f"representing a {gt['percent_shift']}% increase."
    )
    numeric_error = (
        f"There is a structural change at t={gt['break_index']}. "
        f"The mean moves from {perturb(gt['mean_before'])} to {perturb(gt['mean_after'])}, "
        f"which corresponds to roughly {perturb(gt['percent_shift'])}%."
    )
    verbose_incorrect = (
        f"The series remains broadly consistent across the entire period. "
        f"Minor oscillations are present, but no meaningful structural shift occurs. "
        f"A small variation near t={gt['break_index'] + 6} is visible, "
        f"though the average level remains approximately {perturb(gt['mean_before'], 0.5)} throughout. "
        f"Statistical assessment indicates regime stability rather than structural change."
    )
    return faithful, numeric_error, verbose_incorrect

def multi_metric(gt):
    faithful = (
        f"There is an anomaly at t={gt['anomaly_index']}. "
        f"The spike represents a {gt['percent_change']}% increase and has a z-score of {gt['z_score']}. "
        f"The variance increases from {gt['variance_before']} before the spike "
        f"to {gt['variance_after']} afterward."
    )
    numeric_error = (
        f"There is an anomaly at t={gt['anomaly_index']}. "
        f"The spike reflects approximately {perturb(gt['percent_change'])}% growth "
        f"with a z-score near {perturb(gt['z_score'])}. "
        f"The variance shifts from {perturb(gt['variance_before'])} "
        f"to {perturb(gt['variance_after'])}."
    )
    verbose_incorrect = (
        f"The series exhibits relatively smooth behavior without a pronounced anomaly. "
        f"A modest deviation appears around t={gt['anomaly_index'] + 4}, "
        f"amounting to roughly 5% change. "
        f"The variance appears to decrease over time, stabilizing near "
        f"{perturb(gt['variance_before'], 0.4)}. "
        f"Additional smoothing analysis confirms the absence of volatility clustering."
    )
    return faithful, numeric_error, verbose_incorrect

def relative_extremum(gt):
    faithful = (
        f"The largest spike occurs at t={gt['largest_spike_index']}. "
        f"This spike has a magnitude of approximately {gt['largest_spike_magnitude']}, "
        f"which exceeds the other identified spikes."
    )
    numeric_error = (
        f"The largest spike appears at t={gt['largest_spike_index']}. "
        f"It has an estimated magnitude of about {perturb(gt['largest_spike_magnitude'])}, "
        f"making it the dominant extremum."
    )
    verbose_incorrect = (
        f"The most substantial spike occurs at t={gt['spike_indices'][0]}. "
        f"This deviation clearly surpasses all others and represents the primary outlier event. "
        f"The magnitude of this peak is considerably higher than subsequent spikes, "
        f"confirming it as the largest extremum within the dataset. "
        f"Comparative inspection shows no later spike exceeds this level."
    )
    return faithful, numeric_error, verbose_incorrect

def mean_shift_query(gt, series_length=100):
    mid = series_length // 2
    faithful = (
        f"The mean of the first half (t=0 to t={mid - 1}) is approximately {gt['mean_first_half']}, "
        f"while the mean of the second half (t={mid} to t={series_length - 1}) is approximately {gt['mean_second_half']}. "
        f"The second half has a higher mean, representing a {gt['percent_difference']}% increase."
    )
    numeric_error = (
        f"The mean of the first half (t=0 to t={mid - 1}) is approximately {perturb(gt['mean_first_half'])}, "
        f"and the mean of the second half (t={mid} to t={series_length - 1}) is approximately {perturb(gt['mean_second_half'])}. "
        f"The second half has a higher mean, with a difference of roughly {perturb(gt['percent_difference'])}%."
    )
    verbose_incorrect = (
        f"The series maintains a broadly consistent level throughout both halves. "
        f"The first half has an estimated mean of approximately {perturb(gt['mean_first_half'], 0.5)}, "
        f"and the second half shows a similar average near {perturb(gt['mean_first_half'], 0.4)}. "
        f"No meaningful difference in central tendency is observed between the two segments. "
        f"The overall trend suggests stationarity rather than a level shift."
    )
    return faithful, numeric_error, verbose_incorrect

def volatility_shift(gt):
    lower_seg = "first" if gt['higher_volatility_segment'] == "second" else "second"
    faithful = (
        f"The volatility of the series changes around t={gt['break_index']}. "
        f"The standard deviation before this point is approximately {gt['std_before']}, "
        f"compared to {gt['std_after']} afterward. "
        f"The {gt['higher_volatility_segment']} segment exhibits higher variability."
    )
    numeric_error = (
        f"The volatility shifts around t={gt['break_index']}. "
        f"The standard deviation before is approximately {perturb(gt['std_before'])}, "
        f"and after is approximately {perturb(gt['std_after'])}. "
        f"The {gt['higher_volatility_segment']} half shows greater variability."
    )
    verbose_incorrect = (
        f"The series displays uniform variability across its entire length. "
        f"Both the early and later segments show similar spread, "
        f"with standard deviation hovering near {perturb(gt['std_before'], 0.4)} throughout. "
        f"No structural change in volatility is evident, "
        f"and the {lower_seg} segment does not differ meaningfully from the {gt['higher_volatility_segment']} segment. "
        f"The data appears homoscedastic overall."
    )
    return faithful, numeric_error, verbose_incorrect

def trend_comparison(gt, series_length=100):
    mid = series_length // 2
    steeper  = gt["steeper_half"]
    shallower = "second" if steeper == "first" else "first"
    faithful = (
        f"The trend in the first half (t=0 to t={mid-1}) has a slope of approximately "
        f"{gt['slope_first_half']} units per step, while the second half (t={mid} to t={series_length-1}) "
        f"has a slope of approximately {gt['slope_second_half']} units per step. "
        f"The {steeper} half has a steeper trend in absolute terms."
    )
    numeric_error = (
        f"The first half (t=0 to t={mid-1}) has an estimated slope of about "
        f"{perturb(gt['slope_first_half'])} units per step, and the second half "
        f"(t={mid} to t={series_length-1}) has a slope of about {perturb(gt['slope_second_half'])} units per step. "
        f"The {steeper} half exhibits a steeper trend."
    )
    verbose_incorrect = (
        f"Both halves of the series display broadly similar trends with no meaningful difference in slope. "
        f"The first half shows a gradual rise of approximately {perturb(gt['slope_first_half'], 0.6)} units per step, "
        f"and the second half shows a comparable rate near {perturb(gt['slope_first_half'], 0.5)} units per step. "
        f"Overall the series appears to follow a consistent linear trajectory throughout, "
        f"with the {shallower} half being marginally steeper."
    )
    return faithful, numeric_error, verbose_incorrect


def temporal_ordering(gt):
    correct = gt["correct_order"]         # e.g. ["B", "C", "A"]
    means   = gt["segment_means"]         # e.g. {"A": 6.6, "B": 19.06, "C": 23.83}

    # A wrong order: rotate correct by 1 so it's plausibly close but wrong
    wrong_order = [correct[1], correct[2], correct[0]]
    # A completely reversed order
    reversed_order = list(reversed(correct))

    faithful = (
        f"The correct temporal order is {correct[0]}, {correct[1]}, {correct[2]}. "
        f"Segment {correct[0]} has the lowest mean level (~{means[correct[0]]}), "
        f"followed by {correct[1]} (~{means[correct[1]]}), "
        f"and {correct[2]} with the highest mean (~{means[correct[2]]}). "
        f"The increasing mean levels confirm this ordering."
    )
    numeric_error = (
        f"The correct temporal order is {correct[0]}, {correct[1]}, {correct[2]}. "
        f"Segment {correct[0]} has an estimated mean of ~{perturb(means[correct[0]])}, "
        f"segment {correct[1]} has a mean near ~{perturb(means[correct[1]])}, "
        f"and segment {correct[2]} has the highest mean at ~{perturb(means[correct[2]])}."
    )
    verbose_incorrect = (
        f"Based on the value ranges and trend direction within each segment, "
        f"the most plausible temporal ordering is {wrong_order[0]}, {wrong_order[1]}, {wrong_order[2]}. "
        f"Segment {wrong_order[0]} appears to represent the earliest period due to its trajectory, "
        f"followed by {wrong_order[1]} which shows transitional behavior, "
        f"and {wrong_order[2]} which reflects the later dynamics of the series."
    )
    return faithful, numeric_error, verbose_incorrect


def shape_classification(gt):
    shape    = gt["shape"]
    tp_idx   = gt["turning_point_index"]
    tp_val   = gt["value_at_turning_point"]
    v_start  = gt["value_at_start"]
    v_end    = gt["value_at_end"]

    # Map each shape to a plausible wrong label
    wrong_shape_map = {
        "monotone_increase": "concave",
        "monotone_decrease": "convex",
        "concave":           "monotone_increase",
        "convex":            "monotone_decrease",
    }
    wrong_shape = wrong_shape_map[shape]

    if tp_idx is not None:
        faithful = (
            f"The series is best described as '{shape}'. "
            f"It reaches a turning point at approximately t={tp_idx} (value ~{tp_val}), "
            f"starting at ~{v_start} and ending at ~{v_end}."
        )
        numeric_error = (
            f"The series is '{shape}'. "
            f"The turning point occurs near t={tp_idx} with a value of approximately {perturb(tp_val)}, "
            f"with the series starting at ~{perturb(v_start)} and ending at ~{perturb(v_end)}."
        )
    else:
        faithful = (
            f"The series is best described as '{shape}'. "
            f"It moves consistently from ~{v_start} at the start to ~{v_end} at the end "
            f"with no turning point."
        )
        numeric_error = (
            f"The series is '{shape}'. "
            f"It moves from approximately {perturb(v_start)} to {perturb(v_end)} "
            f"without a turning point."
        )

    verbose_incorrect = (
        f"The overall shape of the series is best classified as '{wrong_shape}'. "
        f"The series begins at approximately {perturb(v_start, 0.3)} and "
        f"{'reaches a peak before declining' if wrong_shape == 'concave' else 'reaches a trough before recovering' if wrong_shape == 'convex' else 'rises steadily' if wrong_shape == 'monotone_increase' else 'declines steadily'}, "
        f"ending near {perturb(v_end, 0.3)}. "
        f"The dominant pattern is consistent with a '{wrong_shape}' trajectory."
    )
    return faithful, numeric_error, verbose_incorrect


# ── Dispatch ──────────────────────────────────────────────────────────────────
def generate_candidates(case):
    meta = case["meta"]
    gt   = case["ground_truth"]
    t    = meta.get("type")
    sl   = meta.get("series_length", len(case["series"]))

    if t == "linear_spike":
        f, n, i = linear_spike(gt)
    elif t == "seasonal_drop":
        f, n, i = seasonal_drop(gt)
    elif t == "structural_break":
        f, n, i = structural_break(gt)
    elif t == "multi_metric_consistency":
        f, n, i = multi_metric(gt)
    elif t == "relative_extremum":
        f, n, i = relative_extremum(gt)
    elif t == "mean_shift_query":
        f, n, i = mean_shift_query(gt, series_length=sl)
    elif t == "volatility_shift":
        f, n, i = volatility_shift(gt)
    elif t == "trend_comparison":
        f, n, i = trend_comparison(gt, series_length=sl)
    elif t == "temporal_ordering":
        f, n, i = temporal_ordering(gt)
    elif t == "shape_classification":
        f, n, i = shape_classification(gt)
    else:
        raise ValueError(f"Unknown type: {t}")

    labeled = [(0, f), (1, n), (2, i)]
    random.shuffle(labeled)
    return [x[1] for x in labeled], [x[0] for x in labeled]

# ── File helpers ──────────────────────────────────────────────────────────────
def load_existing_ids(filepath):
    ids = set()
    if pathlib.Path(filepath).exists():
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(json.loads(line)["id"])
    return ids


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(str(PROJECT_ROOT / "Data"), exist_ok=True)

    # Step 1: load existing ids and append missing ones
    print("\n=== Step 1: Generating candidates for new/missing entries ===")
    existing_ids = load_existing_ids(RANDOMIZED_OUTPUT)

    added = 0
    skipped = 0

    with open(RANDOMIZED_OUTPUT, "a") as out, \
         open(ANSWER_KEY_OUTPUT, "a") as key_out:

        for file in INPUT_FILES:
            if not pathlib.Path(file).exists():
                print(f"Skipping missing file: {file}")
                continue
            with open(file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    case     = json.loads(line)
                    entry_id = case["id"]

                    if entry_id in existing_ids:
                        skipped += 1
                        continue

                    candidates, order = generate_candidates(case)
                    out.write(json.dumps({
                        "id":         entry_id,
                        "series":     case["series"],
                        "question":   case["question"],
                        "candidates": candidates,
                    }) + "\n")
                    key_out.write(json.dumps({
                        "id":    entry_id,
                        "order": order,
                    }) + "\n")
                    added += 1

    print(f"  Skipped (already existed): {skipped}")
    print(f"  Newly added: {added}")
    total = skipped + added
    print(f"  Total entries now in file: {total}")


if __name__ == "__main__":
    main()