import json
import re
import asyncio
import aiohttp
import pathlib
from collections import defaultdict

API_KEY         = ""
DATA_FILE       = "Data/synthetic_multianomaly.jsonl"
INDEX_TOLERANCE = 2   # ±2 index positions counts as a match

MODELS = [
    #{"model":       "qwen/qwen3-8b", "output_file": "Results/Qwen/Experiment_4.jsonl",},
    #{ "model":       "meta-llama/llama-3.1-8b-instruct", "output_file": "Results/Llama/Experiment_4.jsonl",},
    #{ "model":       "google/gemma-4-31b-it", "output_file": "Results/Gemma/Experiment_4.jsonl",},
    { "model":       "google/gemma-3-12b-it", "output_file": "Results/Gemma3/Experiment_4.jsonl",},
    #{"model": "anthropic/claude-3-haiku",  "output_file": "Results/Haiku/Experiment_4.jsonl"},
    #{"model": "deepseek/deepseek-v3.2",    "output_file": "Results/Deepseek/Experiment_4.jsonl"},
]

# Total concurrent requests shared across all models
CONCURRENCY = 15

PROMPT_TEMPLATE = """You are an expert analyst of time-series data.
You will receive a time series and a question about anomalies in it.

PHASE 1 — SCAN THE SERIES
Read through the entire series carefully. Look for points where the value
deviates sharply from the surrounding trend and then returns to normal.
Each such point is a separate anomaly. Do not assume a fixed number.

PHASE 2 — IDENTIFY EACH ANOMALY
For each anomaly you find:
- Record the exact time index t
- Compute the percentage change from the previous point: ((value[t] - value[t-1]) / |value[t-1]|) × 100
Do NOT skip anomalies. Do NOT merge nearby points into one.

PHASE 3 — COUNT AND REPORT
Count all anomalies you found. Report each one individually.

Return ONLY valid JSON. No explanation. No extra text. No markdown fences.
{
  "anomaly_count": int,
  "anomalies": [
    {"index": int, "percent_change": float},
    ...
  ]
}
"""

def format_series(series, precision=2):
    return ", ".join(f"{p['t']}:{round(p['value'], precision)}" for p in series)

def build_prompt(entry):
    return (
        PROMPT_TEMPLATE
        + "\n\n"
        + f"Time Series:\n{format_series(entry['series'])}\n\n"
        + f"Question:\n{entry['question']}\n"
    )

async def call_llm(session, model_name, prompt):
    async with session.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"}
        },
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
        if "error" in data:
            raise ValueError(f"API error: {data['error']}")
        content = data["choices"][0]["message"]["content"]
        if content is None:
            raise ValueError("API returned None content")
        return content

def fix_json_text(text: str) -> str:
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r'(?<=\d),(?=\d{3}\b)', '', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = re.sub(r'[\x00-\x1f\x7f]', ' ', text)

    return text.strip()

def extract_json(text):
    if text is None:
        raise ValueError("Input text is None")

    cleaned = fix_json_text(text)

    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())

    raise ValueError("Could not parse JSON")

def load_done_ids(filepath):
    done = set()
    if pathlib.Path(filepath).exists():
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["id"])
    return done

def compute_f1(predicted_indices, gt_indices, tolerance=INDEX_TOLERANCE):
    matched_gt = set()
    true_positives = 0
    for pred in predicted_indices:
        for j, gt in enumerate(gt_indices):
            if j not in matched_gt and abs(pred - gt) <= tolerance:
                true_positives += 1
                matched_gt.add(j)
                break
    precision = true_positives / len(predicted_indices) if predicted_indices else 0.0
    recall    = true_positives / len(gt_indices) if gt_indices else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return round(precision, 3), round(recall, 3), round(f1, 3)

_file_locks: dict[str, asyncio.Lock] = {}

def get_file_lock(filepath: str) -> asyncio.Lock:
    if filepath not in _file_locks:
        _file_locks[filepath] = asyncio.Lock()
    return _file_locks[filepath]

async def process_entry(session, entry, model_cfg, semaphore, counters):
    model_name  = model_cfg["model"]
    output_file = model_cfg["output_file"]
    entry_id    = entry["id"]

    async with semaphore:
        try:
            raw = await call_llm(session, model_name, build_prompt(entry))
        except Exception as e:
            print(f"[{model_name}] HTTP error on {entry_id}: {e}")
            return

        try:
            result = extract_json(raw)
        except Exception:
            print(f"[{model_name}] Failed to parse JSON for {entry_id}. Raw: {(raw or '')}")
            return

        pred_count     = result.get("anomaly_count", 0)
        pred_anomalies = result.get("anomalies", [])
        pred_indices   = [a["index"] for a in pred_anomalies if "index" in a]

        gt            = entry["ground_truth"]
        gt_count      = gt["anomaly_count"]
        gt_indices    = [a["index"] for a in gt["anomalies"]]
        series_length = entry["meta"]["series_length"]

        count_match = pred_count == gt_count
        precision, recall, f1 = compute_f1(pred_indices, gt_indices)

        record = json.dumps({
            "id":            entry_id,
            "model":         model_name,
            "series_length": series_length,
            "gt_count":      gt_count,
            "pred_count":    pred_count,
            "count_correct": count_match,
            "gt_indices":    gt_indices,
            "pred_indices":  pred_indices,
            "precision":     precision,
            "recall":        recall,
            "f1":            f1,
        }) + "\n"

        async with get_file_lock(output_file):
            with open(output_file, "a") as out:
                out.write(record)

        if count_match:
            counters[model_name]["count_correct"] += 1
        counters[model_name]["f1_sum"] += f1
        counters[model_name]["total"]  += 1

        print(
            f"[{model_name}] [{counters[model_name]['total']}] {entry_id} | "
            f"len={series_length} | GT={gt_count} Pred={pred_count} "
            f"count={'✓' if count_match else '✗'} | F1={f1:.3f}"
        )

async def run_model(session, model_cfg, entries, semaphore, counters):
    model_name  = model_cfg["model"]
    output_file = model_cfg["output_file"]

    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(output_file)

    # Seed running totals from already-completed results
    count_correct_so_far = 0
    f1_sum_so_far        = 0.0
    if pathlib.Path(output_file).exists():
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("count_correct"):
                    count_correct_so_far += 1
                f1_sum_so_far += r.get("f1", 0.0)

    counters[model_name] = {
        "total":         len(done_ids),
        "count_correct": count_correct_so_far,
        "f1_sum":        f1_sum_so_far,
    }
    print(f"[{model_name}] Resuming — {len(done_ids)} entries already done.")

    pending = [e for e in entries if e["id"] not in done_ids]
    print(f"[{model_name}] {len(pending)} entries to process.")

    tasks = [
        process_entry(session, entry, model_cfg, semaphore, counters)
        for entry in pending
    ]
    await asyncio.gather(*tasks)

def print_summary(model_cfg, counters):
    model_name  = model_cfg["model"]
    output_file = model_cfg["output_file"]
    c           = counters.get(model_name, {})
    total         = c.get("total", 0)
    count_correct = c.get("count_correct", 0)
    f1_sum        = c.get("f1_sum", 0.0)

    if total == 0:
        print(f"[{model_name}] No samples processed.")
        return

    print(f"\n{'='*50}")
    print(f"[{model_name}]")
    print(f"  Total samples:     {total}")
    print(f"  Count accuracy:    {count_correct}/{total} = {count_correct/total:.2f}")
    print(f"  Mean F1 (indices): {f1_sum/total:.3f}")

    # Breakdown by series length
    if pathlib.Path(output_file).exists():
        with open(output_file) as f:
            all_results = [json.loads(l) for l in f if l.strip()]

        by_length = defaultdict(list)
        for r in all_results:
            by_length[r["series_length"]].append(r)

        print(f"  Breakdown by series length:")
        for sl in sorted(by_length.keys()):
            group = by_length[sl]
            cc  = sum(1 for r in group if r["count_correct"])
            mf1 = sum(r["f1"] for r in group) / len(group)
            print(f"    length={sl:4d}: {len(group):2d} samples | "
                  f"count_acc={cc/len(group):.2f} | mean_F1={mf1:.3f}")

async def main():
    with open(DATA_FILE) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    print(f"Loaded {len(entries)} entries from {DATA_FILE}.\n")

    counters  = {}
    semaphore = asyncio.Semaphore(CONCURRENCY)

    session = aiohttp.ClientSession()
    try:
        await asyncio.gather(*[
            run_model(session, model_cfg, entries, semaphore, counters)
            for model_cfg in MODELS
        ])
    finally:
        await session.close()

    for model_cfg in MODELS:
        print_summary(model_cfg, counters)

if __name__ == "__main__":
    asyncio.run(main())