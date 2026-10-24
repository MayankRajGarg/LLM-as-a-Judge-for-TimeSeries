import json
import re
import asyncio
import aiohttp
import pathlib

DATA_FILE       = "Data/randomized_real_dataset_paraphrased.jsonl"
ANSWER_KEY_FILE = "Data/answer_real_key_paraphrased.jsonl"
LOCAL_API_BASE_URL = "http://localhost:1235/v1"
LOCAL_MODEL = "qwen3-8b"
LOCAL_CHAT_COMPLETIONS_URL = f"{LOCAL_API_BASE_URL}/chat/completions"
LOCAL_REQUEST_TIMEOUT_SECONDS = 180

MODELS = [
    {
        "model": LOCAL_MODEL,
        "output_file": "Results/New_Qwen/RealData_New_Experiment_3.jsonl",
    },
]

# Total concurrent requests shared across all models
CONCURRENCY = 1

JUDGE_TEMPLATE = """You are an expert evaluator of time-series explanations.
You will receive:
- A time series (index: value pairs)
- A question about the time series
- A single candidate explanation

Your goal: classify the explanation using the criteria below.

PHASE 1 — UNDERSTAND THE QUESTION
Identify what the question is asking for:
- Anomaly / spike detection (sudden temporary deviation)
- Regime shift (sustained step change in level)
- Trend (gradual increase or decrease over many steps)
- Volatility change (increase or decrease in variance)
- Extreme value (maximum or minimum)
- Percentage or magnitude of change between specific points
- General behavior description

PHASE 2 — ANALYZE THE TIME SERIES
Identify the relevant evidence needed to answer the question.
For percentage changes: compute manually as ((new - old) / |old|) × 100.
CRITICAL: Do NOT trust the numbers in the explanation.
Verify all numeric claims yourself against the raw data.

PHASE 3 — EVALUATE THE EXPLANATION
Check the explanation against the data on these dimensions:
[A] Does it correctly identify the right pattern, location, and direction?
[B] Are its numeric claims correct within ±2% tolerance?
[C] Does it directly answer the question?
[D] Is its reasoning internally consistent?
[E] Does it introduce unsupported claims not derivable from the data?

PHASE 4 — ASSIGN LABEL
Based on your evaluation assign exactly one label:
0 = Completely incorrect (wrong pattern, wrong location, or contradicts the data)
1 = Correct reasoning and direction but contains numeric errors
2 = Fully correct (right pattern, right location, accurate numbers, answers the question)

Return ONLY valid JSON. No explanation. No extra text. No markdown fences.
{"label": int}
"""

def format_series(series, precision=2):
    return ", ".join(f"{p['t']}:{round(p['value'], precision)}" for p in series)

def build_prompt(entry, explanation):
    return (
        JUDGE_TEMPLATE
        + "\n\n"
        + f"Time Series:\n{format_series(entry['series'])}\n\n"
        + f"Question:\n{entry['question']}\n\n"
        + f"Explanation:\n{explanation}\n"
    )

async def call_llm(session, model_name, prompt):
    async with session.post(
        LOCAL_CHAT_COMPLETIONS_URL,
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            #"response_format": {"type": "json_object"},
        },
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
        if "error" in data:
            raise ValueError(f"API error: {data['error']}")
        return data["choices"][0]["message"]["content"]

def extract_json(text):
    if text is None:
        raise ValueError("Input text is None")
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                # Clean up unescaped control characters and retry
                cleaned = re.sub(r'[\x00-\x1f\x7f]', ' ', match.group())
                return json.loads(cleaned)
        raise ValueError("Could not parse JSON")

def load_done_keys(filepath):
    done = set()
    if pathlib.Path(filepath).exists():
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    done.add((obj["id"], obj["candidate_idx"]))
    return done

_file_locks: dict[str, asyncio.Lock] = {}

def get_file_lock(filepath: str) -> asyncio.Lock:
    if filepath not in _file_locks:
        _file_locks[filepath] = asyncio.Lock()
    return _file_locks[filepath]

async def process_candidate(session, entry, candidate_idx, explanation,
                            model_cfg, answer_key, semaphore, counters):
    model_name  = model_cfg["model"]
    output_file = model_cfg["output_file"]
    entry_id    = entry["id"]
    key         = (entry_id, candidate_idx)

    ground_truth = answer_key.get(key)
    if ground_truth is None:
        return

    async with semaphore:
        try:
            raw_output = await call_llm(session, model_name, build_prompt(entry, explanation))
        except Exception as e:
            print(f"[{model_name}] HTTP error on {entry_id} candidate {candidate_idx}: {e}")
            return

        try:
            result = extract_json(raw_output)
        except Exception:
            print(f"[{model_name}] Failed to parse JSON for {entry_id} c{candidate_idx}. Raw: {(raw_output or '')[:200]}")
            return

        predicted_label = result.get("label")
        if predicted_label is None:
            print(f"[{model_name}] Missing label for {entry_id} candidate {candidate_idx}")
            return

        is_correct = predicted_label == ground_truth

        record = json.dumps({
            "id":            entry_id,
            "model":         model_name,
            "candidate_idx": candidate_idx,
            "predicted":     predicted_label,
            "ground_truth":  ground_truth,
            "correct":       is_correct,
        }) + "\n"

        async with get_file_lock(output_file):
            with open(output_file, "a") as out:
                out.write(record)

        if is_correct:
            counters[model_name]["correct"] += 1
        counters[model_name]["total"] += 1
        print(
            f"[{model_name}] [{counters[model_name]['total']}] "
            f"{entry_id} c{candidate_idx} | "
            f"Pred: {predicted_label} GT: {ground_truth} | Correct: {is_correct}"
        )

async def run_model(session, model_cfg, entries, answer_key, semaphore, counters):
    model_name  = model_cfg["model"]
    output_file = model_cfg["output_file"]

    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    done_keys = load_done_keys(output_file)

    # Seed counters from already-completed results
    correct_so_far = 0
    if pathlib.Path(output_file).exists():
        with open(output_file) as f:
            for line in f:
                if line.strip() and json.loads(line).get("correct"):
                    correct_so_far += 1

    counters[model_name] = {"total": len(done_keys), "correct": correct_so_far}
    print(f"[{model_name}] Resuming — {len(done_keys)} candidate evaluations already done.")

    # Expand entries into (entry, candidate_idx, explanation) tasks, skipping done ones
    tasks = []
    for entry in entries:
        for candidate_idx, explanation in enumerate(entry["candidates"]):
            if (entry["id"], candidate_idx) not in done_keys:
                tasks.append(
                    process_candidate(
                        session, entry, candidate_idx, explanation,
                        model_cfg, answer_key, semaphore, counters,
                    )
                )

    print(f"[{model_name}] {len(tasks)} candidate evaluations to process.")
    await asyncio.gather(*tasks)

async def main():
    # Load answer key once, shared across all models
    answer_key = {}
    with open(ANSWER_KEY_FILE) as f:
        for line in f:
            obj = json.loads(line)
            for idx, order_val in enumerate(obj["order"]):
                answer_key[(obj["id"], idx)] = 2 - order_val

    with open(DATA_FILE) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    entries = [e for e in entries if not e["id"].startswith("real_")]
    print(f"Loaded {len(entries)} entries from {DATA_FILE}.\n")

    # entries = entries[::15]

    counters = {}
    semaphore = asyncio.Semaphore(CONCURRENCY)

    session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=LOCAL_REQUEST_TIMEOUT_SECONDS)
    )
    try:
        await asyncio.gather(*[
            run_model(session, model_cfg, entries, answer_key, semaphore, counters)
            for model_cfg in MODELS
        ])
    finally:
        await session.close()

    print("\n=== Final counts ===")
    for model_cfg in MODELS:
        name = model_cfg["model"]
        c = counters.get(name, {})
        total   = c.get("total", 0)
        correct = c.get("correct", 0)
        acc = f"{correct/total:.2f}" if total else "N/A"
        print(f"  {name}: {correct}/{total} = {acc}")

if __name__ == "__main__":
    asyncio.run(main())
