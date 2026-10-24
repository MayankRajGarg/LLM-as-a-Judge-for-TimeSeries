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
        "output_file": "Results/New_Qwen/RealData_New_Experiment_2.jsonl",
    },
]
 
# Total concurrent requests shared across all models
CONCURRENCY = 1
 
JUDGE_TEMPLATE = """
You are an expert evaluator of time-series explanations.
You will receive:
- A time series (index: value pairs)
- A question about the time series
- Three candidate explanations labeled 0, 1, 2
Your goal: score each candidate using the rubric and select the best one.
 
PHASE 1 — UNDERSTAND THE QUESTION
Identify what the question is asking for:
- Anomaly / spike detection (sudden temporary deviation)
- Regime shift (sustained step change in level)
- Trend (gradual increase or decrease over many steps)
- Volatility change (increase or decrease in variance)
- Extreme value (maximum or minimum)
- Percentage or magnitude of change between specific points
- General behavior description
Determine the question type before doing anything else.
 
PHASE 2 — ANALYZE THE TIME SERIES
Inspect the data to find evidence relevant to the question type.
For percentage changes: compute manually as ((new - old) / |old|) × 100.
CRITICAL: Do NOT trust the numbers in candidate explanations.
Compute or verify all numeric claims yourself against the raw data.
 
PHASE 3 — EVALUATE EACH CANDIDATE USING THE RUBRIC
For each candidate, assess these five dimensions internally:
[A] DATA FAITHFULNESS: Does it correctly identify the right pattern, location, and direction?
[B] NUMERIC ACCURACY: Are its numeric claims correct within ±2% tolerance?
[C] QUESTION RELEVANCE: Does it directly and completely answer the question?
[D] LOGICAL COHERENCE: Is its reasoning internally consistent?
[E] UNSUPPORTED CLAIMS: Does it introduce claims not derivable from the data?
 
PHASE 4 — ASSIGN TERNARY LABEL
Use your rubric assessment to assign exactly one label per candidate:
2 = Fully correct (faithful, accurate numbers, answers the question, no unsupported claims)
1 = Correct reasoning and direction but contains numeric errors or minor unsupported claims
0 = Completely incorrect (wrong pattern, wrong location, or fundamentally contradicts the data)
 
PHASE 5 — SELECT BEST CANDIDATE
Choose the candidate with the highest label.
If tied, prefer the candidate whose numeric claims are closer to the actual data values.
 
Return ONLY valid JSON. No explanation. No extra text. No markdown fences.
{
  "scores": {
    "0": {"label": int},
    "1": {"label": int},
    "2": {"label": int}
  },
  "best_index": int
}
"""
 
def format_series(series, precision=2):
    return ", ".join(f"{p['t']}:{round(p['value'], precision)}" for p in series)
 
def build_prompt(entry):
    return (
        JUDGE_TEMPLATE
        + "\n\n"
        + f"Time Series:\n{format_series(entry['series'])}\n\n"
        + f"Question:\n{entry['question']}\n\n"
        + f"Candidate 0:\n{entry['candidates'][0]}\n\n"
        + f"Candidate 1:\n{entry['candidates'][1]}\n\n"
        + f"Candidate 2:\n{entry['candidates'][2]}\n"
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
    def _parse(s):
        parsed = json.loads(s)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}: {parsed}")
        return parsed
    try:
        return _parse(text)
    except Exception:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return _parse(match.group())
            except Exception:
                cleaned = re.sub(r'[\x00-\x1f\x7f]', ' ', match.group())
                return _parse(cleaned)
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
 
_file_locks: dict[str, asyncio.Lock] = {}
 
def get_file_lock(filepath: str) -> asyncio.Lock:
    if filepath not in _file_locks:
        _file_locks[filepath] = asyncio.Lock()
    return _file_locks[filepath]
 
async def process_entry(session, entry, model_cfg, answer_key, semaphore, counters):
    model_name  = model_cfg["model"]
    output_file = model_cfg["output_file"]
    entry_id    = entry["id"]
 
    ground_truth = answer_key.get(entry_id)
    if ground_truth is None:
        return
 
    async with semaphore:
        try:
            raw_output = await call_llm(session, model_name, build_prompt(entry))
        except Exception as e:
            print(f"[{model_name}] HTTP error on {entry_id}: {e}")
            return
 
        try:
            result = extract_json(raw_output)
        except Exception:
            print(f"[{model_name}] Failed to parse JSON for {entry_id}. Raw:\n{raw_output}")
            return
 
        predicted = result.get("best_index", -1)
        is_correct = predicted == ground_truth
 
        record = json.dumps({
            "id":           entry_id,
            "model":        model_name,
            "predicted":    predicted,
            "ground_truth": ground_truth,
            "correct":      is_correct,
            "scores":       result.get("scores"),
        }) + "\n"
 
        async with get_file_lock(output_file):
            with open(output_file, "a") as out:
                out.write(record)
 
        if is_correct:
            counters[model_name]["correct"] += 1
        counters[model_name]["total"] += 1
        print(f"[{model_name}] [{counters[model_name]['total']}] {entry_id} | Correct: {is_correct}")
 
async def run_model(session, model_cfg, entries, answer_key, semaphore, counters):
    model_name  = model_cfg["model"]
    output_file = model_cfg["output_file"]
 
    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
 
    done_ids = load_done_ids(output_file)
 
    # Seed counters from already-completed results
    correct_so_far = 0
    if pathlib.Path(output_file).exists():
        with open(output_file) as f:
            for line in f:
                if line.strip() and json.loads(line).get("correct"):
                    correct_so_far += 1
 
    counters[model_name] = {"total": len(done_ids), "correct": correct_so_far}
    print(f"[{model_name}] Resuming — {len(done_ids)} entries already done.")
 
    pending = [e for e in entries if e["id"] not in done_ids]
    print(f"[{model_name}] {len(pending)} entries to process.")
 
    tasks = [
        process_entry(session, entry, model_cfg, answer_key, semaphore, counters)
        for entry in pending
    ]
    await asyncio.gather(*tasks)
 
async def main():
    # Load answer key once, shared across all models
    answer_key = {}
    with open(ANSWER_KEY_FILE) as f:
        for line in f:
            obj = json.loads(line)
            try:
                answer_key[obj["id"]] = obj["order"].index(0)
            except Exception:
                answer_key[obj["id"]] = None
 
    with open(DATA_FILE) as f:
        entries = [json.loads(l) for l in f if l.strip()]
 
    # Exclude real_ entries (same as original)
    entries = [e for e in entries if not e["id"].startswith("real_")]
    print(f"Loaded {len(entries)} entries from {DATA_FILE}.\n")

    # entries = entries[::10]
    counters  = {}
    semaphore = asyncio.Semaphore(CONCURRENCY)
 
    # Fix: create session manually so it only closes after all tasks fully complete
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
