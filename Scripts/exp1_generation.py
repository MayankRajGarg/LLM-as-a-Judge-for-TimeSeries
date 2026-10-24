import json
import re
import asyncio
import aiohttp
import pathlib

API_KEY   = ""
DATA_FILE = "Data/synthetic.jsonl"

# Models and their output files.
MODELS = [
    #{"model":       "qwen/qwen3-8b", "output_file": "Results/Qwen/Experiment_1.jsonl", "max_series_len": None,},
    #{"model":       "meta-llama/llama-3.1-8b-instruct",  "output_file": "Results/Llama/Experiment_1.jsonl", "max_series_len": None,},
    #{"model":       "google/gemma-4-31b-it", "output_file": "Results/Gemma/Experiment_1.jsonl", "max_series_len": None,},
    {"model":       "google/gemma-3-12b-it", "output_file": "Results/Gemma3/Experiment_1.jsonl", "max_series_len": None,},
    #{"model": "anthropic/claude-3-haiku",      "output_file": "Results/Haiku/Experiment_1.jsonl",    "max_series_len": None},
    #{"model": "deepseek/deepseek-v3.2",        "output_file": "Results/Deepseek/Experiment_1.jsonl", "max_series_len": None},
]

# Max total concurrent requests across ALL models (tune to stay within rate limits)
CONCURRENCY = 15

GENERATE_TEMPLATE = """You are an expert analyst of time-series data.
You will receive:
- A time series (index: value pairs)
- A question about the time series

Your goal: write a concise explanation that directly answers the question.

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
Inspect the data carefully to find evidence relevant to the question.
For percentage changes: compute manually as ((new - old) / |old|) × 100.
For anomalies: identify the exact index t and verify the value deviates sharply and returns afterward.
For regime shifts: identify where the mean level changes and stays changed.
For trends: assess direction and approximate rate across the full series.

PHASE 3 — WRITE THE EXPLANATION
Write a single concise explanation that:
- Directly answers the question
- References the specific time index and values from the data
- Includes any relevant numeric quantities (percentage change, magnitude, etc.)
- Does not introduce claims not supported by the data

Return ONLY valid JSON. No extra text. No markdown fences.
{"explanation": "your explanation here"}
"""

def format_series(series, precision=2):
    return ", ".join(f"{p['t']}:{round(p['value'], precision)}" for p in series)

def build_prompt(entry):
    return (
        GENERATE_TEMPLATE
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

def load_done_ids(filepath):
    done = set()
    if pathlib.Path(filepath).exists():
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["id"])
    return done

# File write lock per output path so concurrent tasks don't interleave writes
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
            raw_output = await call_llm(session, model_name, build_prompt(entry))
        except Exception as e:
            print(f"[{model_name}] HTTP error on {entry_id}: {e}")
            return

        try:
            result = extract_json(raw_output)
        except Exception:
            print(f"[{model_name}] Failed to parse JSON for {entry_id}. Raw: {raw_output[:200]}")
            return

        explanation = result.get("explanation")
        if not explanation:
            print(f"[{model_name}] Missing explanation for {entry_id}")
            return

        record = json.dumps({
            "id":          entry_id,
            "model":       model_name,
            "question":    entry["question"],
            "explanation": explanation,
        }) + "\n"

        async with get_file_lock(output_file):
            with open(output_file, "a") as out:
                out.write(record)

        counters[model_name] += 1
        print(f"[{model_name}] [{counters[model_name]}] Processed {entry_id}")

async def run_model(session, model_cfg, entries, counters, semaphore):
    model_name     = model_cfg["model"]
    output_file    = model_cfg["output_file"]
    max_series_len = model_cfg["max_series_len"]

    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(output_file)
    print(f"[{model_name}] Resuming — {len(done_ids)} entries already done.")
    counters[model_name] = len(done_ids)

    # Filter to only entries this model still needs to process
    pending = []
    skipped = 0
    for entry in entries:
        if entry["id"] in done_ids:
            continue
        if max_series_len is not None and len(entry["series"]) > max_series_len:
            skipped += 1
            continue
        pending.append(entry)

    if skipped:
        print(f"[{model_name}] Skipped {skipped} entries exceeding max series length {max_series_len}.")
    print(f"[{model_name}] {len(pending)} entries to process.")

    tasks = [
        process_entry(session, entry, model_cfg, semaphore, counters)
        for entry in pending
    ]
    await asyncio.gather(*tasks)

async def main():
    with open(DATA_FILE) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    print(f"Loaded {len(entries)} entries from {DATA_FILE}.\n")

    counters = {}
    # One shared pool — when a model finishes early its slots flow to remaining models
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        # All models share one semaphore pool — freed slots flow to whichever model still has work
        await asyncio.gather(*[
            run_model(session, model_cfg, entries, counters, semaphore)
            for model_cfg in MODELS
        ])

    print("\n=== Final counts ===")
    for model_cfg in MODELS:
        name = model_cfg["model"]
        print(f"  {name}: {counters.get(name, 0)} total explanations")

if __name__ == "__main__":
    asyncio.run(main())