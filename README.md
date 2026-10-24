# TSQueryBench

Reference-free evaluation of time-series explanations using an LLM-as-a-Judge framework with structured rubric-based scoring.

---

## Overview

This repository contains the code, data, and results for **TSQueryBench**. We study large language models as both generators and evaluators of time-series explanations in a reference-free setting, using a rubric-guided prompting framework that conditions directly on raw numerical data.

---

## Repository Structure

```
LLM-Timeseries-Evaluation/
├── Data/       # Synthetic benchmark datasets (350 main instances + 100 anomaly detection instances)
├── Results/    # Experimental outputs and evaluation logs
├── Scripts/    # Experiment scripts (generation, ranking, scoring, anomaly detection)
├── Utils/      # Shared utilities for prompting, parsing, and metrics
└── LICENSE
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running Experiments

```bash
# Explanation Generation
python Scripts/generation.py 

# Relative Ranking
python Scripts/ranking.py 

# Independent Scoring
python Scripts/scoring.py

# Multi-Anomaly Detection
python Scripts/anomaly_detection.py 
```

All experiments are run zero-shot with structured JSON output. Refer to individual scripts for argument options and parameters.

---

## License

MIT — see [LICENSE](LICENSE) for details.
