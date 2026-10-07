# CoA-Spanish-Regional

**Do LLMs Speak Your Spanish? A Chain-of-Agents Approach to Regional Variation in Spanish**

Kyungjin Hwang — Korea University

This repository contains the lexical resources, preference rules, and evaluation scripts accompanying the paper submitted to *Language Resources and Evaluation* (Springer Nature).

---

## Repository Structure

```
coa-spanish-regional/
├── data/
│   └── prefer_map.yaml          # Regional preference map (350+ entries)
├── scripts/
│   ├── coa_agents.py            # CoA agent implementations (Planner, Retriever, Generator, Auditor)
│   ├── dialect_runtime.py       # Rules loading, auditor scoring, hard constraints
│   ├── tag_lexicon.py           # Lexicon loading, indexing, and text matching
│   ├── quality_heuristics.py    # Quality scoring heuristics
│   ├── run_coa.py               # Main CoA pipeline execution (single-utterance)
│   ├── run_baseline_b1.py       # Baseline (B1) evaluation
│   └── my_llm_client.py         # LLM API wrapper
├── evaluation/
│   ├── run_multiturn_eval.py    # Multi-turn dialogue evaluation
│   ├── flatten_multiturn.py     # Convert nested output format to flat JSONL
│   └── eval_multiturn_consistency.py  # Session Target Share Mean computation
├── LICENSE                       # CC BY 4.0
└── README.md
```

## Data

### `prefer_map.yaml`

A regional dialect preference map with 350+ entries covering five major Spanish varieties:

| Code | Variety |
|------|---------|
| ES   | Peninsular Spanish (Spain) |
| MX   | Mexican Spanish |
| AR   | Rioplatense Spanish (Argentina) |
| CL   | Chilean Spanish |
| PE   | Peruvian Spanish |

Each entry maps a canonical Spanish headword to region-specific preferred forms, register labels, and usage notes. This file drives the Planner and Retriever agents in the CoA pipeline and supplies the Auditor's scoring lexicon.

**Example entry:**

```yaml
ordenador:
  gloss: "computer"
  variants:
    ES: { form: ordenador, register: standard }
    MX: { form: computadora, register: standard }
    AR: { form: computadora, register: standard }
    CL: { form: computador, register: standard }
    PE: { form: computadora, register: standard }
```

## Pipeline Overview

The Chain-of-Agents (CoA) framework comprises four agents:

1. **Planner** — Identifies region-relevant lexical targets from the preference map
2. **Retriever** — Retrieves dialectal examples and usage patterns (CoA-R / CoA-H only)
3. **Generator** — Produces candidate utterances via LLM (GPT-4o, n=6, temp=0.8)
4. **Auditor** — Scores and re-ranks candidates using:

   *Score = 0.45 * DialectProb + 0.25 * LexiconHit - 0.20 * BannedHit - 0.15 * SoftBan + 0.10 * Quality*

Four system variants are evaluated:

| System | Components |
|--------|-----------|
| B1     | Single regional prompt (no pipeline) |
| CoA-S  | Planner + Generator + Auditor |
| CoA-R  | Planner + Retriever + Generator + Auditor |
| CoA-H  | CoA-R + post-generation hard constraints |

## Usage

### Requirements

```
pip install openai pyyaml
```

### Single-utterance evaluation

```bash
python scripts/run_coa.py \
  --prefer_map data/prefer_map.yaml \
  --region MX \
  --prompt "Responde como un asistente amable."
```

### Multi-turn evaluation

```bash
# 1. Generate multi-turn outputs
python evaluation/run_multiturn_eval.py \
  --sessions data/multiturn_sessions.jsonl \
  --rules data/rules.yml \
  --out out/multiturn_outputs.jsonl

# 2. Flatten nested output format
python evaluation/flatten_multiturn.py \
  --input out/multiturn_outputs.jsonl \
  --output out/multiturn_outputs_flat.jsonl

# 3. Compute Session Target Share Mean
python evaluation/eval_multiturn_consistency.py \
  --sessions data/multiturn_sessions.jsonl \
  --outputs out/multiturn_outputs.jsonl \
  --out_csv reports/consistency_results.csv
```

## Citation

If you use this resource, please cite:

```bibtex
@article{hwang2026coa,
  title={Do LLMs Speak Your Spanish? A Chain-of-Agents Approach to Regional Variation in Spanish},
  author={Hwang, Kyungjin},
  journal={Language Resources and Evaluation},
  year={2026},
  publisher={Springer Nature}
}
```

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
