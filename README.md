# reasoning-benchmarks
A reproducible harness for evaluating LLM reasoning strategies (CoT, Self-Consistency, ToT, etc.) across benchmarks like GSM8K, ARC-Challenge, and MMLU. Supports OpenAI, Hugging Face, and Ollama backends with unified metrics and plots.

# reasoning-benchmarks

This is my playground for stress-testing LLM reasoning. I wanted a clean, reproducible harness where I can compare different prompting strategies across standard reasoning benchmarks.

## What it does
- Runs **Chain-of-Thought (CoT)**, **Self-Consistency (SC-CoT)**, and **Tree-of-Thoughts (ToT)** prompting.
- Supports multiple backends out of the box: **OpenAI**, **Hugging Face Transformers**, and **Ollama**.
- Benchmarks on the big three reasoning datasets: **GSM8K**, **ARC-Challenge**, and **MMLU**.
- Logs results to JSONL and computes unified metrics + plots.

## Why I built it
I kept finding myself hacking together one-off scripts to test reasoning prompts. Now I have one place to run all of them, compare results, and extend with new ideas. It’s simple enough for me to use on a daily basis but structured enough to share.

## MMLU subjects included
MMLU covers 57 subjects across STEM, humanities, social sciences, and more. In this repo, I include loaders for all of them:

```
Abstract Algebra, Anatomy, Astronomy, Business Ethics, 
Clinical Knowledge, College Biology, College Chemistry, 
College Computer Science, College Mathematics, College Medicine, 
College Physics, Computer Security, Conceptual Physics, Econometrics, 
Electrical Engineering, Elementary Mathematics, Formal Logic, 
Global Facts, High School Biology, High School Chemistry, 
High School Computer Science, High School European History, 
High School Geography, High School Government and Politics, 
High School Macroeconomics, High School Mathematics, 
High School Microeconomics, High School Physics, High School Psychology, 
High School Statistics, High School US History, High School World History, 
Human Aging, Human Sexuality, International Law, Jurisprudence, 
Logical Fallacies, Machine Learning, Management, Marketing, 
Medical Genetics, Miscellaneous, Moral Disputes, Moral Scenarios, 
Nutrition, Philosophy, Prehistory, Professional Accounting, 
Professional Law, Professional Medicine, Professional Psychology, 
Public Relations, Security Studies, Sociology, US Foreign Policy, 
Virology, World Religions.
```
## Requirements
- Python 3.10+
- Install runtime deps:
```bash
pip install -r requirements.txt
```
- Optional dev deps:
```bash
pip install -r requirements-dev.txt
```
- API keys:
  - **OpenAI** → set `OPENAI_API_KEY`
  - **Hugging Face** (optional) → set `HUGGINGFACEHUB_API_TOKEN`
  - **Ollama** (optional) → run `ollama serve`

## Quickstart

Run GSM8K with OpenAI (CoT):
```bash
python -m benchkit.runners.gsm8k_runner \
  --engine openai --engine-config configs/openai.yaml \
  --prompt-style cot --max-samples 50 \
  --out results/gsm8k_openai_cot.jsonl
```

ARC-Challenge with HF (Self-Consistency, k=7):
```bash
python -m benchkit.runners.arc_runner \
  --engine hf --engine-config configs/hf_llama.yaml \
  --prompt-style sc --k 7 --max-samples 50 \
  --out results/arc_hf_sc.jsonl
```

MMLU with Ollama (ToT, breadth=3, depth=2):
```bash
python -m benchkit.runners.mmlu_runner \
  --engine ollama --engine-config configs/ollama.yaml \
  --prompt-style tot --breadth 3 --depth 2 \
  --subjects math,physics,philosophy \
  --max-samples 50 \
  --out results/mmlu_ollama_tot.jsonl
```

## License
MIT