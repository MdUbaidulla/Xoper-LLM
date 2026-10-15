# llm-project

A from-scratch, decoder-only Transformer LLM: data pipeline, the
three-stage training pipeline (pre-training, SFT, DPO alignment),
distributed-training hooks, evaluation, and a serving stack. Companion
code to the "LLM Project Architecture" doc — each section there maps
to a folder here.

## Layout

```
configs/        model, data, training and serving YAML
data/           ingest, filters, dedup, tokenizer, sharding
model/          attention, RoPE, norms, SwiGLU, transformer
train/          pretrain, sft, dpo, lora
parallel/       DDP setup + notes on scaling to ZeRO/TP/PP
eval/           perplexity, qualitative generation, safety probes
serve/          quantization, KV-cache + speculative decoding, RAG, FastAPI
scripts/        launch/resume/export shell & Python entry points
tests/          unit tests (model shapes, RoPE, LoRA, tokenizer)
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Data pipeline

```python
from data.filters import filter_document, document_survives
from data.pii import scrub_pii
from data.dedup import deduplicate

docs = [scrub_pii(filter_document(raw)) for raw in raw_docs if document_survives(raw)]
keep_idx = deduplicate(docs, threshold=0.8)
docs = [docs[i] for i in keep_idx]
```

Train a byte-level BPE tokenizer, then shard the cleaned corpus into
binary token files:

```bash
python -m data.tokenizer_train --input "cleaned/*.txt" --vocab-size 32768 --out tokenizer.json
python -m data.shard --input "cleaned/*.txt" --tokenizer tokenizer.json --out-dir shards/train
```

## 2. Train

```bash
# Stage 1: pre-training (next-token prediction)
NPROC=1 scripts/launch_pretrain.sh configs/train_pretrain.yaml

# Stage 2: supervised fine-tuning (data_raw/sft_examples.jsonl: {"prompt", "response"} per line)
scripts/launch_sft.sh configs/train_sft.yaml

# Stage 3: DPO alignment (data_raw/dpo_pairs.jsonl: {"prompt", "chosen", "rejected"} per line)
scripts/launch_dpo.sh configs/train_dpo.yaml
```

`configs/model_small.yaml` (~40M params) runs on a laptop CPU for
end-to-end testing; `configs/model_1b.yaml` is the scale-up target —
see `parallel/zero_notes.md` for what to add (ZeRO/FSDP, tensor and
pipeline parallelism) once a model copy stops fitting on one GPU.

Domain fine-tuning without touching the base weights:

```python
from train.lora import inject_lora
model = inject_lora(model, target_substrings=("wq", "wv"), r=8, alpha=16)
```

## 3. Evaluate

```bash
python -m eval.perplexity --checkpoint checkpoints/pretrain/final.pt --shard-dir shards/val
python -m eval.generate_eval --checkpoint checkpoints/sft/final.pt --tokenizer tokenizer.json
python -m eval.safety_check --checkpoint checkpoints/dpo/final.pt --tokenizer tokenizer.json
```

## 4. Serve

```bash
python -m serve.quantize --checkpoint checkpoints/dpo/final.pt --out checkpoints/dpo/final.int8.pt
LLM_CHECKPOINT=checkpoints/dpo/final.pt LLM_TOKENIZER=tokenizer.json \
  uvicorn serve.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST localhost:8000/generate -H 'content-type: application/json' \
  -d '{"prompt": "Explain RoPE in one sentence.", "max_new_tokens": 64}'
```

RAG grounding (`serve/rag.py`) and speculative decoding
(`serve/kv_cache_generate.py::speculative_generate`) are available as
library functions to wire into `serve/api.py` once you have a corpus
or a small draft model, respectively.

## Tests

```bash
pytest tests/ -v
```

## Notes on scope

This is a working, readable reference implementation, not a
drop-in replacement for Megatron-LM/DeepSpeed/vLLM at trillion-token,
multi-datacenter scale. Where this repo stops short of that (tensor
and pipeline parallelism, PagedAttention-style batched serving,
calibrated 4-bit quantization), the relevant module says so and points
to the production tool that covers it — see in particular
`parallel/zero_notes.md` and `serve/quantize.py`.
