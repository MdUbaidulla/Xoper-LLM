"""FastAPI inference server: loads one checkpoint, serves /generate (and
optionally /generate_rag), with request logging for later evaluation.

Run: uvicorn serve.api:app --host 0.0.0.0 --port 8000
Configure the checkpoint/tokenizer paths via env vars (see configs/serve.yaml
for the equivalent fields if you prefer to load from YAML in a wrapper script).
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from model import ModelConfig, Transformer
from serve.kv_cache_generate import generate_with_cache
from train.utils import load_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serve.api")

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint_path = os.environ.get("LLM_CHECKPOINT", "checkpoints/sft/final.pt")
    tokenizer_path = os.environ.get("LLM_TOKENIZER", "tokenizer.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from tokenizers import Tokenizer

    payload = torch.load(checkpoint_path, map_location=device)
    model_cfg = ModelConfig(**payload["extra"]["model_cfg"])
    model = Transformer(model_cfg).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    STATE["model"] = model
    STATE["tokenizer"] = Tokenizer.from_file(tokenizer_path)
    STATE["device"] = device
    logger.info("loaded %s on %s (%.1fM non-embedding params)",
                checkpoint_path, device, model.num_params() / 1e6)
    yield
    STATE.clear()


app = FastAPI(title="llm-project inference", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 40


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    tokens_per_second: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    t0 = time.time()
    result = generate_with_cache(
        STATE["model"], STATE["tokenizer"], req.prompt,
        max_new_tokens=req.max_new_tokens, temperature=req.temperature,
        top_k=req.top_k, device=STATE["device"],
    )
    logger.info("prompt_len=%d new_tokens=%d latency=%.3fs",
                len(req.prompt), result["tokens_generated"], time.time() - t0)
    return GenerateResponse(
        text=result["text"],
        tokens_generated=result["tokens_generated"],
        tokens_per_second=result["tokens_per_second"],
    )
