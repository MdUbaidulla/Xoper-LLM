"""End-to-end smoke tests that exercise the actual CLI-facing scripts
(not just the library functions they call), on a tiny synthetic corpus
and a few training steps -- catches wiring bugs the unit tests can't see.
"""
import json
import os

import numpy as np
import torch
import yaml

from data.tokenizer_train import train_tokenizer
from data.shard import shard_corpus
from train.pretrain import train as pretrain_run
from train.sft import train as sft_run
from train.dpo import train as dpo_run
from eval.perplexity import evaluate_perplexity
from eval.generate_eval import generate_completions
from eval.safety_check import run_safety_probes
from serve.quantize import quantize_dynamic_int8
from serve.kv_cache_generate import generate_with_cache
from model import ModelConfig, Transformer
from train.utils import load_checkpoint


def _make_corpus_and_tokenizer(tmp_path):
    text_dir = tmp_path / "cleaned"
    text_dir.mkdir()
    for i in range(3):
        (text_dir / f"doc{i}.txt").write_text(
            ("The quick brown fox jumps over the lazy dog. " * 40 +
             "Transformers predict the next token given the previous tokens. " * 40),
            encoding="utf-8",
        )
    tok_path = str(tmp_path / "tokenizer.json")
    train_tokenizer(str(text_dir / "*.txt"), vocab_size=512, out_path=tok_path)
    return str(text_dir / "*.txt"), tok_path


def _tiny_model_cfg_dict():
    return dict(vocab_size=512, dim=32, n_layers=2, n_heads=4, n_kv_heads=2,
                mlp_ratio=2.0, mult_of=8, max_seq_len=64)


def test_shard_then_pretrain_smoke(tmp_path):
    glob_pat, tok_path = _make_corpus_and_tokenizer(tmp_path)
    shard_dir = str(tmp_path / "shards")
    shard_corpus(glob_pat, tok_path, shard_dir, shard_tokens=2000)
    assert os.path.exists(os.path.join(shard_dir, "meta.txt"))

    ckpt_dir = str(tmp_path / "ckpt" / "pretrain")
    cfg = {
        "model": _tiny_model_cfg_dict(),
        "data": {"shard_dir": shard_dir},
        "train": {
            "micro_batch_size": 2, "grad_accum_steps": 1, "total_steps": 3,
            "warmup_steps": 1, "max_lr": 1e-3, "min_lr": 1e-4, "weight_decay": 0.0,
            "grad_clip": 1.0, "dtype": "fp32", "num_workers": 0, "log_every": 1,
            "ckpt_every": 2, "ckpt_dir": ckpt_dir,
        },
        "seed": 0,
    }
    pretrain_run(cfg)

    final_ckpt = os.path.join(ckpt_dir, "final.pt")
    assert os.path.exists(final_ckpt)

    # perplexity eval script runs against the checkpoint it just produced
    model_cfg = ModelConfig(**_tiny_model_cfg_dict())
    model = Transformer(model_cfg)
    load_checkpoint(final_ckpt, model)
    result = evaluate_perplexity(model, shard_dir, model_cfg.max_seq_len, batch_size=2, max_batches=2)
    assert result["perplexity"] > 0


def test_sft_then_dpo_then_serve_smoke(tmp_path):
    glob_pat, tok_path = _make_corpus_and_tokenizer(tmp_path)
    shard_dir = str(tmp_path / "shards")
    shard_corpus(glob_pat, tok_path, shard_dir, shard_tokens=2000)

    pretrain_ckpt_dir = str(tmp_path / "ckpt" / "pretrain")
    pretrain_run({
        "model": _tiny_model_cfg_dict(),
        "data": {"shard_dir": shard_dir},
        "train": {"micro_batch_size": 2, "grad_accum_steps": 1, "total_steps": 2,
                   "warmup_steps": 1, "max_lr": 1e-3, "min_lr": 1e-4, "weight_decay": 0.0,
                   "grad_clip": 1.0, "dtype": "fp32", "num_workers": 0, "log_every": 1,
                   "ckpt_every": 100, "ckpt_dir": pretrain_ckpt_dir},
        "seed": 0,
    })
    pretrain_ckpt = os.path.join(pretrain_ckpt_dir, "final.pt")

    # --- SFT ---
    sft_jsonl = tmp_path / "sft.jsonl"
    sft_rows = [
        {"prompt": "What does a fox do?", "response": "It jumps over the lazy dog."},
        {"prompt": "What do transformers predict?", "response": "The next token."},
    ]
    sft_jsonl.write_text("\n".join(json.dumps(r) for r in sft_rows), encoding="utf-8")

    sft_ckpt_dir = str(tmp_path / "ckpt" / "sft")
    sft_run({
        "model": _tiny_model_cfg_dict(),
        "init_from_pretrain": pretrain_ckpt,
        "data": {"sft_jsonl": str(sft_jsonl), "tokenizer_path": tok_path},
        "train": {"batch_size": 2, "epochs": 1, "max_lr": 1e-4, "weight_decay": 0.0,
                   "grad_clip": 1.0, "warmup_steps": 1, "log_every": 1, "ckpt_dir": sft_ckpt_dir},
        "seed": 0,
    })
    sft_ckpt = os.path.join(sft_ckpt_dir, "final.pt")
    assert os.path.exists(sft_ckpt)

    # --- DPO ---
    dpo_jsonl = tmp_path / "dpo.jsonl"
    dpo_rows = [
        {"prompt": "What does a fox do?", "chosen": "It jumps over the lazy dog.",
         "rejected": "I don't know anything about that."},
    ]
    dpo_jsonl.write_text("\n".join(json.dumps(r) for r in dpo_rows), encoding="utf-8")

    dpo_ckpt_dir = str(tmp_path / "ckpt" / "dpo")
    dpo_run({
        "model": _tiny_model_cfg_dict(),
        "init_from_sft": sft_ckpt,
        "data": {"dpo_jsonl": str(dpo_jsonl), "tokenizer_path": tok_path},
        "train": {"batch_size": 1, "epochs": 1, "max_lr": 1e-5, "beta": 0.1,
                   "grad_clip": 1.0, "log_every": 1, "ckpt_dir": dpo_ckpt_dir},
        "seed": 0,
    })
    dpo_ckpt = os.path.join(dpo_ckpt_dir, "final.pt")
    assert os.path.exists(dpo_ckpt)

    # --- eval scripts against the real checkpoint ---
    completions = generate_completions(dpo_ckpt, tok_path, ["What does a fox do?"], max_new_tokens=8)
    assert len(completions) == 1 and len(completions[0]["completion"]) > 0

    probes = run_safety_probes(dpo_ckpt, tok_path, max_new_tokens=8)
    assert len(probes) == 4

    # --- quantize + serve path ---
    model_cfg = ModelConfig(**_tiny_model_cfg_dict())
    model = Transformer(model_cfg)
    load_checkpoint(dpo_ckpt, model)
    model.eval()
    q_model = quantize_dynamic_int8(model)
    assert q_model is not None

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    result = generate_with_cache(model, tok, "What does a fox do?", max_new_tokens=8, temperature=0.0)
    assert result["tokens_generated"] > 0
    assert isinstance(result["text"], str)
