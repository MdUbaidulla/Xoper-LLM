import json
import os

from fastapi.testclient import TestClient

from data.tokenizer_train import train_tokenizer
from data.shard import shard_corpus
from model import ModelConfig, Transformer
from train.pretrain import train as pretrain_run


def _tiny_model_cfg_dict():
    return dict(vocab_size=512, dim=32, n_layers=2, n_heads=4, n_kv_heads=2,
                mlp_ratio=2.0, mult_of=8, max_seq_len=64)


def test_health_and_generate_endpoints(tmp_path, monkeypatch):
    text_dir = tmp_path / "cleaned"
    text_dir.mkdir()
    (text_dir / "doc.txt").write_text(
        "The quick brown fox jumps over the lazy dog. " * 40, encoding="utf-8"
    )
    tok_path = str(tmp_path / "tokenizer.json")
    train_tokenizer(str(text_dir / "*.txt"), vocab_size=512, out_path=tok_path)

    shard_dir = str(tmp_path / "shards")
    shard_corpus(str(text_dir / "*.txt"), tok_path, shard_dir, shard_tokens=2000)

    ckpt_dir = str(tmp_path / "ckpt")
    pretrain_run({
        "model": _tiny_model_cfg_dict(),
        "data": {"shard_dir": shard_dir},
        "train": {"micro_batch_size": 2, "grad_accum_steps": 1, "total_steps": 2,
                   "warmup_steps": 1, "max_lr": 1e-3, "min_lr": 1e-4, "weight_decay": 0.0,
                   "grad_clip": 1.0, "dtype": "fp32", "num_workers": 0, "log_every": 1,
                   "ckpt_every": 100, "ckpt_dir": ckpt_dir},
        "seed": 0,
    })
    ckpt_path = os.path.join(ckpt_dir, "final.pt")

    monkeypatch.setenv("LLM_CHECKPOINT", ckpt_path)
    monkeypatch.setenv("LLM_TOKENIZER", tok_path)

    from serve.api import app
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        resp = client.post("/generate", json={"prompt": "The quick brown", "max_new_tokens": 8})
        assert resp.status_code == 200
        body = resp.json()
        assert body["tokens_generated"] > 0
        assert isinstance(body["text"], str) and len(body["text"]) > 0
