import torch

from model import ModelConfig, Transformer


def tiny_config(**overrides) -> ModelConfig:
    base = dict(vocab_size=256, dim=32, n_layers=2, n_heads=4, n_kv_heads=2,
                mlp_ratio=2.0, mult_of=8, max_seq_len=64)
    base.update(overrides)
    return ModelConfig(**base)


def test_forward_logits_shape():
    cfg = tiny_config()
    model = Transformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(tokens)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is None


def test_forward_with_loss():
    cfg = tiny_config()
    model = Transformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(tokens, targets=targets)
    assert loss is not None
    assert loss.dim() == 0
    assert loss.item() > 0
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g == g for g in grad_norms)  # no NaNs


def test_loss_mask_zeroes_masked_positions():
    cfg = tiny_config()
    model = Transformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    targets = torch.randint(0, cfg.vocab_size, (1, 8))

    all_mask = torch.ones(1, 8)
    half_mask = torch.zeros(1, 8)
    half_mask[:, 4:] = 1.0

    torch.manual_seed(0)
    _, loss_all = model(tokens, targets=targets, loss_mask=all_mask)
    _, loss_half = model(tokens, targets=targets, loss_mask=half_mask)
    assert loss_all.item() != loss_half.item()  # masking actually changes the computed loss


def test_gqa_requires_divisible_heads():
    import pytest

    with pytest.raises(ValueError):
        tiny_config(n_heads=5, n_kv_heads=2)


def test_tied_embeddings_share_storage():
    cfg = tiny_config(tie_embeddings=True)
    model = Transformer(cfg)
    assert model.lm_head.weight.data_ptr() == model.tok_embeddings.weight.data_ptr()


def test_generate_produces_requested_length_and_is_deterministic_at_temp0():
    cfg = tiny_config()
    model = Transformer(cfg)
    model.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))

    out1 = model.generate(prompt, max_new_tokens=5, temperature=0.0)
    out2 = model.generate(prompt, max_new_tokens=5, temperature=0.0)

    assert out1.shape == (1, 8)
    assert torch.equal(out1, out2)  # greedy decoding is deterministic


def test_generate_matches_noncached_forward_logits():
    """The KV-cache path must produce the same next-token logits as a plain
    full-sequence forward pass -- this is the property that makes caching a
    pure speed optimization rather than a behavior change."""
    cfg = tiny_config()
    model = Transformer(cfg)
    model.eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 6))

    with torch.no_grad():
        full_logits, _ = model(tokens)

        caches = model.init_caches(1)
        step_logits, _ = model(tokens[:, :1], caches=caches)
        for t in range(1, 6):
            step_logits, _ = model(tokens[:, t : t + 1], caches=caches)

    # last step's logits (predicting the token after the full prompt) should match
    assert torch.allclose(full_logits[:, -1, :], step_logits[:, -1, :], atol=1e-4)


def test_num_params_excludes_embeddings_by_default():
    cfg = tiny_config()
    model = Transformer(cfg)
    total = sum(p.numel() for p in model.parameters())
    non_emb = model.num_params(non_embedding=True)
    assert non_emb < total
    assert model.num_params(non_embedding=False) == total
