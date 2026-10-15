import torch
from torch import nn

from model import ModelConfig, Transformer
from train.lora import LoRALinear, inject_lora, lora_trainable_parameters


def test_lora_linear_matches_base_at_init():
    """lora_B starts at zero, so a freshly-wrapped layer must be numerically
    identical to the un-wrapped base layer."""
    base = nn.Linear(16, 8, bias=False)
    wrapped = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 16)
    assert torch.allclose(base(x), wrapped(x), atol=1e-6)


def test_inject_lora_freezes_base_and_leaves_adapters_trainable():
    cfg = ModelConfig(vocab_size=128, dim=32, n_layers=2, n_heads=4, n_kv_heads=2,
                       mlp_ratio=2.0, mult_of=8, max_seq_len=32)
    model = Transformer(cfg)
    model = inject_lora(model, target_substrings=("wq", "wv"), r=4, alpha=8)

    trainable = lora_trainable_parameters(model)
    assert len(trainable) > 0

    for name, p in model.named_parameters():
        if "lora_" in name:
            assert p.requires_grad
        else:
            assert not p.requires_grad

    # wq/wv got wrapped; wk/wo did not
    assert any(isinstance(m, LoRALinear) for n, m in model.named_modules() if n.endswith("wq"))
    assert any(isinstance(m, LoRALinear) for n, m in model.named_modules() if n.endswith("wv"))
    assert not any(isinstance(m, LoRALinear) for n, m in model.named_modules() if n.endswith("wk"))


def test_lora_forward_and_backward_through_full_model():
    cfg = ModelConfig(vocab_size=128, dim=32, n_layers=2, n_heads=4, n_kv_heads=2,
                       mlp_ratio=2.0, mult_of=8, max_seq_len=32)
    model = Transformer(cfg)
    model = inject_lora(model, r=4, alpha=8)

    tokens = torch.randint(0, cfg.vocab_size, (2, 10))
    targets = torch.randint(0, cfg.vocab_size, (2, 10))
    _, loss = model(tokens, targets=targets)
    loss.backward()

    for p in lora_trainable_parameters(model):
        assert p.grad is not None
