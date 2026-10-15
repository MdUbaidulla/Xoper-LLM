import torch

from model.rope import apply_rope, precompute_rope_freqs


def test_rope_preserves_vector_norm():
    """RoPE is a rotation, so it must not change vector magnitude."""
    head_dim, seq_len = 16, 10
    freqs = precompute_rope_freqs(head_dim, seq_len)
    q = torch.randn(1, seq_len, 2, head_dim)
    k = torch.randn(1, seq_len, 2, head_dim)

    q_rot, k_rot = apply_rope(q, k, freqs)

    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4)
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-4)


def test_rope_dot_product_depends_only_on_relative_distance():
    """The core RoPE property: <RoPE(q, i), RoPE(k, j)> depends only on (i - j),
    not on the absolute positions i and j."""
    head_dim = 8
    freqs = precompute_rope_freqs(head_dim, max_seq_len=50)

    torch.manual_seed(0)
    q_vec = torch.randn(1, 1, 1, head_dim)
    k_vec = torch.randn(1, 1, 1, head_dim)

    def rotated_dot(pos_q: int, pos_k: int) -> float:
        # build length-(max(pos)+1) sequences where only the target position is non-zero
        seq_len = max(pos_q, pos_k) + 1
        q_seq = torch.zeros(1, seq_len, 1, head_dim)
        k_seq = torch.zeros(1, seq_len, 1, head_dim)
        q_seq[0, pos_q, 0] = q_vec[0, 0, 0]
        k_seq[0, pos_k, 0] = k_vec[0, 0, 0]
        q_rot, k_rot = apply_rope(q_seq, k_seq, freqs)
        return (q_rot[0, pos_q, 0] * k_rot[0, pos_k, 0]).sum().item()

    # same relative distance (3), different absolute positions
    dot_a = rotated_dot(pos_q=5, pos_k=2)
    dot_b = rotated_dot(pos_q=20, pos_k=17)
    assert abs(dot_a - dot_b) < 1e-3


def test_rope_requires_even_head_dim():
    import pytest

    with pytest.raises(ValueError):
        precompute_rope_freqs(head_dim=7, max_seq_len=10)
