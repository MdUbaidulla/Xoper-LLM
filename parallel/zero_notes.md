# Scaling past data parallelism

`ddp.py` covers data parallelism (DP): every worker holds a full model
copy and syncs gradients by All-Reduce. That's enough as long as one
model copy — weights, gradients and optimizer states — fits on one
accelerator. Past that point, add these in order:

1. **ZeRO (stages 1-3).** Shards optimizer states, then gradients,
   then parameters across the data-parallel group, removing the
   redundant copies DDP keeps on every worker. Use
   [DeepSpeed](https://www.deepspeed.ai/) or PyTorch's native FSDP
   (`torch.distributed.fsdp.FullyShardedDataParallel`) — both drop in
   wherever `DistributedState.wrap_ddp` wraps the model in `pretrain.py`.
2. **Tensor parallelism (TP).** Splits individual weight matrices
   inside one attention or MLP block column-/row-wise across GPUs
   (Megatron-LM style). Needed once a single layer's matrices don't
   fit, or to cut latency within a layer. Requires fast intra-node
   interconnect, since it forces communication mid-layer.
3. **Pipeline parallelism (PP).** Splits layers across GPUs
   (e.g. GPU 0 = layers 1-20, GPU 1 = layers 21-40), with micro-batch
   scheduling (1F1B or similar) to keep every stage busy.

A production run typically combines all four (3D parallelism + ZeRO).
This repo's training loops (`train/pretrain.py`) are written against a
plain `nn.Module` / DDP-module interface precisely so that swapping in
DeepSpeed or FSDP changes `parallel/`, not the step logic in `train/`.

Also see, already implemented here:

- **Mixed precision (BF16):** `train/pretrain.py` autocasts under
  `torch.bfloat16` on CUDA, matching FP32's dynamic range at half the
  memory and without FP16's underflow/overflow issues.
- **FlashAttention:** `model/attention.py` calls
  `F.scaled_dot_product_attention`, which dispatches to a fused
  FlashAttention kernel on supported GPUs automatically — no separate
  integration needed.
