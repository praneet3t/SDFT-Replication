# Self-Distillation Fine-Tuning (SDFT) Replication

This is an independent replication of the On-Policy Self-Distillation Fine-Tuning (SDFT) algorithm from the paper *"Self-Distillation Enables Continual Learning."*

The goal was to replicate the paper's findings on the `Qwen2.5-7B-Instruct` model, specifically for the Tool Use and Science Q&A tasks. Because I was running this on a shared HPC cluster with strict memory constraints, I had to modify the training pipeline to fit the 7B model + vLLM into a single node without OOMing.

## Results vs. The Paper

I evaluated the fine-tuned model on two of the paper's primary tasks. The results map closely to the original paper, confirming that the self-distillation loop works. Should try it out with other datasets.

| Dataset / Task | Paper's Base | Paper's Best SDFT | **My Run (SDFT)** | Delta from Paper |
| :--- | :---: | :---: | :---: | :---: |
| **Science Q&A** (SciKnowEval) | 32.1% | 70.2% | **68.64%** | -1.56% |
| **Tool Use** (ToolAlpaca) | 42.9% | 70.6% | **68.04%** | -2.56% |

*Note: The ~2% gap is likely due to running on a single GPU with a fixed batch size and skipping the extensive hyperparameter sweeps (like EMA alpha) that the authors did.*

## Implementation Details & Code Changes

The original repo assumes a multi-GPU DDP setup. When I tried running the training graph alongside the vLLM engine across 4x 80GB A100s, it instantly threw an Out-Of-Memory (OOM) error. GPU 0 couldn't handle the extra ~1GB NCCL buffer needed to sync gradients across the nodes while vLLM was hogging the VRAM.

To get this to actually train, I dropped the DDP approach and constrained it to a single GPU with a few strict memory tweaks:

1. **8-bit AdamW:** Swapped the default optimizer for `bitsandbytes` 8-bit AdamW to shrink the optimizer states during the backward pass.
2. **vLLM Memory Clamping:** By default, vLLM tries to reserve 90% of the GPU for its KV cache. I hard-clamped `vllm_gpu_memory_utilization=0.3` in the config so PyTorch had enough room to breathe.
3. **Single GPU Isolation:** Forced execution on `CUDA_VISIBLE_DEVICES=0` to completely avoid the NCCL communication overhead.

It takes longer to train on one GPU, but it fits perfectly inside an 80GB envelope without crashing.

## How to Run

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_name tooluse \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir ./results_tooluse_7b \
  --learning_rate 5e-5 \
  --num_train_epochs 2 \
  --num_prompts_per_batch 16