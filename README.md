<div align="center">

  <h3>[ICML2026 Oral] OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in <i>Every</i> Iteration</h3>

  <p>
    <a href="https://gszfwsb.github.io/">Shaobo Wang</a><sup>1,2&#42;</sup>&nbsp;&middot;&nbsp;
    <a href="https://yancyou.github.io/">Xuan Ouyang</a><sup>1,3&#42;</sup>&nbsp;&middot;&nbsp;
    <a href="https://tianyi0216.github.io/">Tianyi Xu</a><sup>1,3&#42;</sup>&nbsp;&middot;&nbsp;
    <a href="https://mirnegg.github.io/">Yuzheng Hu</a><sup>4</sup>&nbsp;&middot;&nbsp;
    Jialin Liu<sup>1</sup>&nbsp;&middot;&nbsp;
    Guo Chen<sup>1</sup>
    <br>
    <a href="https://ai.zhangtianyu.org/">Tianyu Zhang</a><sup>5</sup>&nbsp;&middot;&nbsp;
    <a href="https://scholar.google.com/citations?user=eyh-5tkAAAAJ&hl=zh-CN">Junhao Zheng</a><sup>2</sup>&nbsp;&middot;&nbsp;
    <a href="https://scholar.google.com/citations?user=bwv4Aj4AAAAJ&hl=zh-CN">Kexin Yang</a><sup>2</sup>&nbsp;&middot;&nbsp;
    <a href="https://scholar.google.com/citations?user=3YzSsyIAAAAJ&hl=en">Xingzhang Ren</a><sup>2&#9993;</sup>&nbsp;&middot;&nbsp;
    <a href="https://liudayiheng.github.io/">Dayiheng Liu</a><sup>2&#9993;</sup>&nbsp;&middot;&nbsp;
    <a href="http://www.zhanglinfeng.tech/">Linfeng Zhang</a><sup>1&#9993;</sup>
  </p>

  <p>
    <sup>1</sup> EPIC Lab, Shanghai Jiao Tong University &nbsp;
    <sup>2</sup> Qwen Team, Alibaba Group &nbsp;
    <sup>3</sup> UW–Madison
    <br>
    <sup>4</sup> UIUC &nbsp;
    <sup>5</sup> Mila–Quebec AI Institute
  </p>

  <p>
    <sup>&#42;</sup> Equal contribution &nbsp;&nbsp; <sup>&#9993;</sup> Corresponding authors
  </p>

  <p>
    <a href="https://arxiv.org/abs/2602.05400">
      <img src="https://img.shields.io/badge/arXiv-2602.05400-B31B1B?style=flat-square&logo=arxiv" alt="arXiv">
    </a>
    <a href="https://huggingface.co/papers/2602.05400">
      <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Daily%20Paper-Day%201-FF6B6B?style=flat-square" alt="Daily Paper">
    </a>
    <a href="https://github.com/gszfwsb/OPUS">
      <img src="https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github" alt="Code">
    </a>
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch">
  </p>

</div>

---

## Overview

As high-quality public text approaches exhaustion, a phenomenon known as the Data Wall, pre-training is shifting from more tokens to better tokens. However, existing methods either rely on heuristic static filters that ignore training dynamics, or use dynamic yet optimizer-agnostic criteria based on raw gradients. We propose \textbf{OPUS} (Optimizer-induced Projected Utility Selection), a dynamic framework that defines utility in the optimizer-induced update space. OPUS scores candidates by projecting their effective updates, shaped by modern optimizers, onto a target direction derived from a stable, in-distribution proxy. To ensure scalability, we employ Ghost technique with CountSketch for computational efficiency, and Boltzmann sampling for data diversity, incurring only 4.7\% additional compute overhead. OPUS achieves remarkable results across diverse corpora, quality tiers, optimizers, and model scales. It also outperforms previous data selection methods across different stages of training, including from-scratch pre-training and also mid-training. Beyond online selection, the OPUS utility score also demonstrates potential as a static filter for flagging and removing toxic documents from contaminated training corpora prior to training.

---

## Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/gszfwsb/OPUS.git
cd OPUS

# 2. Create environment
conda create -n opus python=3.10
conda activate opus

# 3. Install dependencies
pip install -r requirements.txt

# (Optional) Install OpenCompass for downstream evaluation
pip install opencompass
```

> **Hardware Requirements:** Training from scratch is tested on 8×A100/H100 (80 GB). CPT and smaller models can run on fewer GPUs. Adjust batch size and gradient accumulation accordingly.

### Train with OPUS (Minimal Example)

```bash
# Edit DATA_ROOT in run.sh, or override inline
bash run.sh
```

### Evaluate a Checkpoint

```bash
bash eval/run_lighteval.sh logs/<experiment>/state_step010000.pt gpt2-xl mmlu
```

---

## Data Preparation

All data scripts live under `data/`. The typical pipeline is:

1. **Download** raw FineWeb parquet files.
2. **Score** (optional) each document with BETR-score for proxy construction.
3. **Tokenize** parquet files into GPT-2 `.bin` shards for training.

### Download

```bash
# FineWeb (200B tokens, plain parquet)
bash data/run_download_fineweb.sh

# FineWeb-Edu (score-filtered higher quality data)
python data/download_fineweb_edu.py --output_dir finewebedu --target_tokens_b 30.0 --continuous
```

All downloads support manifest-based resume.

### Tokenize

Convert downloaded parquet files to `.bin` shards (GPT-2 tokenizer, 100M tokens per shard):

```bash
PARQUET_DIR=fineweb_parquet \
TRAIN_OUTPUT_DIR=bins/fineweb_train \
VAL_OUTPUT_DIR=bins/fineweb_val \
  bash data/run_tokenize_gpt2.sh
```

### Build BENCH-Proxy for OPUS

OPUS requires a small proxy dataset, selected by scoring documents via embedding similarity to benchmark tasks. The full pipeline:

```bash
INPUT_ROOT=/path/to/fineweb_parquet \
OUTPUT_SCORED=/path/to/scored_parquet \
PROXY_OUTPUT=/path/to/proxy_bins \
  bash data/run_build_proxy.sh
```

This runs three steps: (1) embed benchmarks, (2) BETR-score all documents, (3) select top-scored documents into `.bin` shards.

---

## Project Structure

```
OPUS/
├── layers/                 # Custom layers (GCLinear, LoRA) for ghost gradients
│   ├── linear.py
│   └── lora_layers.py
└── train/                  # Core OPUS algorithm
    ├── data_selection.py   # Scoring + selection logic
    └── random_projection.py

train.py                    # Main training script (pre-training & CPT)
model.py                    # GPT model: RMSNorm, RoPE, QK-norm, GCLinear
run.sh                      # Launch script for OPUS training

data/                       # Data pipeline scripts
├── download_fineweb.py
├── download_fineweb_edu.py
├── gpt2tokenize.py
├── betr_score_parquet.py
├── build_betr_proxy.py
├── embed_benchmarks.py
├── run_download_fineweb.sh
├── run_tokenize_gpt2.sh
├── run_build_proxy.sh
└── run_betr_score_fineweb_parquet.sh

cpt/
├── run_qwen3_cpt.sh        # Qwen-3 continual pre-training with OPUS
└── verify_weight_init.py

eval/
├── lighteval.py            # Offline evaluation runner
└── run_lighteval.sh
```

---

## Training

### OPUS (Online Data Selection)

`run.sh` trains with OPUS data selection. Key environment variables:

```bash
# Minimal launch (edit DATA_ROOT in the script, or override inline)
bash run.sh

# Override OPUS hyperparameters
OPUS_SELECTION_RATIO=0.25 \
OPUS_BUFFER_MULTIPLIER=64 \
OPUS_PRECONDITIONER=muon \
OPUS_TEMPERATURE=1.5 \
  bash run.sh
```

### Direct `train.py` Usage

All configuration goes through command-line arguments to `train.py`:

```bash
torchrun --nproc_per_node=8 train.py \
    --model_type gpt2-xl \
    --optimizer_type adamw_unified \
    --total_tokens_b 30.0 \
    --eval_every_tokens_b 1.0 \
    --train_files "/path/to/train_*.bin" \
    --val_files "/path/to/val_*.bin" \
    --use_opus \
    --selection_strategy opus \
    --opus_preconditioner auto \
    --opus_selection_ratio 0.5 \
    --opus_buffer_size_multiplier 32 \
    --opus_temperature 0.9 \
    --opus_score_len 512 \
    --opus_proxy_dir "/path/to/proxy_*.bin" \
    --opus_proxy_tokens 30000000 \
    --use_random_projection \
    --projection_dim 8192 \
    --experiment_name "my_experiment"
```

---

## Key Arguments

### Model and Optimizer

| Argument | Choices | Default | Description |
|---|---|---|---|
| `--model_type` | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b` | `gpt2-xl` | Model architecture |
| `--optimizer_type` | `muon_hybrid`, `adamw_unified` | `adamw_unified` | `muon_hybrid` uses Muon for matrix params + AdamW for the rest; `adamw_unified` uses AdamW for all params |
| `--adam_lr` / `--muon_lr` | float | model-specific | Override default learning rates |

### Training Schedule

| Argument | Default | Description |
|---|---|---|
| `--total_tokens_b` | `30.0` | Total training tokens (billions) |
| `--eval_every_tokens_b` | `1.0` | Validation interval (billions) |
| `--train_seq_len` | `6144` | Training sequence length |
| `--val_seq_len` | `8192` | Validation sequence length |
| `--grad_accum_steps` | `1` | Gradient accumulation steps |
| `--lr_schedule` | `cosine` | LR schedule (`cosine`, `linear`, `legacy`) |
| `--warmup_frac` | `0.0` | Warmup fraction of training |

### OPUS Data Selection

| Argument | Default | Description |
|---|---|---|
| `--use_opus` | off | Enable OPUS data selection |
| `--selection_strategy` | `opus` | `opus` (gradient-based), `ppl` (perplexity), `random` (shuffled baseline) |
| `--opus_preconditioner` | `auto` | `auto`, `muon`, `adamw`, `sgd` |
| `--opus_selection_ratio` | `0.5` | Fraction of candidate buffer to select |
| `--opus_buffer_size_multiplier` | `32` | Candidate pool = multiplier × train_seq_len |
| `--opus_temperature` | `0.9` | Temperature for stochastic selection |
| `--opus_score_len` | `512` | Token window length for scoring (shorter = faster) |
| `--opus_proxy_batch` | `16` | Proxy batch size for gradient computation |
| `--opus_proxy_dir` | - | Glob pattern for proxy `.bin` files |
| `--opus_proxy_tokens` | `30000000` | Total proxy tokens to load |

### Random Projection

| Argument | Default | Description |
|---|---|---|
| `--use_random_projection` | off | Project gradients to lower dimension |
| `--projection_dim` | `8192` | Target dimension |
| `--projection_seed` | `42` | Random seed for projection matrix |

### Continual Pre-Training (CPT)

| Argument | Description |
|---|---|
| `--init_model` | Path or HuggingFace name for base model |
| `--use_loss_mask` | Mask metadata tokens during CPT |
| `--domain_root_dir` | Comma-separated directories for domain-organized data |

### Checkpointing

| Argument | Description |
|---|---|
| `--experiment_name` | Experiment name (determines log directory) |
| `--resume_from_checkpoint` | Path to `.pt` checkpoint to resume from |

---

## How OPUS Works

At each training step, OPUS selects a high-quality, diverse subset from a candidate buffer:

1. **Buffer**: Rank 0 loads a candidate buffer (e.g. 32× the training batch), broadcasts to all ranks.
2. **Gradient Scoring**: Compute per-candidate preconditioned gradient inner products with the proxy (validation-like) data. The preconditioner adapts to the optimizer: Muon uses a dense low-rank approximation; AdamW uses the diagonal inverse RMSprop term.
3. **Diversity-aware Selection**: Stochastic selection via Boltzmann distribution picks candidates with high proxy alignment while penalizing redundancy via the candidate-candidate similarity matrix.
4. **Random Projection** (optional): Project gradients to a lower-dimensional space to reduce memory and compute cost while preserving selection quality.

---

## Results

Please refer to our [paper](https://arxiv.org/abs/2602.05400) for detailed results.

### Key Findings

- **Better sample efficiency**: OPUS achieves lower validation loss and better downstream task performance compared to static filtering and random baselines.
- **Scalable**: Random projection reduces per-step overhead to negligible levels even for billion-parameter models.
- **General**: Effective across GPT-2, Llama, and Qwen architectures.

---


## Acknowledgments

This project builds upon the excellent [nanoGPT](https://github.com/KellerJordan/modded-nanogpt) training framework by [Keller Jordan](https://github.com/KellerJordan). We thank the open-source community for the Muon optimizer, FlexAttention, and the nanoGPT ecosystem that made this work possible.



## Citation

```bibtex
@misc{wang2026opus,
      title={OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration}, 
      author={Shaobo Wang and Xuan Ouyang and Tianyi Xu and Yuzheng Hu and Jialin Liu and Guo Chen and Tianyu Zhang and Junhao Zheng and Kexin Yang and Xingzhang Ren and Dayiheng Liu and Linfeng Zhang},
      year={2026},
      eprint={2602.05400},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.05400}, 
}
```
