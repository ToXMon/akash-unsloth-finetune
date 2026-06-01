# Akash Unsloth Fine-tune Demo

Fine-tune a Qwen 3.5 4B model on Akash H200 GPUs using Unsloth QLoRA for under $3.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Akash H200 GPU Node                        │
│                                                               │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐  │
│  │ prepare_data │──▶│    train.py   │──▶│    upload.py      │  │
│  │              │   │              │   │                   │  │
│  │ HF Dataset   │   │ Unsloth QLoRA│   │ LoRA → HF Hub    │  │
│  │ → Alpaca     │   │ 4-bit Quant  │   │ Merged → HF Hub  │  │
│  │ → JSONL      │   │ LoRA r=16    │   │ Model Card       │  │
│  └─────────────┘   └──────────────┘   └───────────────────┘  │
│         │                 │                    │              │
│         ▼                 ▼                    ▼              │
│  /app/data/         /app/output/         HuggingFace         │
│  train.jsonl        lora_adapters/       Hub Repo             │
│                     merged_model/                             │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    CostTracker                          │  │
│  │  $0.75/hr · Warn 80% · Hard-stop 100% · Budget $3.00  │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
   entrypoint.sh              train_config.yaml
   (orchestration)             (all settings)
```

## Prerequisites

| Requirement | Purpose | Install |
|-------------|---------|--------|
| Akash wallet | Fund deployments | [Akash Docs](https://akash.network/docs/) |
| ~5 ACT (~$5) | Deployment escrow | `akash tx bme mint-act 5000000uakt --from wallet -y` |
| HuggingFace token | Upload model | [hf.co/settings/tokens](https://huggingface.co/settings/tokens) |
| Docker | Build image | [docker.com](https://docs.docker.com/get-docker/) |
| provider-services | Deploy to Akash | `curl -sSfL https://get.akash.network | sh` |

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env — set HF_TOKEN and HF_REPO_ID
vim .env
```

### 2. Build Docker image

```bash
docker build -t akash-unsloth-finetune:latest .
```

### 3. Deploy to Akash

```bash
# Create certificate (first time only)
provider-services tx cert create client --from wallet

# Deploy
provider-services tx deployment create deploy.yaml --from wallet
```

Follow the standard Akash deployment flow: get bids, accept provider, send manifest.

## Configuration

All settings externalized through `configs/train_config.yaml` and environment variables.

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | **Yes** | — | HuggingFace API token |
| `HF_REPO_ID` | **Yes** | — | Target repo (e.g. `user/model-name`) |
| `BASE_MODEL` | No | `unsloth/qwen3.5-4b-instruct-bnb-4bit` | Base model identifier |
| `MAX_SEQ_LENGTH` | No | `2048` | Maximum sequence length |
| `DATA_SOURCE` | No | `OpenBMB/OlympiadBench` | HuggingFace dataset |
| `SUBSET_SIZE` | No | `5000` | Number of training samples |
| `DATASET_DOMAIN` | No | `medical_qa` | Domain label |
| `LORA_RANK` | No | `16` | LoRA attention rank |
| `LORA_ALPHA` | No | `32` | LoRA scaling factor |
| `LORA_DROPOUT` | No | `0.05` | LoRA dropout probability |
| `LEARNING_RATE` | No | `2e-4` | Learning rate |
| `NUM_EPOCHS` | No | `3` | Training epochs |
| `BATCH_SIZE` | No | `4` | Per-device batch size |
| `GRADIENT_ACCUMULATION_STEPS` | No | `4` | Gradient accumulation |
| `WARMUP_STEPS` | No | `50` | LR warmup steps |
| `SEED` | No | `42` | Random seed |
| `MAX_STEPS` | No | (all) | Override max training steps |
| `PUSH_MERGED` | No | `true` | Upload merged model |
| `PUSH_ADAPTERS` | No | `true` | Upload LoRA adapters |
| `HOURLY_RATE` | No | `0.75` | Cost rate ($/hr) |
| `MAX_BUDGET` | No | `3.00` | Max budget ($) |
| `CONFIG_PATH` | No | `configs/train_config.yaml` | Config file path |
| `OUTPUT_PATH` | No | `/app/data/train.jsonl` | Training data output |

### LoRA Target Modules

All linear projection layers in the transformer:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

## Cost Estimation

| Component | Duration | Cost (@ $0.75/hr) |
|-----------|----------|-------------------|
| Data download + prep | ~5 min | ~$0.06 |
| Training (3 epochs, 5K samples) | ~2 hr | ~$1.50 |
| Model merge + upload | ~15 min | ~$0.19 |
| **Total estimated** | **~2.5 hr** | **~$1.88** |

Budget safety: warns at $2.40 (80%), hard-stops at $3.00 (100%).

## Project Structure

```
projects/akash-unsloth-finetune/
├── Dockerfile              # Multi-stage CUDA + Python build
├── deploy.yaml             # Akash SDL (H200, 8 CPU, 32Gi RAM)
├── entrypoint.sh           # Main orchestration script
├── src/
│   ├── __init__.py
│   ├── prepare_data.py     # Dataset download → Alpaca JSONL
│   ├── train.py            # Unsloth QLoRA fine-tuning
│   ├── upload.py           # HF Hub upload with retry
│   └── utils.py            # GPU check, logging, cost tracking
├── configs/
│   └── train_config.yaml   # All training configuration
├── tests/
│   ├── __init__.py
│   ├── test_utils.py           # Unit tests (19 tests)
│   ├── test_prepare_data.py    # Data pipeline tests
│   └── test_integration.py     # End-to-end verification
├── .env.example            # Environment variable template
├── .dockerignore
└── README.md
```

## Local Development (No GPU)

```bash
# Install dependencies
pip install datasets pyyaml pytest numpy

# Run tests (no GPU needed)
python -m pytest tests/ -v

# Syntax check all source files
python -c "import ast; [ast.parse(open(f).read()) for f in ['src/utils.py', 'src/prepare_data.py', 'src/train.py', 'src/upload.py']]" && echo 'All syntax OK'

# Verify config
python -c "import yaml; yaml.safe_load(open('configs/train_config.yaml'))" && echo 'Config OK'
```

## Troubleshooting

### No GPU / nvidia-smi not found

**Cause**: Deployment landed on a provider without GPU or driver issue.

```bash
# Check GPU inside container
nvidia-smi

# Verify deploy.yaml specifies GPU
grep -A5 'gpu:' deploy.yaml
```

**Fix**: Close deployment and recreate — ensure SDL specifies `nvidia` vendor with `h200` model.

### Out of Memory (OOM)

**Cause**: Model + batch size exceeds 80GB H200 memory.

**Fix**: Reduce in `train_config.yaml`:
- `batch_size: 2` (from 4)
- `gradient_accumulation_steps: 8` (from 4) to maintain effective batch size
- `max_seq_length: 1024` (from 2048)

### Upload to HuggingFace fails

**Cause**: Invalid token, repo name, or network issue.

```bash
# Verify token
echo $HF_TOKEN

# Test locally
huggingface-cli login --token $HF_TOKEN
```

The system retries 3 times with exponential backoff (5s, 15s, 45s).
If upload fails completely, it falls back to HTTP on port 8080 showing local artifact paths.

### Escrow exhausted / Budget exceeded

**Cause**: Training ran longer than expected.

**Fix**:
- Reduce `max_budget` in config to stop earlier
- Reduce `num_epochs` to 1-2
- Set `MAX_STEPS=500` for a quick training run

Fund more: `akash tx bme mint-act 10000000uakt --from wallet`

### Training data format errors

**Cause**: Dataset fields don't match expected format.

The converter handles: `question/answer`, `prompt/response`, `instruction/output`, `problem/solution`, `final_answer`.

Check logs for filter rate. If >50% filtered, inspect the dataset structure:
```bash
python -c "from datasets import load_dataset; ds = load_dataset('OpenBMB/OlympiadBench', split='train'); print(ds[0])"
```

## Cleanup

```bash
# Close Akash deployment
provider-services tx deployment close --dseq $DSEQ --from wallet

# Remove local Docker image
docker rmi akash-unsloth-finetune:latest

# Delete HuggingFace repo (if needed)
# Go to https://huggingface.co/settings/repos → delete
```

## References

- [Unsloth Documentation](https://github.com/unslothai/unsloth) — 2x faster fine-tuning
- [Akash Network Docs](https://akash.network/docs/) — Decentralized cloud marketplace
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) — Efficient finetuning of quantized LLMs
- [HuggingFace TRL](https://huggingface.co/docs/trl) — Transformer Reinforcement Learning
- [OlympiadBench Dataset](https://huggingface.co/datasets/OpenBMB/OlympiadBench) — Competition math problems
- [provider-services CLI](https://github.com/akash-network/provider-services) — Akash deployment tool

## Deployment Results

> **Status**: 4 deployment attempts on Akash Mainnet — all crashed at different stages. Full autopsy in `deployment-artifacts/`.

### Deployment History

| DSEQ | Escrow | Runtime | Root Cause | Fix |
|------|--------|---------|------------|-----|
| 27062068 | 0.5 ACT | 14 min | Insufficient escrow funds | Funded 8 ACT |
| 27067167 | 8 ACT | ~1 min | `${SUBSET_SIZE:-5000}` literal string | Plain env values |
| 27067332 | 8 ACT | ~2 min | Wrong dataset name + HF token as literal | Fixed dataset, made HF optional |
| 27067570 | 8 ACT | ~2 min | Dataset has no `train` split | **Pending** — need `DATA_SPLIT` env var |

### Cost Analysis

| Metric | Value |
|--------|-------|
| GPU hourly rate | ~$0.75/hr (H200) |
| Cost per block | 3,600.77 uact/block |
| Estimated training cost | ~$0.98 (1.3hr run) |
| Total spent on failed deploys | ~$0.015 |
| Escrow per attempt (8 ACT) | ~$7.84 |

### Lessons Learned

1. **Over-fund escrow**: 0.5 ACT lasted 14 minutes on H200. Use 8+ ACT.
2. **No bash substitution in YAML**: `${VAR:-default}` is shell-only. Docker/K8s env sections pass it as literal string.
3. **Verify dataset splits**: `lmms-lab/OlympiadBench` has `test_en` and `test_cn` — no `train` split. Always check `dataset.info.splits`.
4. **Make optional vars truly optional**: HF_TOKEN should not block training — only upload.
5. **Pin image tags**: `:latest` causes non-reproducible deployments. Use `:v1.0.0`.
6. **Each failure teaches something different**: 4 deploys, 4 distinct root causes — env vars, funding, dataset name, dataset splits.

### Artifacts

| File | Description |
|------|-------------|
| `deployment-artifacts/SESSION_SUMMARY.md` | Full lifecycle from idea to deployment |
| `deployment-artifacts/DEPLOYMENT_HISTORY.md` | Detailed per-deployment autopsy |
| `deployment-artifacts/logs-dseq-27067570.txt` | Captured lease logs from final attempt |
| `deploy.yaml` | Akash SDL with H200 GPU config |
| `.github/workflows/docker-build.yml` | CI/CD for GHCR image builds |

### Next Steps

- [ ] Add `DATA_SPLIT` env var to `prepare_data.py` (default: `test_en`)
- [ ] Rebuild Docker image as `v1.1.0`
- [ ] Deploy DSEQ 27067571+ with `DATA_SPLIT=test_en`
- [ ] Verify training starts successfully
- [ ] Monitor loss convergence on H200

## License

MIT
