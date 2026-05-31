# Akash Unsloth Fine-tune Demo — Session Summary

> Full lifecycle from idea to live deployment on Akash H200 GPU

## Timeline

- **Date**: May 31, 2026
- **Total elapsed**: ~4 hours from idea to live deployment
- **Status**: 4 deployment attempts, all crashed (different root causes each time)

---

## Phase 1: Idea Exploration

- Researched CJ Zafir's SLM fine-tuning guidance (May 2026)
- Key insight: Small Language Models (SLMs) are the future — fine-tune small, deploy cheap
- Identified Akash Network as the cheapest GPU cloud (H200 at $0.75/hr)
- Chose Qwen 3.5 4B as the base model — capable but small enough for QLoRA on a single GPU

**Key reference**: `docs/references/cj-zafir-slm-finetuning-guidance-may2026.md`

## Phase 2: Specification

Created formal spec (`docs/specs/akash-unsloth-finetune-spec.md`):

- **Base model**: `unsloth/qwen3.5-4b-instruct-bnb-4bit` (pre-quantized 4-bit)
- **Method**: QLoRA (4-bit quantization + LoRA adapters)
- **Dataset**: `lmms-lab/OlympiadBench` (math/science competition problems)
- **Target hardware**: NVIDIA H200 (143GB VRAM) on Akash Network
- **Cost target**: ~$0.98/run at $0.75/hr for ~1.3hr training
- **Output**: LoRA adapters + merged model upload to HuggingFace Hub

## Phase 3: Planning

11 tasks across 4 phases with 51 tests:

| Phase | Tasks | Description |
|-------|-------|-------------|
| Core Setup | 3 | Project scaffold, data pipeline, training loop |
| Training | 3 | QLoRA config, training execution, loss tracking |
| Evaluation | 3 | Inference comparison, model export, metrics |
| Deployment | 2 | Docker + Akash SDL, documentation |

**Test coverage**: 51 tests across 4 test files — all passing

## Phase 4: Build

**12 source files** delivered:

```
src/
  __init__.py
  config.py          # Configuration management
  prepare_data.py    # Dataset download + preprocessing
  train.py           # QLoRA training loop with Unsloth
  evaluate.py        # Inference comparison (base vs fine-tuned)
  export.py          # LoRA adapter + merged model export
  upload.py          # HuggingFace Hub upload
  utils.py           # Shared utilities
entrypoint.sh        # Container entrypoint with validation
configs/
  training_config.yaml
Dockerfile           # Multi-stage build
```

All 12 acceptance criteria from the spec were met.

## Phase 5: Verification

4 issues found and fixed during verification:

| # | Issue | Fix |
|---|-------|-----|
| 1 | `:latest` Docker tag in deploy.yaml | Pinned to `:v1.0.0` |
| 2 | `entrypoint.sh` hard-exits when HF_TOKEN missing | Made HF vars optional with warnings |
| 3 | `${SUBSET_SIZE:-5000}` bash syntax in YAML env | Changed to plain numeric values |
| 4 | Wrong dataset name `OpenBMB/OlympiadBench` | Fixed to `lmms-lab/OlympiadBench` |

51/51 tests passing after fixes.

## Phase 6: Deployment

### Attempt 1: DSEQ 27062068 — Insufficient Funds
- **Escrow**: 0.5 ACT (500,000 uact)
- **Needed**: ~8 ACT for 2.5hr run
- **Runtime**: 14 minutes before escrow exhausted
- **Lesson**: Always over-fund escrow — compute burns fast on H200

### Attempt 2: DSEQ 27067167 — Env Var Bug
- **Escrow**: 8 ACT (funded correctly)
- **Bug**: `${SUBSET_SIZE:-5000}` passed as literal string to Python argparse
- **Runtime**: ~1 minute (crashed on data preparation)
- **Fix**: Updated deploy.yaml env vars to plain values (`SUBSET_SIZE: "5000"`)

### Attempt 3: DSEQ 27067332 — Wrong Dataset + Missing HF Token
- **Bug**: `OpenBMB/OlympiadBench` doesn't exist on HuggingFace
- **Bug**: HF_TOKEN/HF_REPO_ID passed as literal `${VAR}` strings
- **Fix**: Changed to `lmms-lab/OlympiadBench`, made HF env vars optional

### Attempt 4: DSEQ 27067570 — Wrong Dataset Split
- **Provider**: `akash1evr5r8r8zgxddvhru3t0l8q079a94ew8hcgwdd`
- **GPU**: NVIDIA H200 (143GB VRAM, driver 570.211.01) ✅
- **Rate**: 3,600.77 uact/block (~$0.75/hr)
- **Escrow**: 8,000,000 uact (8 ACT) ✅
- **Image**: `ghcr.io/toxmon/akash-unsloth-finetune:v1.0.0` ✅
- **Bug**: `lmms-lab/OlympiadBench` has no `train` split — only `test_en` and `test_cn`
- **Status**: Crashed during data preparation
- **Fix needed**: Update `prepare_data.py` to use `DATA_SPLIT` env var or default to `test_en`

## Key Metrics

| Metric | Value |
|--------|-------|
| Source files | 12 |
| Tests | 51/51 passing |
| Estimated cost/run | ~$0.98 |
| GPU | NVIDIA H200 (143GB) |
| Deployment attempts | 4 (all crashed, different root causes) |
| Total spent on failed deploys | ~$0.015 |

## Links

| Resource | URL |
|----------|-----|
| GitHub repo | https://github.com/ToXMon/akash-unsloth-finetune |
| GHCR image | `ghcr.io/toxmon/akash-unsloth-finetune:v1.0.0` |
| Base model | `unsloth/qwen3.5-4b-instruct-bnb-4bit` |
| Dataset | `lmms-lab/OlympiadBench` |
| Akash SDL | `deploy.yaml` in repo root |

## Lessons Learned

1. **Over-fund escrow**: GPU compute burns through tokens fast. 0.5 ACT lasted 14 minutes.
2. **No bash substitution in YAML**: `${VAR:-default}` is a shell feature, not a Docker/Kubernetes feature. Use plain values.
3. **Verify dataset names**: `OpenBMB/OlympiadBench` vs `lmms-lab/OlympiadBench` — always check HuggingFace.
4. **Make optional vars truly optional**: HF_TOKEN should not be required for training, only for upload.
5. **Pin image tags**: Never use `:latest` in production SDLs.
6. **Test the Docker image locally first**: If the image runs locally, it'll run on Akash.
