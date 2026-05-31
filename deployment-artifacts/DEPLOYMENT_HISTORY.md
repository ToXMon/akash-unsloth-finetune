# Akash Deployment History

Full deployment history for the Akash Unsloth Fine-tune demo project.

---

## DSEQ 27062068 — ❌ Insufficient Funds

- **Created**: Block 27062070
- **Provider**: `akash1evr5r8r8zgxddvhru3t0l8q079a94ew8hcgwdd`
- **GPU**: NVIDIA H200 (143GB)
- **Rate**: 3,600.77 uact/block
- **Escrow**: 500,000 uact (0.5 ACT)
- **Runtime**: ~14 minutes (143 blocks)
- **State**: closed/insufficient_funds
- **Root Cause**: Escrow funded with only 0.5 ACT. At ~3,600 uact/block, this lasted ~143 blocks (~14 min). A full training run (~2.5hr) requires ~7-8 ACT.
- **Fix**: Funded next deployment with 8,000,000 uact (8 ACT)
- **Lesson**: Always calculate escrow as `rate * blocks_needed * 1.5` for safety margin

**Timeline**:
```
Block 27062070 — Deployment created, escrow 500,000 uact
Block 27062072 — Bid received from provider
Block 27062073 — Lease created, manifest sent
Block 27062075 — Container started, training began
Block 27062100 — Data download progressing
Block 27062213 — Escrow exhausted, deployment closed
Runtime: ~143 blocks ≈ 14 minutes
```

---

## DSEQ 27067167 — ❌ Env Var Bug

- **Created**: Block 27067168
- **Provider**: `akash1evr5r8r8zgxddvhru3t0l8q079a94ew8hcgwdd`
- **GPU**: NVIDIA H200
- **Rate**: 3,600.77 uact/block
- **Escrow**: 8,000,000 uact (8 ACT) ✅
- **Runtime**: ~1 minute
- **State**: closed/owner
- **Root Cause**: `deploy.yaml` passed `${SUBSET_SIZE:-5000}` as a literal string to the container. Python's argparse received `${SUBSET_SIZE:-5000}` instead of `5000`, causing `prepare_data.py` to crash with `invalid int value`.
- **Fix**: Changed deploy.yaml env vars to plain values: `SUBSET_SIZE: "5000"`
- **Lesson**: Bash parameter expansion (`${VAR:-default}`) is a shell feature. Docker/Kubernetes YAML env sections don't interpret it. Use plain values or entrypoint.sh defaults.

**Error log**:
```
[entrypoint] Running data preparation...
usage: prepare_data.py [-h] [--dataset DATASET] [--subset_size SUBSET_SIZE] ...
prepare_data.py: error: argument --subset_size: invalid int value: '${SUBSET_SIZE:-5000}'
```

---

## DSEQ 27067332 — ❌ Wrong Dataset + Missing HF Token

- **Created**: Block 27067333 (approx)
- **Provider**: `akash1evr5r8r8zgxddvhru3t0l8q079a94ew8hcgwdd`
- **GPU**: NVIDIA H200
- **Rate**: 3,600.77 uact/block
- **Escrow**: 8,000,000 uact (8 ACT) ✅
- **Runtime**: ~2 minutes
- **State**: closed/owner
- **Root Cause 1**: Dataset `OpenBMB/OlympiadBench` doesn't exist on HuggingFace. Correct name is `lmms-lab/OlympiadBench`.
- **Root Cause 2**: `HF_TOKEN` and `HF_REPO_ID` were still being passed as literal `${VAR}` strings from deploy.yaml, causing upload.py to fail.
- **Fix 1**: Changed dataset to `lmms-lab/OlympiadBench` in deploy.yaml and training_config.yaml
- **Fix 2**: Made HF_TOKEN/HF_REPO_ID truly optional — entrypoint.sh warns but continues if missing
- **Lesson**: Always verify HuggingFace dataset names by checking the URL. Don't assume naming conventions.

**Error log**:
```
[entrypoint] Running data preparation...
Downloading dataset: OpenBMB/OlympiadBench
Traceback (most recent call last):
  File ".../datasets/load.py", line 1234, in load_dataset
    ...
DatasetNotFoundError: Dataset 'OpenBMB/OlympiadBench' doesn't exist on HuggingFace
```

---

## DSEQ 27067570 — ❌ Wrong Dataset Split

- **Created**: Block 27067571 (approx)
- **Provider**: `akash1evr5r8r8zgxddvhru3t0l8q079a94ew8hcgwdd`
- **GPU**: NVIDIA H200 (143GB VRAM, driver 570.211.01) ✅
- **Rate**: 3,600.77 uact/block (~$0.75/hr)
- **Escrow**: 8,000,000 uact (8 ACT) ✅
- **Image**: `ghcr.io/toxmon/akash-unsloth-finetune:v1.0.0` ✅
- **Runtime**: ~2 minutes
- **State**: crashed
- **Root Cause**: `lmms-lab/OlympiadBench` has no `train` split. Available splits are `test_en` and `test_cn`. The `prepare_data.py` script defaults to `split="train"` which doesn't exist for this dataset.
- **Fix needed**: Update `prepare_data.py` to use `split="test_en"` for this dataset, or add `DATA_SPLIT` env var support
- **Wallet**: `akash1jw93z5t6veshx3w4hs2mkl8004qh57cm855jp0`

**What was fixed from previous attempts**:
1. ✅ Escrow funded with 8 ACT (was 0.5 ACT)
2. ✅ Env vars use plain values (no bash substitution)
3. ✅ Dataset name is correct: `lmms-lab/OlympiadBench`
4. ✅ HF_TOKEN/HF_REPO_ID are optional (training proceeds without upload)
5. ✅ Image tag pinned to `:v1.0.0` (no `:latest`)

**What still needs fixing**:
- Dataset split: code assumes `train` split exists, but OlympiadBench only has `test_en` and `test_cn`
- Need to either change dataset or update code to handle arbitrary splits

**Error log (captured from lease-logs)**:
```
[entrypoint] Phase 1/3: Data Preparation
Generating test_en split: 100%|██████████| 2126/2126
Generating test_cn split: 100%|██████████| 6351/6351
Download failed (attempt 1): Unknown split "train". Should be one of ['test_en', 'test_cn'].
Download failed (attempt 2): Unknown split "train". Should be one of ['test_en', 'test_cn'].
Download failed (attempt 3): Unknown split "train". Should be one of ['test_en', 'test_cn'].
RuntimeError: Failed to download dataset 'lmms-lab/OlympiadBench' after 3 attempts
ERROR: Script failed at line 57 with exit code 1
```

---

## Cost Summary

| DSEQ | Escrow | Runtime | Cost | Outcome |
|------|--------|---------|------|--------|
| 27062068 | 0.5 ACT | 14 min | ~$0.01 | ❌ Insufficient funds |
| 27067167 | 8 ACT | ~1 min | ~$0.001 | ❌ Env var bug |
| 27067332 | 8 ACT | ~2 min | ~$0.002 | ❌ Wrong dataset |
| 27067570 | 8 ACT | ~2 min | ~$0.002 | ❌ Wrong dataset split |

**Total spent on failed deployments**: ~$0.015
**Total deployment attempts**: 4, all failed (different root causes each time)

### Next Steps for DSEQ 27067571+
- Fix `prepare_data.py` to accept `DATA_SPLIT` env var (default: `test_en`)
- Or switch to a dataset that has a `train` split
- Rebuild Docker image, push to GHCR, redeploy
