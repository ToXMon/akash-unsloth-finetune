# Akash Deployment Attempt Report — v1.2.0
Date: 2026-06-02
Image: ghcr.io/toxmon/akash-unsloth-finetune:v1.2.0
Dataset: lmms-lab/OlympiadBench (test_en)
RPC: https://akash-rpc.polkachu.com:443 (primary rpc.akashnet.net was timing out)

## Deployment Attempts (ALL FAILED — Zero GPU Bids)

| Attempt | DSEQ | GPU Models | CPU | RAM | Storage | Pricing (uact) | Deposit | Bids | Duration |
|---------|------|-----------|-----|-----|---------|----------------|---------|------|----------|
| 1 | 27096764 | H200 | 8 | 32Gi | 100Gi | 100000 | 8 ACT | 0 | 10+ min |
| 2 | 27096795 | A100/H100/H200 | 8 | 32Gi | 100Gi | 150000 | 8 ACT | 0 | 10+ min |
| 3 | 27097085 | A100/A40/L40/RTX4090 | 4 | 24Gi | 50Gi | 150000 | 5 ACT | 0 | 10+ min |
| 4 | 27097173 | A100/A40/L40/L40S/RTX4090/RTX3090/RTX4080/V100/T4 | 4 | 24Gi | 50Gi | 200000 | 5 ACT | 0 | 10+ min |

## Root Cause
- GPU providers on Akash mainnet are extremely scarce
- Zero bids across all configurations including consumer GPUs (RTX 3090/4080)
- Primary RPC (rpc.akashnet.net) was intermittently timing out
- Used alternative RPC: akash-rpc.polkachu.com (worked reliably)

## Wallet Status
- Remaining: ~11.88 ACT + ~4.83 AKT (after escrow returns)
- Gas spent on failed deployments: ~0.02 AKT total

## Recommendations
1. Wait for Akash GPU supply to increase
2. Try during off-peak hours (providers may have capacity)
3. Consider using Akash Console with credit card (may route to different providers)
4. Add more AKT to mint additional ACT for higher pricing
5. Consider alternative GPU cloud providers (RunPod, Vast.ai, Lambda Labs)
