#!/usr/bin/env python
"""
alphaprune.py – Exact AlphaPruning replication (macOS / CPU / MPS‑friendly)

Example:
    python alphaprune.py --model distilgpt2 --sparsity 0.70 --backend wanda
"""

import argparse, math, json, os, sys, time
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ────────────────── Heavy‑tail α metric (Hill estimator, Eq. 3) ─────────────
def hill_alpha(eigs, k=None, eps=1e-12):
    """
    Robust Hill estimator that ignores non‑positive eigen‑values.

    eigs : 1‑D NumPy array of eigen‑values (non‑negative, in theory)
    k    : tail length. If None, use Fix‑finger rule.
    eps  : small constant added before log to avoid log(0).
    """
    # ❶ keep only strictly positive values
    eigs = eigs[eigs > eps]
    if eigs.size == 0:            # all zeros → return sentinel large α
        return float("inf")

    eigs = np.sort(eigs)
    n = len(eigs)

    # ❷ choose tail length k  (Fix‑finger peak rule on *finite* logs)
    if k is None:
        logs = np.log(eigs + eps)
        hist, _ = np.histogram(logs, bins=60)
        k = max(10, n - np.argmax(hist) - 1)
        k = min(k, n - 1)         # guard against tiny n

    tail = eigs[-k:]
    return 1.0 + k / np.sum(np.log(tail + eps) - np.log(tail[0] + eps))

# ────────────────── Map α → per‑layer sparsity (Eq. 4) ──────────────────────
def allocate_sparsity(metrics, n_params, S_global=0.70,
                      s1=0.15, s2=0.90):
    mmin, mmax = metrics.min(), metrics.max()
    raw = (metrics - mmin) / (mmax - mmin) * (s2 - s1) + s1
    eta = (S_global * n_params.sum()) / (raw * n_params).sum()
    return np.clip(eta * raw, 0.0, 0.98)

# ────────────────── Magnitude pruning fallback ──────────────────────────────
def magnitude_prune_tensor(t, keep_frac):
    k = int(t.numel() * keep_frac)
    if k == 0:
        t.zero_()
        return
    thresh = torch.topk(t.abs().flatten(), k, largest=True).values[-1]
    mask = t.abs() >= thresh
    t.data.mul_(mask)

# ────────────────── Main driver ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model name or local path")
    ap.add_argument("--sparsity", type=float, default=0.70,
                    help="Global sparsity target (0‑1)")
    ap.add_argument("--backend", choices=["magnitude", "wanda", "sparsegpt"],
                    default="magnitude")
    ap.add_argument("--skip_dim", type=int, default=4096,
                    help="Skip matrices where max(shape) > skip_dim "
                         "(embeddings / LM head)")
    args = ap.parse_args()

    # ─── Load model (CPU / MPS fits on all Apple‑silicon) ───────────────────
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,   # keep FP32 for eigen‑analysis stability
        device_map={"": device}
    )
    model.eval()

    # ─── Spectral scan ──────────────────────────────────────────────────────
    alphas, names, n_params = [], [], []
    with torch.no_grad():
        for n, p in tqdm(model.named_parameters(), desc="spectral scan"):
            if p.ndim != 2:
                continue                      # skip biases / LayerNorm
            if max(p.shape) > args.skip_dim:  # skip giant matrices
                continue

            W = p.detach().float().cpu()
            G = W.T @ W if W.shape[0] > W.shape[1] else W @ W.T
            eigs = torch.linalg.eigvalsh(G).detach().numpy()
            alphas.append(hill_alpha(eigs))
            names.append(n)
            n_params.append(p.numel())

    alphas     = np.array(alphas)
    n_params   = np.array(n_params)
    layer_S    = allocate_sparsity(alphas, n_params, args.sparsity)

    # ─── Pruning pass ───────────────────────────────────────────────────────
    print(f"\nPruning with backend: {args.backend}")
    if args.backend == "wanda":
        try:
            from wanda import prune_model as wanda_prune_model
        except ImportError:
            print("✖ Wanda not found; falling back to magnitude pruning.",
                  file=sys.stderr)
            args.backend = "magnitude"
    elif args.backend == "sparsegpt":
        try:
            from sparsegpt import prune_model as sparsegpt_prune_model
        except ImportError:
            print("✖ SparseGPT not found; falling back to magnitude pruning.",
                  file=sys.stderr)
            args.backend = "magnitude"

    if args.backend == "wanda":
        wanda_prune_model(model, layerwise_sparsities=layer_S.tolist(),
                          param_names=names)
    elif args.backend == "sparsegpt":
        sparsegpt_prune_model(model, layerwise_sparsities=layer_S.tolist(),
                              param_names=names)
    else:  # magnitude
        for (n, p), S in zip(model.named_parameters(), layer_S):
            if p.ndim != 2 or max(p.shape) > args.skip_dim:
                continue
            magnitude_prune_tensor(p, 1.0 - S)

    # ─── Save results ───────────────────────────────────────────────────────
    tag   = os.path.basename(args.model)
    odir  = f"alphaprune_{tag}_{int(args.sparsity*100)}sparse"
    os.makedirs(odir, exist_ok=True)
    model.save_pretrained(odir)
    info = {
        "layers": names,
        "alpha":  alphas.tolist(),
        "sparsity_per_layer": layer_S.tolist(),
        "global_sparsity": args.sparsity,
        "backend": args.backend,
        "skip_dim": args.skip_dim,
        "timestamp": time.asctime(),
    }
    with open(os.path.join(odir, "pruning_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n✔  Saved pruned model →  {odir}\n")

if __name__ == "__main__":
    main()
