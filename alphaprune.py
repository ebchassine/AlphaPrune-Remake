# alphaprune.py  (run with   python alphaprune.py --model meta-llama/Llama-2-7b-hf --sparsity 0.70)

import argparse, math, torch, numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.linalg import svdvals

# ----------------- α‑metric (Hill estimator, Eq. 3) -----------------
def hill_alpha(eigs, k=None):                            # :contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13}
    eigs = np.sort(eigs)
    n = len(eigs)
    if k is None:                                        # Fix‑finger peak rule
        hist, _ = np.histogram(np.log(eigs+1e-12), bins=60)
        k = max(10, n - np.argmax(hist) - 1)
    tail = eigs[-k:]
    return 1 + k / np.sum(np.log(tail) - math.log(tail[0]))

# ----------------- sparsity allocation (Eq. 4) ---------------------
def allocate_sparsity(metric, n_params, s_global=0.7, s1=0.15, s2=0.90):  # :contentReference[oaicite:14]{index=14}:contentReference[oaicite:15]{index=15}
    mmin, mmax = metric.min(), metric.max()
    raw = (metric - mmin) / (mmax - mmin) * (s2 - s1) + s1
    η = (s_global * n_params.sum()) / (raw * n_params).sum()
    return np.clip(η * raw, 0.0, 0.98)      # per‑block sparsity

# ----------------- magnitude pruning (baseline) --------------------
def prune_tensor(tensor, keep_frac):
    k = int(tensor.numel() * keep_frac)
    th = torch.topk(tensor.abs().flatten(), k, largest=True).values[-1]
    mask = tensor.abs() >= th
    tensor.data.mul_(mask)

# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--sparsity', type=float, default=0.70)
    ap.add_argument('--backend', choices=['magnitude','wanda','sparsegpt'], default='magnitude')
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map='auto'
    );  model.eval()

    # --- step 1 : metric for every transformer block ----------------
    block_metrics, n_params = [], []

    """
     for name, p in tqdm(model.named_parameters(), desc='spectral scan'):
        if p.ndim != 2:             # keep only full weight matrices
            continue
        X = (p @ p.T).float().cpu()
        block_metrics.append(hill_alpha(torch.linalg.eigvalsh(X).numpy()))
        n_params.append(p.numel())


    """
    for n, p in tqdm(model.named_parameters(), desc="spectral scan"):
        if p.ndim != 2:
            continue                   # skip biases & LayerNorm weights

        # ---- skip gigantic embedding and LM‑head matrices ----
        if max(p.shape) > 4096:        # distilgpt2: 50k x 768 embeddings
            continue

        W = p.float().cpu()
        A = W.T @ W if W.shape[0] > W.shape[1] else W @ W.T   # choose smaller side
        eigs = torch.linalg.eigvalsh(A).numpy()
        alphas.append(hill_alpha(eigs))
        names.append(n)
        num_params.append(p.numel())

    block_metrics, n_params = np.array(block_metrics), np.array(n_params)

    # --- step 2 : layerwise sparsities ------------------------------
    layer_S = allocate_sparsity(block_metrics, n_params, args.sparsity)

    # --- step 3 : pruning pass -------------------------------------
    if args.backend == 'magnitude':
        for (name, p), S in zip(model.named_parameters(), layer_S):
            if p.ndim == 2 and S < 0.99:
                prune_tensor(p, 1-S)
    elif args.backend == 'wanda':
        from wanda_prune import wanda_prune_one_layer
        wanda_prune_one_layer(model, layer_S)
    else:
        from sparsegpt_prune import sparsegpt_prune_model
        sparsegpt_prune_model(model, layer_S)

    # --- optional : save mask for later LoRA fine‑tune -------------
    model.save_pretrained('alphaprune_%s_%dsparse' %
                          (args.model.split('/')[-1], int(args.sparsity*100)))

if __name__ == '__main__':
    main()
