from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, numpy as np, datasets, tqdm, os

model_dir = "alphaprune_distilgpt2_70sparse"  # adjust to your output
tok   = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir,
        torch_dtype=torch.float16, device_map={"": "mps" if torch.backends.mps.is_available() else "cpu"})
model.eval()

val = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")[:512]["text"]

def ppl(txt):
    inp = tok(txt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        loss = model(**inp, labels=inp["input_ids"]).loss
    return torch.exp(loss).item()

scores = [ppl(t) for t in tqdm.tqdm(val) if len(t.strip()) > 0]
print("Perplexity:", np.mean(scores))
