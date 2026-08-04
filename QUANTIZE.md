# OFA-Tiny weight inspection (Colab)

Analysis only — cells 0–8 are read-only (nothing shrunk, converted, or saved).
Cells 9–11 are the fp32-slim shrink: Cell 9 writes `ofa-tiny-slim/`, Cells 10–11
verify it (10 = structural, 11 = inference). Cell 12 uploads it to Hugging Face
(weights-only, run once); Cell 13 is the fresh-Colab reuse path. Each cell does
exactly one thing (SRP); run top to bottom.

Checkpoint: `OFA-Sys/ofa-tiny` (`pytorch_model.bin`, ~322 MB fp32, ~80M params).
Config: d_model=256, 4+4 layers, ffn=1024, 4 heads, vocab=59457, resnet50.

---

## Cell 0 — Setup

Installs `transformers` and downloads the checkpoint to `./ofa-tiny`. That's all
the analysis needs — cells 1–8 only use `torch.load`, they do NOT require the
vendored `ofa/` source tree.

```python
!pip install -q transformers==4.44.2
```

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="OFA-Sys/ofa-tiny", local_dir="ofa-tiny")
```

```python
import os
print("checkpoint exists:", os.path.exists("ofa-tiny/pytorch_model.bin"))
print("vendored ofa/ package present:", os.path.isdir("ofa"))
```

## Cell 0b — Compat patches (OPTIONAL, skip for analysis)

Only needed to run inference with the vendored `ofa/` code (e.g. `infer_caption.py`)
— the weight analysis does not use it. Requires the repo code present in Colab
(its `ofa/` folder); skips cleanly if absent. Idempotent.

```python
import os

if not os.path.isdir("ofa"):
    print("ofa/ not present in Colab -> patches skipped (analysis does not need them)")
else:
    # patch 1: file_utils was removed from modern transformers
    for path in ["ofa/__init__.py", "ofa/modeling_ofa.py"]:
        s = open(path, encoding="utf-8").read()
        s2 = s.replace("from transformers.file_utils import", "from transformers.utils import")
        if s2 != s:
            open(path, "w", encoding="utf-8").write(s2)
            print("patched", path)
        else:
            print("already patched", path)

    # patch 2: transformers >= 4.40 passes a 4th positional arg to this method
    p = "ofa/modeling_ofa.py"
    s = open(p, encoding="utf-8").read()
    old = "self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None\n    ):"
    new = "self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None, generation_config=None\n    ):"
    if "generation_config=None" not in s and old in s:
        open(p, "w", encoding="utf-8").write(s.replace(old, new))
        print("patched _prepare_encoder_decoder_kwargs_for_generation")
    else:
        print("already patched")
```

---

## Cell 1 — Load state dict

Loads `pytorch_model.bin` and prints the grand totals: tensors, params, size.

```python
import torch
from collections import defaultdict

sd = torch.load("ofa-tiny/pytorch_model.bin", map_location="cpu")
n_params = sum(v.numel() for v in sd.values())
n_bytes = sum(v.numel() * v.element_size() for v in sd.values())
first = sd[list(sd)[0]]
print(f"tensors: {len(sd)}   params: {n_params:,}   size: {n_bytes/1e6:.1f} MB   dtype: {first.dtype}")
```

---

## Cell 2 — Component breakdown

Splits every tensor into a category (backbone / encoder / decoder / embeddings /
LayerNorm / BN stats) and prints params + MB + % of total per category.

```python
def component(key):
    if key.startswith("encoder.embed_images"):
        return "resnet50 backbone"
    if key.startswith("encoder.layers"):
        return "encoder transformer layers"
    if key.startswith("decoder.layers"):
        return "decoder transformer layers"
    if "embed_tokens" in key or key.endswith("output_projection.weight"):
        return "token embedding (59457 x 256)"
    if "embed_positions" in key or "embed_image_positions" in key:
        return "positional embeddings"
    if "layer_norm" in key or "layernorm" in key or "_ln.weight" in key or "_ln.bias" in key:
        return "LayerNorms"
    if "bn" in key or "running_" in key or key.endswith("num_batches_tracked"):
        return "BatchNorm stats"
    if key.startswith("encoder."):
        return "encoder misc"
    if key.startswith("decoder."):
        return "decoder misc"
    return "other"

groups = defaultdict(lambda: [0, 0])
for k, v in sd.items():
    g = component(k)
    groups[g][0] += v.numel()
    groups[g][1] += v.numel() * v.element_size()

total = sum(s for _, s in groups.values())
print(f"{'component':30s} {'params':>12s} {'MB':>7s} {'%':>6s}")
for g, (n, s) in sorted(groups.items(), key=lambda x: -x[1][1]):
    print(f"{g:30s} {n:12,} {s/1e6:6.1f} {s/total*100:5.1f}%")
```

---

## Cell 3 — Heads anatomy

Prints every tensor inside `encoder.layers.0` and `decoder.layers.0` so you can see
the attention heads (`q/k/v/out_proj`), FFN (`fc1`/`fc2`), and LayerNorms.

```python
def show_layer(prefix):
    print(f"--- {prefix} ---")
    for k, v in sd.items():
        if k.startswith(prefix):
            print(f"{k.replace(prefix, ''):38s} {str(tuple(v.shape)):18s} {v.numel():>10,}")

show_layer("encoder.layers.0.")
show_layer("decoder.layers.0.")
```

---

## Cell 4 — Quantizability map

Marks every tensor as `int8-capable` (Linear/Conv weights) or `fp16-only`
(Embedding / LayerNorm / BatchNorm / biases) and totals the MB per class.
This is the basis for any future fp16 / int8 shrink decision.

```python
def quant_class(k):
    if k.endswith(".weight") and any(x in k for x in
        ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2",
         "pos_q_linear", "pos_k_linear", "linear"]):
        return "int8-capable (Linear weight)"
    if k.endswith(".weight") and ".conv" in k:
        return "int8-capable (Conv2d weight)"
    if "embed_tokens" in k or "output_projection" in k or "embed_positions" in k or "embed_image_positions" in k or "type_embedding" in k:
        return "fp16-only (Embedding)"
    if "layer_norm" in k or "layernorm" in k or "_ln.weight" in k or "_ln.bias" in k:
        return "fp16-only (LayerNorm)"
    if "bn" in k or "running_" in k or k.endswith("num_batches_tracked"):
        return "fp16-only (BatchNorm)"
    if k.endswith(".bias"):
        return "fp16-only (bias)"
    return "fp16-only (other)"

mb = defaultdict(float)
for k, v in sd.items():
    mb[quant_class(k)] += v.numel() * v.element_size()
for c, s in sorted(mb.items(), key=lambda x: -x[1]):
    print(f"{c:26s} {s/1e6:7.1f} MB")
```

---

## Cell 5 — Duplicate embedding check

The decoder ties `output_projection` to `embed_tokens`. Checks whether the state
dict stores duplicate (identical) copies of the big 59457 x 256 embedding —
the classic source of "unnecessary" megabytes.

```python
keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "decoder.output_projection.weight"]
for a in keys:
    for b in keys:
        if a < b and a in sd and b in sd:
            print(f"{a}  ==  {b}  :  {torch.equal(sd[a], sd[b])}")
for k in keys:
    if k in sd:
        v = sd[k]
        print(f"{k:40s} {str(tuple(v.shape)):18s} {v.numel()*v.element_size()/1e6:6.1f} MB")
    else:
        print(f"{k:40s} (not in state dict)")
```

---

## Cell 6 — ResNet50 backbone

Lists every conv/BN tensor in the vision backbone (`encoder.embed_images.*`)
with shape and size — the ResNet50 half of the model.

```python
print(f"{'key':52s} {'shape':18s} {'MB':>6s}")
for k, v in sd.items():
    if k.startswith("encoder.embed_images"):
        print(f"{k:52s} {str(tuple(v.shape)):18s} {v.numel()*v.element_size()/1e6:5.1f}")
```

---

## Cell 7 — Real footprint

Scans every tensor: groups identical tensors by content hash (the duplicated
token embeddings), flags the regenerable index buffers (`*_rp_bucket`,
`image_position_idx` — pure functions of config, rebuilt at load), and prints
the minimum "real learned weights" size.

```python
import hashlib

def digest(t):
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()

def mb_of(k):
    v = sd[k]
    return v.numel() * v.element_size() / 1e6

# 1. identical copies (by content hash)
buckets = defaultdict(list)
for k, v in sd.items():
    buckets[digest(v)].append(k)
dup_groups = {h: ks for h, ks in buckets.items() if len(ks) > 1}
print("duplicate groups:")
for ks in sorted(dup_groups.values(), key=lambda x: -mb_of(x[0])):
    print(f"  x{len(ks)}  {mb_of(ks[0]):6.1f} MB  [{', '.join(ks)}]")

# 2. regenerable index buffers (rebuilt at load, so fully droppable)
buf_keys = {k for k in sd if "rp_bucket" in k or k.endswith("image_position_idx")}
buf_mb = sum(mb_of(k) for k in buf_keys)
print(f"regenerable buffers ({len(buf_keys)}): {buf_mb:.1f} MB  [{', '.join(sorted(buf_keys))}]")

# 3. duplicate waste = all but the first copy, ignoring buffers (already counted above)
dup_waste = 0.0
for ks in dup_groups.values():
    nb = [k for k in ks if k not in buf_keys]
    if len(nb) > 1:
        dup_waste += (len(nb) - 1) * mb_of(nb[0])
print(f"duplicate MB saveable (keep 1 copy each): {dup_waste:.1f}")

# 4. minimum real learned footprint
total_mb = sum(mb_of(k) for k in sd)
real_mb = total_mb - dup_waste - buf_mb
print(f"total {total_mb:.1f} - duplicates {dup_waste:.1f} - buffers {buf_mb:.1f} = real {real_mb:.1f} MB fp32  ->  {real_mb/2:.1f} MB fp16")
```

---

## Cell 8 — Shrink decision table

Turns the measured numbers into a per-component menu of actions and target sizes
(fp32 vs fp16 vs int8). Pick a row/column for the shrink phase — nothing is done
here.

```python
def mb_of(k):
    v = sd[k]
    return v.numel() * v.element_size() / 1e6

linear = sum(mb_of(k) for k in sd if k.endswith(".weight") and any(
    x in k for x in ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"]))
conv   = sum(mb_of(k) for k in sd if k.endswith(".weight") and ".conv" in k)
tok    = mb_of("encoder.embed_tokens.weight")          # 1 real copy
pos    = sum(mb_of(k) for k in sd if "embed_positions" in k or "embed_image_positions" in k)
small  = sum(mb_of(k) for k in sd if "layer_norm" in k or "layernorm" in k
             or "bn" in k or "running_" in k or k.endswith(".bias"))
bufs   = sum(mb_of(k) for k in sd if "rp_bucket" in k or k.endswith("image_position_idx"))

rows = [
    ("token embedding",       "dedupe x3",      tok,   tok/2,   tok/4),
    ("rp_bucket buffers",     "drop (rebuild)", bufs,  0.0,     0.0),
    ("resnet50 convs",        "int8",           conv,  conv/2,  conv/4),
    ("Linear weights",        "int8",           linear, linear/2, linear/4),
    ("positional embeddings", "fp16",           pos,   pos/2,   pos/2),
    ("BN/LN/bias",            "fp16",           small, small/2, small/2),
]
print(f"{'component':24s} {'action':17s} {'fp32':>6s} {'fp16':>6s} {'int8':>6s}")
tot = [0.0, 0.0, 0.0]
for name, act, a, b, c in rows:
    print(f"{name:24s} {act:17s} {a:6.1f} {b:6.1f} {c:6.1f}")
    tot[0] += a; tot[1] += b; tot[2] += c
print(f"{'TOTAL':24s} {'':17s} {tot[0]:6.1f} {tot[1]:6.1f} {tot[2]:6.1f}")
```

---

## Cell 9 — Build fp32-slim checkpoint

Loads the original state dict, drops the 5 regenerable buffers and the 2 duplicate
token-embedding copies, and saves `ofa-tiny-slim/pytorch_model.bin` (~134 MB).

```python
import os, shutil

sd = torch.load("ofa-tiny/pytorch_model.bin", map_location="cpu")

# 1. drop regenerable index buffers (rebuilt identically at load)
for k in [k for k in sd if "rp_bucket" in k or k.endswith("image_position_idx")]:
    del sd[k]

# 2. dedupe: keep one token-embedding copy (all 3 names are one shared tensor)
for k in ["decoder.embed_tokens.weight", "decoder.output_projection.weight"]:
    if k in sd:
        del sd[k]

os.makedirs("ofa-tiny-slim", exist_ok=True)
for f in ["config.json", "vocab.json", "merges.txt"]:
    src = os.path.join("ofa-tiny", f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join("ofa-tiny-slim", f))

torch.save(sd, "ofa-tiny-slim/pytorch_model.bin")
mb = sum(v.numel() * v.element_size() for v in sd.values()) / 1e6
print(f"saved {len(sd)} tensors -> ofa-tiny-slim/pytorch_model.bin  ({mb:.1f} MB)")
```

---

## Cell 10 — Structural verify (no ofa/ needed)

Re-loads the slim state dict and checks that only the 7 expected keys were removed
and every remaining tensor is byte-identical to the original.

```python
sd_orig = torch.load("ofa-tiny/pytorch_model.bin", map_location="cpu")
sd_slim = torch.load("ofa-tiny-slim/pytorch_model.bin", map_location="cpu")

removed = [k for k in sd_orig if k not in sd_slim]
print(f"removed keys ({len(removed)}):")
for k in removed:
    print(f"  {k}  {tuple(sd_orig[k].shape)}  {sd_orig[k].numel()*sd_orig[k].element_size()/1e6:.1f} MB")

extra = [k for k in sd_slim if k not in sd_orig]
print(f"unexpected keys: {extra or 'none'}")

mismatch = [k for k in sd_slim if not torch.equal(sd_orig[k], sd_slim[k])]
print(f"value mismatches vs original: {mismatch or 'none'}")

mb = sum(v.numel() * v.element_size() for v in sd_slim.values()) / 1e6
print(f"slim size: {mb:.1f} MB   ({len(sd_slim)} tensors)")
```

---

## Cell 11 — Inference verify (needs ofa/ + Cell 0b patches)

Loads `ofa-tiny-slim` with the vendored `OFAModel` and captions a demo image.
The slim caption must equal the fp32 caption. Guards cleanly if `ofa/` is absent.

```python
import os, torch
from PIL import Image
from torchvision import transforms

if not os.path.isdir("ofa"):
    print("ofa/ not in Colab -> copy the repo's ofa/ folder (or clone the fork), re-run Cell 0b, then this cell")
else:
    from ofa.tokenization_ofa import OFATokenizer
    from ofa.modeling_ofa import OFAModel

    image_path = "ofa-tiny/caption_demo.png"  # <-- set to your demo image if missing
    if not os.path.exists(image_path):
        image_path = "resources/caption_demo.png"

    tfm = transforms.Compose([
        lambda im: im.convert("RGB"),
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    prompt = " what does the image describe?"

    def caption(ckpt):
        tok = OFATokenizer.from_pretrained(ckpt)
        model = OFAModel.from_pretrained(ckpt).eval()
        img = tfm(Image.open(image_path)).unsqueeze(0)
        src = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        src = torch.cat([torch.tensor([tok.bos_token_id]), src,
                         torch.tensor([tok.eos_token_id])]).unsqueeze(0)
        with torch.no_grad():
            gen = model.generate(input_ids=src, patch_images=img,
                                 patch_masks=torch.tensor([True]),
                                 num_beams=5, max_length=16, min_length=1,
                                 no_repeat_ngram_size=3)
        return tok.batch_decode(gen, skip_special_tokens=True)[0]

    c32 = caption("ofa-tiny")
    cs  = caption("ofa-tiny-slim")
    print(f"fp32 : {c32}")
    print(f"slim : {cs}")
    print(f"match: {c32 == cs}")
```

---

## Cell 12 — Upload slim checkpoint to Hugging Face

Publishes `ofa-tiny-slim/` as a **weights-only** model repo (~134 MB). Run once.
Set `REPO_ID` to your HF username + repo name first.

```python
from huggingface_hub import login, HfApi

REPO_ID = "dragoon49/ofa-tiny-slim-fp32"   # published 2026-08-04, 134.3 MB

login()                      # paste an HF write token when prompted
api = HfApi()
api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=REPO_ID, folder_path="ofa-tiny-slim")
print("files on repo:")
for f in api.list_repo_files(REPO_ID):
    print("  ", f)
```

---

## Cell 13 — Reuse from HF (fresh Colab, no shrink needed)

Future sessions: skip Cells 1–9 (and Cell 12). Download the slim repo, then load it
with the vendored `OFAModel` exactly as Cell 11 does. Needs the fork's `ofa/` +
Cell 0b patches (weights-only repo = no model code inside).

```python
from huggingface_hub import snapshot_download

REPO_ID = "dragoon49/ofa-tiny-slim-fp32"
snapshot_download(repo_id=REPO_ID, local_dir="ofa-tiny-slim")
print("downloaded ofa-tiny-slim/")
```

```python
import os, torch
from PIL import Image
from torchvision import transforms

assert os.path.isdir("ofa"), "copy the fork's ofa/ folder + run Cell 0b first"

from ofa.tokenization_ofa import OFATokenizer
from ofa.modeling_ofa import OFAModel

image_path = "ofa-tiny/caption_demo.png"
if not os.path.exists(image_path):
    image_path = "resources/caption_demo.png"

tfm = transforms.Compose([
    lambda im: im.convert("RGB"),
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
prompt = " what does the image describe?"

tok = OFATokenizer.from_pretrained("ofa-tiny-slim")
model = OFAModel.from_pretrained("ofa-tiny-slim").eval()
img = tfm(Image.open(image_path)).unsqueeze(0)
src = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
src = torch.cat([torch.tensor([tok.bos_token_id]), src,
                 torch.tensor([tok.eos_token_id])]).unsqueeze(0)
with torch.no_grad():
    gen = model.generate(input_ids=src, patch_images=img,
                         patch_masks=torch.tensor([True]),
                         num_beams=5, max_length=16, min_length=1,
                         no_repeat_ngram_size=3)
print("caption:", tok.batch_decode(gen, skip_special_tokens=True)[0])
```
