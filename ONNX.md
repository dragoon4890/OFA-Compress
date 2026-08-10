# OFA-Tiny → ONNX export (Colab)

Exports the **fp32** OFA-Tiny encoder + decoder to two ONNX graphs (static shapes),
verifies them against PyTorch, runs a **greedy decode loop with ONNX Runtime**, then
builds the on-device artifact: **int8** via `quantize_dynamic` (Cell 10, with the
dedup slim in Cell 10b) or `quantize_static` QDQ (Cell 10q). fp16 (Cell 10c/10d) is
an optional fallback for engines that need native fp16 files.

- Checkpoint: `OFA-Sys/ofa-tiny` (322 MB fp32). Loaded **as-is** — no new fp32
  upload is needed; the official repo IS the fp32 endpoint.
- Image input: **480×480** (`patch_image_size`) → 900 image patches + text → ~908
  encoder tokens (fits `max_position_embeddings=1024`).
- Both graphs are **fully static** (no dynamic axes):
  - encoder: fixed 480×480 image + fixed caption prompt → constant `input_ids` length.
  - decoder: fixed window `WIN=32` with `attention_mask` (**True = padded**, the
    full model's convention) and an already-expanded `encoder_attention_mask`
    `[1,1,WIN,E]`; logits read at the last real-token index. Padding is safe because
    `_prepare_decoder_attention_mask` (modeling_ofa.py:1471) merges mask + causal mask.
- Batch-1 only (the encoder's no-padding `has_pads=False` path).
- The encoder's `if has_pads:` (modeling_ofa.py:1212) is a **data-dependent branch**.
  Newer torch versions default `torch.onnx.export` to the dynamo exporter, which
  refuses to trace it — so Cell 4b force-patches `has_pads = False` (valid because
  our inputs are padding-free) before exporting. The decoder has no such guards.
- Deliberately deferred: KV-cache/with-past decoder, NCNN, ExecuTorch. int8 (Cell 10)
  and QDQ int8 (Cell 10q) are covered in this workbook.

Known notes:
- The "Some weights of the model checkpoint were not initialized" warning is
  EXPECTED (5 buffers regenerated at init) — do not suppress or worry.
- `OFAModel.decoder.forward` output `last_hidden_state` already includes
  `output_projection` (modeling_ofa.py:1710) → it is vocab logits `[1, WIN, 59457]`.
- `code_masks=None` for captioning, so all code-branch ops drop out of the graphs.
- fp16 parity is checked with a **relaxed** tolerance and by decoded-token equality,
  since fp16 logits legitimately differ from fp32.
- If the default exporter fails, re-run with `dynamo=True` (Cell 5b / Cell 7b).

Run cells top to bottom. Everything is copy-paste into a Colab notebook.

---

## Cell 0 — Setup

```python
!git clone https://github.com/dragoon4890/OFA-Compress.git ofa-repo
%cd ofa-repo
!pip install -q transformers==4.44.2 onnx onnxruntime onnxconverter_common onnxscript
```

---

## Cell 1 — Patch-verify guard

The fork's `ofa/` needs its two compat patches (already committed). If this cell
prints `MISSING`, run `git pull` and **restart the kernel** (Runtime > Restart session),
then re-run from Cell 0.

```python
s = open("ofa/modeling_ofa.py", encoding="utf-8", errors="replace").read()
missing = []
if "from transformers.file_utils import" in s:
    missing.append("file_utils import still present (patch 1)")
if "generation_config=None" not in s:
    missing.append("_prepare_encoder_decoder_kwargs_for_generation patch (patch 2)")
print("compat patch check:", "OK" if not missing else "MISSING -> git pull + restart kernel: " + "; ".join(missing))
```

---

## Cell 2 — Download fp32 checkpoint

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="OFA-Sys/ofa-tiny", local_dir="ofa-tiny")
```

Faster alternative (same weights, buffers dropped + deduped): use
`dragoon49/ofa-tiny-slim-fp32` and `local_dir="ofa-tiny"`.

---

## Cell 3 — Load fp32 model + tokenizer (CPU)

```python
import torch
from ofa.tokenization_ofa import OFATokenizer
from ofa.modeling_ofa import OFAModel

tok = OFATokenizer.from_pretrained("ofa-tiny")
model = OFAModel.from_pretrained("ofa-tiny").to("cpu").eval()

assert model.dtype == torch.float32, f"expected fp32, got {model.dtype}"
print("model dtype:", model.dtype)
print("pad/bos/eos ids:", model.config.pad_token_id, model.config.bos_token_id, model.config.eos_token_id)
print("vocab:", model.config.vocab_size, " d_model:", model.config.d_model, " heads:", model.config.encoder_attention_heads)
```

Note: the un-initialized-weights warning is expected (5 regenerated buffers).

---

## Cell 4 — Smoke test: reference caption at 480×480

Builds the exact encoder inputs used for export, runs `model.generate` (greedy) —
this is the reference output for all parity checks. `S` = `1 (bos) + prompt + 1 (eos)`
is computed here and reused by the export cells.

```python
import os
from PIL import Image
from torchvision import transforms

IMG_SIZE = 480           # patch_image_size — baked into the graphs
PAD_ID = model.config.pad_token_id
BOS_ID = model.config.bos_token_id
EOS_ID = model.config.eos_token_id

tfm = transforms.Compose([
    lambda im: im.convert("RGB"),
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

image_path = "resources/caption_demo.png"
if not os.path.exists(image_path):
    image_path = "ofa-tiny/caption_demo.png"
img = tfm(Image.open(image_path)).unsqueeze(0)          # [1,3,480,480] fp32

prompt = " what does the image describe?"
src = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
src = torch.cat([torch.tensor([BOS_ID]), src, torch.tensor([EOS_ID])]).unsqueeze(0)  # [1,S]
patch_masks = torch.tensor([True])                      # [1] bool
S = src.size(1)

print("text seq S:", S, "-> encoder total:", 900 + S)

with torch.no_grad():
    ref = model.generate(
        input_ids=src, patch_images=img, patch_masks=patch_masks,
        num_beams=1, do_sample=False, max_length=16, min_length=1,
    )
ref_text = tok.batch_decode(ref, skip_special_tokens=True)[0]
print("reference greedy caption:", ref_text)
```

---

## Cell 4b — No-pad export patch (required before encoder export)

The encoder computes `has_pads = encoder_padding_mask.any()` (modeling_ofa.py:1203) and
branches on it at line 1212 (`if has_pads:`). That is a **data-dependent Python
branch**: whether padding exists depends on the actual input values. Newer torch
versions default `torch.onnx.export` to the dynamo exporter, which refuses to trace
such a guard (`GuardOnDataDependentSymNode: Could not guard on data-dependent
expression Eq(u0, 1)`).

Our inputs are always padding-free (batch-1, no pad tokens, `patch_masks=[True]`), so
we force `has_pads = False` — a constant — via an **in-memory** monkey-patch. The
same patch drops the bool `index_put` line `image_padding_mask[~patch_masks] = True`
(modeling_ofa.py:1192, a no-op when `patch_masks` is all-True): PyTorch exports it as
a `Where(16)` node with bool inputs, which ONNX Runtime CPU cannot run
(`NOT_IMPLEMENTED: Could not find an implementation for Where(16)`). The
sanity check proves the patched encoder is numerically identical on the smoke input.
This only affects the current Colab kernel (nothing written to disk).

**Warning:** do NOT keep this patch for a training run — padded batches would
silently take the wrong branch. Restart the kernel before any finetune/distill work.

```python
import inspect
import textwrap
from ofa import modeling_ofa as _m
from ofa.modeling_ofa import OFAEncoder

if getattr(OFAEncoder, "_export_patched", False):
    print("no-pad patch already applied this kernel — skipping")
else:
    assert not (src == PAD_ID).any(), "input has pad tokens — no-pad patch not valid"
    assert bool(patch_masks.all()), "patch_masks must be all-True — no-pad patch not valid"

    with torch.no_grad():
        _before = model.encoder(input_ids=src, patch_images=img, patch_masks=patch_masks).last_hidden_state

    _src = textwrap.dedent(inspect.getsource(OFAEncoder.forward))
    _old = "has_pads = encoder_padding_mask.any()"
    assert _old in _src, "unexpected encoder forward source"
    _src = _src.replace(_old, "has_pads = False")
    _noop = "image_padding_mask[~patch_masks] = True"
    assert _noop in _src, "unexpected index_put line"
    _src = _src.replace(_noop, "pass  # no-op for export: patch_masks all-True")

    _globs = dict(vars(_m))
    exec(_src.replace("def forward(", "def _export_forward("), _globs)
    OFAEncoder.forward = _globs["_export_forward"]
    OFAEncoder._export_patched = True

    with torch.no_grad():
        _after = model.encoder(input_ids=src, patch_images=img, patch_masks=patch_masks).last_hidden_state
    assert torch.equal(_before, _after), "no-pad patch changed encoder output — abort"
    print("no-pad patch applied; encoder output unchanged on smoke input")
```

Note: this bakes the batch-1/no-padding contract into the exported encoder. A
padded-batch export would instead need to trace the `has_pads=True` branch.

---

## Cell 5 — Export encoder (fully static)

Wraps `model.encoder` so positional `input_ids`/`patch_images`/`patch_masks` are
explicit (the raw forward is `input_ids, patch_images, patch_images_2, patch_masks,
...` — positional order would be a bug). Exports on CPU fp32. Requires Cell 4b's
no-pad patch to be applied first (otherwise the default exporter — which on newer
torch is the dynamo exporter — fails on the data-dependent `if has_pads:`).

```python
import torch

class EncoderWrapper(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
    def forward(self, input_ids, patch_images, patch_masks):
        out = self.enc(input_ids=input_ids, patch_images=patch_images, patch_masks=patch_masks)
        return out.last_hidden_state, out.padding_mask, out.position_embedding

enc_wrap = EncoderWrapper(model.encoder).eval()

with torch.no_grad():
    torch.onnx.export(
        enc_wrap,
        (src, img, patch_masks),
        "encoder.onnx",
        input_names=["input_ids", "patch_images", "patch_masks"],
        output_names=["last_hidden_state", "padding_mask", "position_embedding"],
        opset_version=17,
    )
print("encoder.onnx written")
```

---

## Cell 5b — (Only if Cell 5 failed) Export encoder with dynamo

Explicit dynamo exporter. Now works because Cell 4b removed the data-dependent
`if has_pads:` guard. Use **opset 18** (the dynamo exporter targets opset ≥ 18).
If it still trips, re-run with `TORCH_LOGS="dynamic"` or `draft_export()` and
report the traceback.

```python
with torch.no_grad():
    torch.onnx.export(
        enc_wrap,
        (src, img, patch_masks),
        "encoder.onnx",
        input_names=["input_ids", "patch_images", "patch_masks"],
        output_names=["last_hidden_state", "padding_mask", "position_embedding"],
        opset_version=18,
        dynamo=True,
    )
print("encoder.onnx written (dynamo)")
```

If this also fails, STOP and report the traceback — do not silently degrade.

---

## Cell 6 — Verify encoder (ONNX vs PyTorch)

Runs the exported graph in ONNX Runtime on CPU and compares all three
outputs to PyTorch. Pass/fail: max abs diff < 1e-4 (fp32).

```python
import onnxruntime as ort
import numpy as np

sess_enc = ort.InferenceSession("encoder.onnx", providers=["CPUExecutionProvider"])
inputs = {
    "input_ids": src.numpy(),
    "patch_images": img.numpy(),
    "patch_masks": np.array([True]),
}
onnx_last, onnx_pad, onnx_pos = sess_enc.run(None, inputs)

with torch.no_grad():
    ref_enc = model.encoder(input_ids=src, patch_images=img, patch_masks=patch_masks)

checks = [
    ("last_hidden_state", onnx_last, ref_enc.last_hidden_state.numpy()),
    ("padding_mask",      onnx_pad,  ref_enc.padding_mask.numpy()),
    ("position_embedding",onnx_pos,  ref_enc.position_embedding.numpy()),
]
ok = True
for name, a, b in checks:
    ma = np.abs(a.astype(np.float64) - b.astype(np.float64)).max()
    shape_ok = a.shape == b.shape
    status = "OK " if shape_ok and ma < 1e-4 else "FAIL"
    if status == "FAIL":
        ok = False
    print(f"{status} {name}: shape {a.shape} vs {b.shape}, max abs diff {ma:.2e}")
print("encoder parity:", "PASS" if ok else "FAIL")
```

---

## Cell 7 — Export decoder (static window)

Static window `WIN` (≥ max decode length). Inputs: `input_ids [1,WIN]`,
`attention_mask [1,WIN]` (**True = padded** — the full model's convention, see
modeling_ofa.py:1891-1892), `encoder_hidden_states [1,E,256]`,
`encoder_attention_mask [1,1,WIN,E]` (already expanded via `_expand_mask`, as the
full model does at modeling_ofa.py:1895-1897), `src_pos_embed [1,E,256]`.
Output: logits `[1,WIN,59457]`.
The decoder needs **no** no-pad patch: every branch in its forward
(`code_masks=None`, `past_key_values=None`, `use_cache=False`, config flags) is on a
constant, not on input data (modeling_ofa.py:1555-1719).

```python
import torch

WIN = 32   # decoder static window (must be >= max decode length)

class DecoderWrapper(torch.nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec
    def forward(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, src_pos_embed):
        out = self.dec(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            src_pos_embed=src_pos_embed,
            code_masks=None,
        )
        return out.last_hidden_state

dec_wrap = DecoderWrapper(model.decoder).eval()

with torch.no_grad():
    ref_enc = model.encoder(input_ids=src, patch_images=img, patch_masks=patch_masks)
E = ref_enc.last_hidden_state.size(1)          # 900 image patches + S text tokens
print("encoder length E:", E)

ex_ids = torch.full((1, WIN), PAD_ID, dtype=torch.long)
ex_ids[0, :2] = torch.tensor([BOS_ID, EOS_ID])
ex_am = torch.ones((1, WIN), dtype=torch.bool)   # decoder convention: True = padded
ex_am[0, :2] = False                              # first 2 tokens are real
ex_enc_hs = torch.zeros(1, E, model.config.d_model)
ex_enc_am = torch.zeros(1, 1, WIN, E)             # already-expanded 4D float mask
ex_src_pos = torch.zeros(1, E, model.config.d_model)

with torch.no_grad():
    torch.onnx.export(
        dec_wrap,
        (ex_ids, ex_am, ex_enc_hs, ex_enc_am, ex_src_pos),
        "decoder.onnx",
        input_names=["input_ids", "attention_mask", "encoder_hidden_states", "encoder_attention_mask", "src_pos_embed"],
        output_names=["logits"],
        opset_version=17,
    )
print("decoder.onnx written")
```

---

## Cell 7b — (Only if Cell 7 failed) Export decoder with dynamo

Explicit dynamo exporter, opset 18. The decoder has no data-dependent guards, so
this should succeed.

```python
with torch.no_grad():
    torch.onnx.export(
        dec_wrap,
        (ex_ids, ex_am, ex_enc_hs, ex_enc_am, ex_src_pos),
        "decoder.onnx",
        input_names=["input_ids", "attention_mask", "encoder_hidden_states", "encoder_attention_mask", "src_pos_embed"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
    )
print("decoder.onnx written (dynamo)")
```

If this also fails, STOP and report the traceback.

---

## Cell 8 — Verify decoder (ONNX vs PyTorch)

Feeds real encoder outputs. Logits differ slightly (fp32 op-reordering accumulates
through 4 decoder layers — a few 1e-4, sometimes 1e-3); logits are only consumed via
`argmax`, so **identical argmax is the pass criterion**, with `max abs diff < 1e-3`
as a sanity bound. The final word is Cell 9's decoded-caption match.

```python
import onnxruntime as ort
import numpy as np
from ofa.modeling_ofa import _expand_mask

with torch.no_grad():
    ref_enc = model.encoder(input_ids=src, patch_images=img, patch_masks=patch_masks)
enc_hs = ref_enc.last_hidden_state
enc_am = _expand_mask(ref_enc.padding_mask, torch.float32, WIN)   # [1,1,WIN,E] float
src_pos = ref_enc.position_embedding

T = 2
ids = torch.full((1, WIN), PAD_ID, dtype=torch.long)
ids[0, :T] = torch.tensor([BOS_ID, 42])     # arbitrary 2nd token; just a forward check
am = torch.ones((1, WIN), dtype=torch.bool)  # decoder convention: True = padded
am[0, :T] = False

sess_dec = ort.InferenceSession("decoder.onnx", providers=["CPUExecutionProvider"])
onnx_logits = sess_dec.run(None, {
    "input_ids": ids.numpy(),
    "attention_mask": am.numpy(),
    "encoder_hidden_states": enc_hs.numpy(),
    "encoder_attention_mask": enc_am.numpy(),
    "src_pos_embed": src_pos.numpy(),
})[0]

with torch.no_grad():
    ref_logits = model.decoder(
        input_ids=ids, attention_mask=am,
        encoder_hidden_states=enc_hs, encoder_attention_mask=enc_am,
        src_pos_embed=src_pos, code_masks=None,
    ).last_hidden_state

ma = np.abs(onnx_logits.astype(np.float64) - ref_logits.numpy().astype(np.float64)).max()
argmax_match = (onnx_logits.argmax(-1) == ref_logits.numpy().argmax(-1)).all()
print(f"logits shape {onnx_logits.shape} vs {ref_logits.shape}")
print(f"max abs diff {ma:.2e}   argmax identical: {argmax_match}")
print("decoder parity:", "PASS" if (ma < 1e-3 and argmax_match) else "FAIL")
```

---

## Cell 9 — Greedy decode loop with ONNX Runtime

Encoder once → padded-window decoder steps → argmax at the last real token → stop on
EOS or MAX_LEN. Primary pass/fail: **decoded text equals PyTorch greedy `generate`**
(also run on CPU so both sides are same-device).

```python
import onnxruntime as ort
import numpy as np

MAX_LEN = 16
WIN = 32

sess_enc = ort.InferenceSession("encoder.onnx", providers=["CPUExecutionProvider"])
sess_dec = ort.InferenceSession("decoder.onnx", providers=["CPUExecutionProvider"])

last_hs, pad2d, src_pos = sess_enc.run(None, {
    "input_ids": src.numpy(),
    "patch_images": img.numpy(),
    "patch_masks": np.array([True]),
})
E = last_hs.shape[1]
enc_am = np.where(np.broadcast_to(pad2d, (1, 1, WIN, E)), np.float32(-np.inf), np.float32(0.0))

def step(ids_list):
    L = len(ids_list)
    ids = np.full((1, WIN), PAD_ID, dtype=np.int64)
    ids[0, :L] = ids_list
    am = np.ones((1, WIN), dtype=bool)   # decoder convention: True = padded
    am[0, :L] = False
    logits = sess_dec.run(None, {
        "input_ids": ids,
        "attention_mask": am,
        "encoder_hidden_states": last_hs,
        "encoder_attention_mask": enc_am,
        "src_pos_embed": src_pos,
    })[0]
    return int(np.argmax(logits[0, L - 1]))     # logits at the last REAL token

dec = [BOS_ID]
for _ in range(MAX_LEN):
    nxt = step(dec)
    if nxt == EOS_ID:
        break
    dec.append(nxt)
ort_text = tok.batch_decode([dec], skip_special_tokens=True)[0]
print("ORT greedy caption   :", ort_text)

with torch.no_grad():
    ref = model.generate(
        input_ids=src, patch_images=img, patch_masks=patch_masks,
        num_beams=1, do_sample=False, max_length=16, min_length=1,
    )
ref_text = tok.batch_decode(ref, skip_special_tokens=True)[0]
print("PyTorch greedy caption:", ref_text)
print("MATCH:", ort_text == ref_text)
```

If lengths differ slightly (max_length counting semantics), compare the text; both
are greedy argmax so the token sequences should be identical.

**STOP — fp32 verified.** If `MATCH: True`, the fp32 ONNX pair is good. The primary
on-device artifact is **int8** (Cell 10) with the shared-embedding slim (Cell 10b).
Wait — the `onnxconverter_common.float16` pass is **dead** (produces invalid graphs);
fp16 (Cell 10c) is optional and only for fp16-only engines. Run order:

1. Cell 9b — audit what's inside the fp32 graphs (explains the 236 MB on disk).
2. Cell 10 — int8 (ORT `quantize_dynamic`) + verify greedy vs fp32 PyTorch.
3. Cell 10b — dedup the shared 59457×256 embedding across the int8 pair (`shared.data`).
4. Cell 10q — OPTIONAL `quantize_static` (QDQ int8), only if Cell 10 still drifts.
5. Cell 10c/10d — OPTIONAL fp16, only if a fp16-only engine is planned.
6. Cell 11 — save + summary + roadmap.

---

## Cell 9b — what is actually inside the fp32 graphs (initializer audit)

The fp32 pair is ~236 MB on disk. This cell audits where it goes — the motivation
for the int8 + slim steps. Expect the shared `59457×256` token embedding (~61 MB
fp32, ~30 MB fp16) to appear in **both** `encoder.onnx` and `decoder.onnx` — that duplication is
exactly what Cell 10b dedups.

```python
# Cell 9b — initializer census: largest tensors per graph
import onnx, os

def _n(i):
    n = 1
    for d in i.dims:
        n *= d
    return n

def _disk(f):
    s = os.path.getsize(f)
    if os.path.exists(f + ".data"):
        s += os.path.getsize(f + ".data")
    return s

for f in ["encoder.onnx", "decoder.onnx"]:
    m = onnx.load(f, load_external_data=False)
    inits = m.graph.initializer
    print(f"\n{f}: {_disk(f)/1e6:.1f} MB | {len(inits)} initializers | {sum(_n(i) for i in inits):,} params")
    for i in sorted(inits, key=_n, reverse=True)[:6]:
        print(f"    {'x'.join(map(str, i.dims)):22s} {_n(i):>12,}  {i.name}")
```

---

## Cell 10 — int8: the primary on-device artifact (dynamic quantization)

The fp32 pair is the interchange layer. The actual artifact is **int8**: ONNX
Runtime's `quantize_dynamic` keeps weights at QInt8 (per-channel) while activations
stay fp32 — so the greedy-loop IO contract from Cells 6/8 is unchanged (no
calibration set needed). **Note:** the `Embedding`/`Gather` weights are not
compressed — the shared 59457×256 token table stays fp32 in both graphs (that is
~2 copies of ~61 MB fp32); the slim step (Cell 10b) dedups it. If int8 accuracy drifts
(rare here), the upgrade path is `quantize_static` (QDQ + calibration; on ARM prefer
`QInt8` activations/weights and `per_channel`).

```python
# Cell 10 — Block A: int8 quantization of both validated fp32 graphs
import onnxruntime.quantization as qz
import onnx, os

# The decoder's final node emits the vocab logits through the tied output_projection
# (bias-free 59457×256 MatMul, modeling_ofa.py:1710). Argmax over 59k classes is the
# single most sensitive op — quantizing it is the #1 cause of int8 greedy drift
# (MATCH: False). Keep that node fp32.
dec = onnx.load("decoder.onnx")
logits_nodes = [n.name for n in dec.graph.node if "logits" in n.output]
print("kept fp32 (not quantized):", logits_nodes or "none found")

for f in ["encoder", "decoder"]:
    src_path = f"{f}.onnx"
    dst = f"{f}-int8.onnx"
    qz.quantize_dynamic(
        model_input=src_path,
        model_output=dst,
        weight_type=qz.QuantType.QInt8,
        per_channel=True,
        nodes_to_exclude=logits_nodes if f == "decoder" else [],
    )
    m = onnx.load(dst)
    onnx.save(m, dst, save_as_external_data=False)   # self-contained inline
    print(dst, f"{os.path.getsize(dst)/1e6:.1f} MB")
```

**What to expect:** the embedding table (`59457×256` fp32 = ~30 MB) is *not*
quantized and appears in both graphs — so the int8 pair still carries ~2× that table
plus the int8-ized matrices. That duplication is what Cell 10b kills with one shared
`shared.data` blob (~shrinks to the 34-MB-class target).

```python
# Block B — verify: int8 greedy decode == fp32 PyTorch output
import onnxruntime as ort
import numpy as np

s8e = ort.InferenceSession("encoder-int8.onnx", providers=["CPUExecutionProvider"])
s8d = ort.InferenceSession("decoder-int8.onnx", providers=["CPUExecutionProvider"])

last_hs, pad2d, src_pos = s8e.run(None, {
    "input_ids": src.numpy(),
    "patch_images": img.numpy(),
    "patch_masks": np.array([True]),
})
E = last_hs.shape[1]
enc_am = np.where(np.broadcast_to(pad2d, (1, 1, WIN, E)), np.float32(-np.inf), np.float32(0.0))

def step8(ids_list):
    L = len(ids_list)
    ids = np.full((1, WIN), PAD_ID, dtype=np.int64)
    ids[0, :L] = ids_list
    am = np.ones((1, WIN), dtype=bool)   # decoder convention: True = padded
    am[0, :L] = False
    logits = s8d.run(None, {
        "input_ids": ids,
        "attention_mask": am,
        "encoder_hidden_states": last_hs,
        "encoder_attention_mask": enc_am,
        "src_pos_embed": src_pos,
    })[0]
    return int(np.argmax(logits[0, L - 1]))

dec8 = [BOS_ID]
for _ in range(MAX_LEN):
    nxt = step8(dec8)
    if nxt == EOS_ID:
        break
    dec8.append(nxt)
ort8_text = tok.batch_decode([dec8], skip_special_tokens=True)[0]
print("ORT int8 greedy caption:", ort8_text)
print("MATCH:", ort8_text == ref_text)

# --- Diagnostics: where does int8 drift start? ---
# If MATCH is False, compare int8 vs fp32 ONNX directly: encoder outputs first,
# then the decoder's greedy token at each step. Step 0 of a mismatch pins it to
# the encoder (or decoder-projection) vs a later cascade from token choice.
import onnxruntime as ort
import numpy as np

sfe = ort.InferenceSession("encoder.onnx", providers=["CPUExecutionProvider"])
sfd = ort.InferenceSession("decoder.onnx", providers=["CPUExecutionProvider"])

fe_last, fe_pad, fe_pos = sfe.run(None, {
    "input_ids": src.numpy(),
    "patch_images": img.numpy(),
    "patch_masks": np.array([True]),
})
enc_diff = np.abs(fe_last.astype(np.float64) - last_hs.astype(np.float64)).max()
print(f"encoder int8 vs fp32  last_hidden_state max abs diff: {enc_diff:.2e}")

def step_fp(ids_list, hs, pad, pos):
    L = len(ids_list)
    ids = np.full((1, WIN), PAD_ID, dtype=np.int64)
    ids[0, :L] = ids_list
    am = np.ones((1, WIN), dtype=bool)
    am[0, :L] = False
    eam = np.where(np.broadcast_to(pad, (1, 1, WIN, hs.shape[1])), np.float32(-np.inf), np.float32(0.0))
    logits = sfd.run(None, {
        "input_ids": ids,
        "attention_mask": am,
        "encoder_hidden_states": hs,
        "encoder_attention_mask": eam,
        "src_pos_embed": pos,
    })[0]
    return int(np.argmax(logits[0, L - 1]))

d_fp, d_i8 = [BOS_ID], [BOS_ID]
first_div = None
for step in range(MAX_LEN):
    nf = step_fp(d_fp, fe_last, fe_pad, fe_pos)
    ni = step8(d_i8)
    if nf != ni and first_div is None:
        first_div = step
    if nf == EOS_ID and ni == EOS_ID:
        break
    d_fp.append(nf)
    d_i8.append(ni)
print("first divergent greedy step:", first_div)
print("fp32 tokens :", d_fp)
print("int8 tokens :", d_i8)

```

---

## Cell 10b — Slim the int8 pair: dedup the shared embedding into `shared.data`

Both int8 graphs still carry a private copy of the `59457×256` token embedding
(~61 MB each at fp32). The graphs reference that table by name (`Gather` input), so
the tensor must **stay present** — we only move its bytes out into one shared blob:

1. Save the encoder with all weights external into `shared.data` (`size_threshold=0`).
2. Repoint the decoder's embedding to the **same** offset/length inside `shared.data`
   (`set_external_data` + clear `raw_data` — do **not** remove the tensor from the
   graph; that makes the graph invalid for ORT).
3. Append the decoder's remaining weights to the existing `shared.data`
   (`write_external_data_tensors` — it appends; `onnx.save_model` on an existing
   blob may truncate it).

```python
# Cell 10b — Block A: build slim shells + shared blob (int8 pair)
import onnx, os
from onnx.external_data_helper import set_external_data, write_external_data_tensors

EMB_DIMS = [59457, 256]
ENC,  DEC  = "encoder-int8.onnx", "decoder-int8.onnx"
S_ENC, S_DEC = "encoder-int8-slim.onnx", "decoder-int8-slim.onnx"
BLOB = "shared.data"

def _emb_graph(m):
    hits = [t for t in m.graph.initializer if list(t.dims) == EMB_DIMS]
    assert len(hits) == 1, f"expected exactly 1 embedding, got {len(hits)}"
    return hits[0]

enc = onnx.load(ENC)
dec = onnx.load(DEC)
emb_enc, emb_dec = _emb_graph(enc), _emb_graph(dec)
assert emb_dec.raw_data == emb_enc.raw_data, "embeddings differ (bytes)"

if os.path.exists(BLOB):
    os.remove(BLOB)

# 1) encoder: all weights external into shared.data
onnx.save_model(
    enc, S_ENC,
    save_as_external_data=True, all_tensors_to_one_file=True,
    location=BLOB, size_threshold=0,
)
info = {kv.key: kv.value for kv in emb_enc.external_data}
emb_off, emb_len = int(info["offset"]), int(info["length"])
print("shared embedding blob: offset", emb_off, "length", emb_len)

# 2) decoder: repoint embedding to the same blob; append the rest of its weights
set_external_data(emb_dec, BLOB, emb_off, emb_len)
emb_dec.ClearField("raw_data")           # so it is NOT re-written below
for t in dec.graph.initializer:
    if t is not emb_dec:
        set_external_data(t, BLOB)       # location known; offset filled at write time
write_external_data_tensors(dec, os.getcwd())   # appends to existing shared.data
onnx.save_model(dec, S_DEC)

for f in [S_ENC, S_DEC, BLOB]:
    print(f"{f:26s} {os.path.getsize(f)/1e6:7.1f} MB")
inline = os.path.getsize(ENC) + os.path.getsize(DEC)
slim = os.path.getsize(S_ENC) + os.path.getsize(S_DEC) + os.path.getsize(BLOB)
print(f"inline: {inline/1e6:.1f} MB  ->  slim: {slim/1e6:.1f} MB")
```

```python
# Cell 10b — Block B: verify the slim pair decodes identically
import onnxruntime as ort
import numpy as np

s9e = ort.InferenceSession("encoder-int8-slim.onnx", providers=["CPUExecutionProvider"])
s9d = ort.InferenceSession("decoder-int8-slim.onnx", providers=["CPUExecutionProvider"])
last_hs, pad2d, src_pos = s9e.run(None, {
    "input_ids": src.numpy(),
    "patch_images": img.numpy(),
    "patch_masks": np.array([True]),
})
E = last_hs.shape[1]
enc_am = np.where(np.broadcast_to(pad2d, (1, 1, WIN, E)), np.float32(-np.inf), np.float32(0.0))
dec_s = [BOS_ID]
for _ in range(MAX_LEN):
    L = len(dec_s)
    ids = np.full((1, WIN), PAD_ID, dtype=np.int64)
    ids[0, :L] = dec_s
    am = np.ones((1, WIN), dtype=bool)
    am[0, :L] = False
    logits = s9d.run(None, {
        "input_ids": ids,
        "attention_mask": am,
        "encoder_hidden_states": last_hs,
        "encoder_attention_mask": enc_am,
        "src_pos_embed": src_pos,
    })[0]
    nxt = int(np.argmax(logits[0, L - 1]))
    if nxt == EOS_ID:
        break
    dec_s.append(nxt)
slim8_text = tok.batch_decode([dec_s], skip_special_tokens=True)[0]
print("ORT int8 slim greedy caption:", slim8_text)
print("MATCH:", slim8_text == ref_text)
```

---

## Cell 10c — OPTIONAL fp16 export (only if a fp16-only engine is required)

int8 is the primary artifact; this cell is a retained fallback for engines that need
native fp16 files (e.g. NCNN fp16a). Same direct-re-export approach as before.

```python
# fp16 export trips the data-dependent fp16 NaN/Inf clamp in OFAEncoderLayer.forward
# (modeling_ofa.py:514-518) — `torch.isinf(hidden_states).any()` cannot be traced
# (same dynamo problem as the has_pads guard). Neutralize it: a no-op for fp32, so
# the already-validated fp32 graph is unaffected.
import inspect, textwrap
from ofa import modeling_ofa as _m
from ofa.modeling_ofa import OFAEncoderLayer

if not getattr(OFAEncoderLayer, "_export_clamp_patched", False):
    _src = textwrap.dedent(inspect.getsource(OFAEncoderLayer.forward))
    _old = (
        "if hidden_states.dtype == torch.float16 and (\n"
        "    torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()\n"
        "):\n"
        "    clamp_value = torch.finfo(hidden_states.dtype).max - 1000\n"
        "    hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)"
    )
    assert _old in _src, "unexpected fp16 clamp block in encoder layer"
    _globs = dict(vars(_m))
    exec(
        _src.replace(_old, "pass  # no-op for export").replace("def forward(", "def _export_forward("),
        _globs,
    )
    OFAEncoderLayer.forward = _globs["_export_forward"]
    OFAEncoderLayer._export_clamp_patched = True
    print("fp16 NaN/Inf clamp patched out (encoder layer)")

with torch.no_grad():
    _c0 = model.encoder(input_ids=src, patch_images=img, patch_masks=patch_masks).last_hidden_state
    _c1 = model.encoder(input_ids=src, patch_images=img, patch_masks=patch_masks).last_hidden_state
assert torch.equal(_c0, _c1), "clamp patch changed fp32 encoder output — abort"
print("fp32 encoder output unchanged after clamp patch")

model16 = model.half()   # NOTE: in-place — restart kernel to get fp32 back
img16 = img.half()

enc_wrap16 = EncoderWrapper(model16.encoder).eval()
dec_wrap16 = DecoderWrapper(model16.decoder).eval()

WIN = 32
MAX_LEN = 16

with torch.no_grad():
    torch.onnx.export(
        enc_wrap16,
        (src, img16, patch_masks),
        "encoder-fp16.onnx",
        input_names=["input_ids", "patch_images", "patch_masks"],
        output_names=["last_hidden_state", "padding_mask", "position_embedding"],
        opset_version=17,
    )
print("encoder-fp16.onnx written")

with torch.no_grad():
    ref_enc16 = model16.encoder(input_ids=src, patch_images=img16, patch_masks=patch_masks)
E16 = ref_enc16.last_hidden_state.size(1)

ex_ids = torch.full((1, WIN), PAD_ID, dtype=torch.long)
ex_ids[0, :2] = torch.tensor([BOS_ID, EOS_ID])
ex_am = torch.ones((1, WIN), dtype=torch.bool)   # True = padded
ex_am[0, :2] = False
ex_enc_hs16 = torch.zeros(1, E16, model16.config.d_model, dtype=torch.float16)
ex_enc_am16 = torch.zeros(1, 1, WIN, E16, dtype=torch.float16)
ex_src_pos16 = torch.zeros(1, E16, model16.config.d_model, dtype=torch.float16)

with torch.no_grad():
    torch.onnx.export(
        dec_wrap16,
        (ex_ids, ex_am, ex_enc_hs16, ex_enc_am16, ex_src_pos16),
        "decoder-fp16.onnx",
        input_names=["input_ids", "attention_mask", "encoder_hidden_states", "encoder_attention_mask", "src_pos_embed"],
        output_names=["logits"],
        opset_version=17,
    )
print("decoder-fp16.onnx written")

# Inline weights -> one self-contained file per graph
import onnx
for f in ["encoder-fp16.onnx", "decoder-fp16.onnx"]:
    m = onnx.load(f)
    onnx.save(m, f, save_as_external_data=False)

import os
for f in ["encoder-fp16.onnx", "decoder-fp16.onnx"]:
    print(f, f"{os.path.getsize(f)/1e6:.1f} MB")
```

```python
# Block B — verify fp16: greedy decode vs fp16 PyTorch
import onnxruntime as ort
import numpy as np

sess_enc16 = ort.InferenceSession("encoder-fp16.onnx", providers=["CPUExecutionProvider"])
sess_dec16 = ort.InferenceSession("decoder-fp16.onnx", providers=["CPUExecutionProvider"])

last_hs, pad2d, src_pos = sess_enc16.run(None, {
    "input_ids": src.numpy(),
    "patch_images": img16.numpy(),
    "patch_masks": np.array([True]),
})
E16 = last_hs.shape[1]
enc_am = np.where(np.broadcast_to(pad2d, (1, 1, WIN, E16)), np.float16(-np.inf), np.float16(0.0))

def step16(ids_list):
    L = len(ids_list)
    ids = np.full((1, WIN), PAD_ID, dtype=np.int64)
    ids[0, :L] = ids_list
    am = np.ones((1, WIN), dtype=bool)   # decoder convention: True = padded
    am[0, :L] = False
    logits = sess_dec16.run(None, {
        "input_ids": ids,
        "attention_mask": am,
        "encoder_hidden_states": last_hs,
        "encoder_attention_mask": enc_am,
        "src_pos_embed": src_pos,
    })[0]
    return int(np.argmax(logits[0, L - 1]))

dec16 = [BOS_ID]
for _ in range(MAX_LEN):
    nxt = step16(dec16)
    if nxt == EOS_ID:
        break
    dec16.append(nxt)
ort16_text = tok.batch_decode([dec16], skip_special_tokens=True)[0]
print("ORT fp16 greedy caption:", ort16_text)

with torch.no_grad():
    ref16 = model16.generate(
        input_ids=src, patch_images=img16, patch_masks=patch_masks,
        num_beams=1, do_sample=False, max_length=16, min_length=1,
    )
ref16_text = tok.batch_decode(ref16, skip_special_tokens=True)[0]
print("PyTorch fp16 greedy caption:", ref16_text)
print("MATCH:", ort16_text == ref16_text)
```

Note: ONNX Runtime CPU has few native fp16 kernels; the loop above executes with
automatic casts. fp16 is for fp16-only engines; the int8 path (Cell 10 / Cell 10q)
is the on-device artifact. If either fp16 export fails, STOP and report the
traceback — do not fall back to `onnxconverter_common`, which produced the invalid
`_to_copy` graph.

---

## Cell 10d — OPTIONAL fp16 slim (shared embedding blob)

The checkpoint stores each weight **once** (~32.6 M params → 67 MB fp16). The ONNX
pair carries the shared 59457×256 token embedding in **both** graphs (an extra
~30 MB fp16). Pointing both graphs at one blob inside a shared `shared.data` drops
that duplicate, so the total lands at **~67 MB — the same as the fp16 checkpoint**
(the same trick on the fp32 pair reaches ~134 MB).

Note: the earlier 90.2/48.4 MB sizes were inflated by the buggy converter (it left
some tensors fp32). After the fixed Cell 10c (fp16 re-export) expect ~55 MB +
~41 MB; the slim set (Cell 10d) is what reaches 67 MB.

**Caveat for NCNN:** `onnx2ncnn` only reads inline `raw_data` — it does not load
external `.data` files. So this slim set is for **storage/distribution**; before
onnx2ncnn, re-inline (`onnx.load` + `onnx.save(m, f, save_as_external_data=False)`,
back to ~96 MB). `shared.data` must travel with the two slim shells.

```python
# Block A — build the shared external embedding blob
import onnx, os
from onnx.external_data_helper import set_external_data, write_external_data_tensors

ENC, DEC = "encoder-fp16.onnx", "decoder-fp16.onnx"
EMB_N = model.config.vocab_size * model.config.d_model   # 59457 * 256

def numel(t):
    n = 1
    for d in t.dims:
        n *= d
    return n

def is_emb(t):
    return numel(t) == EMB_N and t.data_type == onnx.TensorProto.FLOAT16

# 0) diagnostic: largest tensors per graph (the embedding should be the only >10M)
for f in (ENC, DEC):
    m = onnx.load(f, load_external_data=False)
    big = sorted((numel(i), "x".join(map(str, i.dims)), i.name) for i in m.graph.initializer)[-5:]
    print(f, "initializers:", len(m.graph.initializer), " params:", sum(numel(i) for i in m.graph.initializer))
    for n, dims, name in big:
        print(f"    {dims:20s} {n:>12,}  {name}")

enc = onnx.load(ENC)                 # weights loaded into memory
dec = onnx.load(DEC)

e_embs = [t for t in enc.graph.initializer if is_emb(t)]
d_embs = [t for t in dec.graph.initializer if is_emb(t)]
assert len(e_embs) == 1, f"expected 1 embedding in encoder, got {len(e_embs)}"
assert len(d_embs) >= 1, f"expected >=1 embedding in decoder, got {len(d_embs)}"

emb_bytes = bytes(e_embs[0].raw_data)
for t in d_embs:
    assert t.raw_data == emb_bytes, "decoder embedding != encoder embedding (bytes)"

if os.path.exists("shared.data"):
    os.remove("shared.data")

# 1) encoder -> all weights (incl. embedding) into shared.data
onnx.save_model(
    enc, "encoder-fp16-slim.onnx",
    save_as_external_data=True, all_tensors_to_one_file=True,
    location="shared.data", size_threshold=0,
)
info = {kv.key: kv.value for kv in e_embs[0].external_data}
emb_off, emb_len = int(info["offset"]), int(info["length"])
print("shared embedding blob: offset", emb_off, "length", emb_len)

# 2) decoder -> append non-embedding weights to shared.data; embedding -> shared blob
for t in dec.graph.initializer:
    if is_emb(t):
        set_external_data(t, "shared.data", emb_off, emb_len)
        t.ClearField("raw_data")     # so it is NOT re-written below
    else:
        set_external_data(t, "shared.data")   # offset/length assigned at write time
write_external_data_tensors(dec, os.getcwd())   # appends to the existing shared.data
onnx.save_model(dec, "decoder-fp16-slim.onnx")

for f in [ENC, DEC, "encoder-fp16-slim.onnx", "decoder-fp16-slim.onnx", "shared.data"]:
    print(f"{f:26s} {os.path.getsize(f)/1e6:7.1f} MB")
inline = os.path.getsize(ENC) + os.path.getsize(DEC)
slim = (os.path.getsize("encoder-fp16-slim.onnx") + os.path.getsize("decoder-fp16-slim.onnx")
        + os.path.getsize("shared.data"))
print(f"inline total: {inline/1e6:.1f} MB -> slim total: {slim/1e6:.1f} MB")
```

```python
# Block B — verify the slim pair decodes identically
import onnxruntime as ort
import numpy as np

sess_enc_s = ort.InferenceSession("encoder-fp16-slim.onnx", providers=["CPUExecutionProvider"])
sess_dec_s = ort.InferenceSession("decoder-fp16-slim.onnx", providers=["CPUExecutionProvider"])

last_hs, pad2d, src_pos = sess_enc_s.run(None, {
    "input_ids": src.numpy(),
    "patch_images": img16.numpy(),
    "patch_masks": np.array([True]),
})
E16 = last_hs.shape[1]
enc_am = np.where(np.broadcast_to(pad2d, (1, 1, WIN, E16)), np.float16(-np.inf), np.float16(0.0))

dec_s = [BOS_ID]
for _ in range(MAX_LEN):
    L = len(dec_s)
    ids = np.full((1, WIN), PAD_ID, dtype=np.int64)
    ids[0, :L] = dec_s
    am = np.ones((1, WIN), dtype=bool)   # True = padded
    am[0, :L] = False
    logits = sess_dec_s.run(None, {
        "input_ids": ids,
        "attention_mask": am,
        "encoder_hidden_states": last_hs,
        "encoder_attention_mask": enc_am,
        "src_pos_embed": src_pos,
    })[0]
    nxt = int(np.argmax(logits[0, L - 1]))
    if nxt == EOS_ID:
        break
    dec_s.append(nxt)
slim_text = tok.batch_decode([dec_s], skip_special_tokens=True)[0]
print("ORT slim fp16 greedy caption:", slim_text)
print("MATCH vs fp16 PyTorch:", slim_text == ref16_text)
```

---

## Cell 10q — OPTIONAL `quantize_static` (QDQ int8) if Cell 10 still drifts

Only if the Cell 10 `MATCH` stays False after the logits-node exclusion. QDQ
quantizes **activations** too (with calibration), which dynamic quantize leaves
fp32 — usually recovers the last few % of accuracy. Needs a small calibration set
of representative images (ideally from your deployment domain / train TSV).

```python
# Cell 10q — Block A: calibrate + build QDQ int8 pair
import os, base64, glob
from PIL import Image

# 1) gather calibration images: train.tsv base64 > calib_imgs/ > demo image
calib_pil = []
if os.path.exists("train.tsv"):
    import io
    with open("train.tsv", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            try:
                calib_pil.append(Image.open(io.BytesIO(base64.b64decode(parts[4]))).convert("RGB"))
            except Exception:
                continue
            if len(calib_pil) >= 32:
                break
if not calib_pil:
    calib_pil = [Image.open(p).convert("RGB") for p in sorted(glob.glob("calib_imgs/*"))]
if not calib_pil:
    p = "resources/caption_demo.png" if os.path.exists("resources/caption_demo.png") else "ofa-tiny/caption_demo.png"
    calib_pil = [Image.open(p).convert("RGB")]
print(f"calibration images: {len(calib_pil)}")

# 2) encoder calibration reader: fixed src + each image
import onnxruntime as ort
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantFormat, QuantType

sess_enc_fp32 = ort.InferenceSession("encoder.onnx", providers=["CPUExecutionProvider"])
src_np, pm_np = src.numpy(), np.array([True])

class EncCalib(CalibrationDataReader):
    def __init__(self, imgs):
        self.samples = [tfm(im).unsqueeze(0).numpy() for im in imgs]
        self.i = 0
    def get_next(self):
        if self.i >= len(self.samples):
            return None
        v = {"input_ids": src_np, "patch_images": self.samples[self.i], "patch_masks": pm_np}
        self.i += 1
        return v
    def rewind(self):
        self.i = 0

quantize_static(
    "encoder.onnx", "encoder-int8-qdq.onnx",
    calibration_data_reader=EncCalib(calib_pil),
    quant_format=QuantFormat.QDQ, per_channel=True,
    weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
)
print("encoder-int8-qdq.onnx written")

# 3) decoder calibration: encoder outputs per image, sampled at several seq lengths
import onnx
dec = onnx.load("decoder.onnx")
logits_nodes = [n.name for n in dec.graph.node if "logits" in n.output]
E = None
dec_calib = []
for im in calib_pil:
    hs, pad, pos = sess_enc_fp32.run(None, {
        "input_ids": src_np, "patch_images": tfm(im).unsqueeze(0).numpy(), "patch_masks": pm_np,
    })
    E = hs.shape[1]
    eam = np.where(np.broadcast_to(pad, (1, 1, WIN, E)), np.float32(-np.inf), np.float32(0.0))
    for L in (1, 2, 4, 8, 16):
        ids = np.full((1, WIN), PAD_ID, dtype=np.int64)
        ids[0, :L] = [BOS_ID] + [3] * (L - 1)          # synthetic prefix for range capture
        am = np.ones((1, WIN), dtype=bool)
        am[0, :L] = False
        dec_calib.append({
            "input_ids": ids,
            "attention_mask": am,
            "encoder_hidden_states": hs,
            "encoder_attention_mask": eam,
            "src_pos_embed": pos,
        })

class DecCalib(CalibrationDataReader):
    def __init__(self, samples):
        self.samples = samples
        self.i = 0
    def get_next(self):
        if self.i >= len(self.samples):
            return None
        v = self.samples[self.i]
        self.i += 1
        return v
    def rewind(self):
        self.i = 0

quantize_static(
    "decoder.onnx", "decoder-int8-qdq.onnx",
    calibration_data_reader=DecCalib(dec_calib),
    quant_format=QuantFormat.QDQ, per_channel=True,
    weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
    nodes_to_exclude=logits_nodes,        # keep the argmax-critical projection fp32
)
print("decoder-int8-qdq.onnx written")
for f in ["encoder-int8-qdq.onnx", "decoder-int8-qdq.onnx"]:
    print(f, f"{os.path.getsize(f)/1e6:.1f} MB")
```

```python
# Cell 10q — Block B: verify QDQ int8 greedy == fp32 PyTorch
import onnxruntime as ort
import numpy as np

qe = ort.InferenceSession("encoder-int8-qdq.onnx", providers=["CPUExecutionProvider"])
qd = ort.InferenceSession("decoder-int8-qdq.onnx", providers=["CPUExecutionProvider"])

last_hs, pad2d, src_pos = qe.run(None, {
    "input_ids": src.numpy(),
    "patch_images": img.numpy(),
    "patch_masks": np.array([True]),
})
E = last_hs.shape[1]
enc_am = np.where(np.broadcast_to(pad2d, (1, 1, WIN, E)), np.float32(-np.inf), np.float32(0.0))

def stepq(ids_list):
    L = len(ids_list)
    ids = np.full((1, WIN), PAD_ID, dtype=np.int64)
    ids[0, :L] = ids_list
    am = np.ones((1, WIN), dtype=bool)   # True = padded
    am[0, :L] = False
    logits = qd.run(None, {
        "input_ids": ids,
        "attention_mask": am,
        "encoder_hidden_states": last_hs,
        "encoder_attention_mask": enc_am,
        "src_pos_embed": src_pos,
    })[0]
    return int(np.argmax(logits[0, L - 1]))

decq = [BOS_ID]
for _ in range(MAX_LEN):
    nxt = stepq(decq)
    if nxt == EOS_ID:
        break
    decq.append(nxt)
ortq_text = tok.batch_decode([decq], skip_special_tokens=True)[0]
print("ORT QDQ int8 greedy caption:", ortq_text)
print("MATCH:", ortq_text == ref_text)
```

---

## Cell 11 — Save + summary

```python
import os, glob
print("artifacts:")
for f in sorted(set(glob.glob("*.onnx") + glob.glob("*.data") + glob.glob("*.onnx.data"))):
    print(f"  {f:26s} {os.path.getsize(f)/1e6:7.1f} MB")
```

Notes for on-device handoff:
- **Deployment roadmap.** ONNX is the interchange format, not a runtime — latency
  comes from the engine under it. Primary: **ONNX Runtime Mobile** (CPU EP, or
  NNAPI/CoreML later); Android alternative: **NCNN** (via `onnx2ncnn`). ExecuTorch is
  deferred: it would re-export via `torch.export`, resurrecting the same
  data-dependent guard pain (has_pads, fp16 clamp), for little gain at 33M params.
- **int8 is the on-device artifact** (Cell 10): `quantize_dynamic` QInt8 weights,
  fp32 activations, greedy loop IO unchanged. The shared 59457×256 embedding stays
  fp32 (Gather is not quantized) and appears in **both** graphs — Cell 10b points
  both at one blob in `shared.data`, giving ~2× the single-embedding size. If int8
  accuracy drifts, the upgrade is `quantize_static` QDQ (Cell 10q: calibrate on
  representative images; on ARM use `QInt8` activations/weights, `per_channel`).
- **fp16 (Cell 10c/10d) is optional** and only for fp16-only engines (NCNN fp16a).
  Cell 10c re-exports directly with fp16 weights (the `onnxconverter_common.float16`
  pass is dead — it produced the invalid `_to_copy` decoder); Cell 10d dedups the
  fp16 embedding the same way.
- **`onnx2ncnn` only reads inline `raw_data`** — it cannot load external `.data`
  files. So all slim sets are for storage/distribution only; before conversion
  re-inline (`onnx.load(m)` + `onnx.save(m, f, save_as_external_data=False)`).
  `shared.data` must stay next to the two slim shells.
- The exporter may write external `*.onnx.data` weight files next to small `.onnx`
  shells; keep files together when moving a model.
- `IMG_SIZE=480` is baked into `encoder.onnx` (re-export to change; 256 needs
  `ImageBucketSize`/position-id recompute — check `get_patch_images_info`).
- For a fp16 hub repo, the `config.json` should carry `"torch_dtype": "float16"`.
