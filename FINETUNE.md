# OFA-Large caption fine-tune + TextBrewer distill to OFA-Tiny (Colab)

End-to-end Colab workbook. Runs on the fork's vendored `ofa/` + `textbrewer/` code
(no `trust_remote_code`). Flow: upload your TSVs -> build a CIDEr-D cache from your
**train** captions (the training code loads this unconditionally, see Cell 6) ->
fine-tune `OFA-Sys/ofa-large` on `caption_stage1` (cross-entropy, no stage-2 CIDEr
optimization) -> TextBrewer-distill that fine-tuned model into `OFA-Sys/ofa-tiny`.

Assumptions (verified against the fork's code):
- TSV is 5 tab-separated columns, read with `--selected-cols=0,4,2`:
  col 0 = uniq_id, col 4 = image (base64), col 2 = caption. Cell 5 prints the first
  row so you can confirm before training.
- `transformers==4.44.2` is pinned: `main_train.py`/`main_distill.py` use
  `transformers.AdamW`, which is removed in >= 4.50.
- The two `ofa/` compat patches are already committed in the fork (Cell 3 verifies).

Each cell does exactly one thing (SRP); run top to bottom. Cells 8 and 11 are the
long training runs (T4: ~hours for 5 epochs; the last line of the cell log shows
`Eval results` CIDEr at each checkpoint).

---

## Cell 1 — Config

Fill in your public fork URL (the clone target) and your TSV file names, then run.

```python
REPO_URL = "https://github.com/dragoon4890/OFA-Compress.git"  # <-- your fork URL
TRAIN_TSV = "train.tsv"   # 5 cols: id, ?, caption, ?, image-b64
VAL_TSV   = "val.tsv"     # same layout; multi-ref captions joined with '&&'
```

---

## Cell 2 — Clone fork + move into it

Clones the fork (contains the already-committed `ofa/` compat patches, the vendored
`textbrewer/`, `main_train.py`, `main_distill.py`, and the `tokenizer/` dir that
`init_task.py` requires in cwd).

```python
!git clone {REPO_URL} ofa-repo
%cd /content/ofa-repo
```

```python
import os
for p in ["ofa", "textbrewer", "train", "tokenizer", "main_train.py",
          "main_distill.py", "scripts/finetune/caption_finetune.sh",
          "scripts/distill/caption_distill.sh"]:
    print("ok " if os.path.exists(p) else "MISSING ", p)
```

---

## Cell 3 — Verify `ofa/` compat patches (already committed)

The fork has the two patches committed (commit 50f2285). This cell just confirms
them. If `False`, your clone is stale -> `!git pull` (or re-clone), then run
**Runtime > Restart session** before importing `ofa.modeling_ofa` (a running kernel
can hold a stale module -> false "already patched").

```python
s = open("ofa/modeling_ofa.py", encoding="utf-8").read()
p1 = "transformers.file_utils" not in s and "transformers.utils import" in s
p2 = "generation_config=None" in s
print("file_utils -> utils patch present:", p1)
print("generation_config kwarg present: ", p2)
assert p1 and p2, "patches missing -> pull latest fork, restart kernel"
```

---

## Cell 4 — Install deps

Pinned `transformers==4.44.2` (AdamW). Everything else is Colab-default
(torch, torchvision, pillow, numpy).

```python
!pip install -q transformers==4.44.2
```

```python
import torch, transformers
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
```

---

## Cell 5 — Upload + validate TSVs

Uploads `train.tsv` and `val.tsv` into the Colab VM (select both when the dialog
opens). If the files are already present (e.g. you copied them in via Drive), it
skips the upload prompt. Prints the first row so you can eyeball the columns before
a long training run.

```python
from google.colab import files
import os

if not (os.path.exists(TRAIN_TSV) and os.path.exists(VAL_TSV)):
    print("Upload both TSVs (multi-select):")
    uploaded = files.upload()
    for name, data in uploaded.items():
        with open(name, "wb") as f:
            f.write(data)
print("train exists:", os.path.exists(TRAIN_TSV))
print("val exists:  ", os.path.exists(VAL_TSV))
```

```python
for p in [TRAIN_TSV, VAL_TSV]:
    with open(p, encoding="utf-8") as f:
        first = f.readline().rstrip("\n")
        n = sum(1 for _ in f) + 1
    cols = first.split("\t")
    print(f"{p}: {n} rows, {len(cols)} cols")
    for i, c in enumerate(cols):
        print(f"  col {i}: {c[:50]!r}")
```

```python
# demo image for the verify cells: decode the first train row's base64 image
import base64, io
from PIL import Image

line = open(TRAIN_TSV, encoding="utf-8").readline().rstrip("\n").split("\t")
Image.open(io.BytesIO(base64.urlsafe_b64decode(line[4]))).convert("RGB").save("demo.png")
print("demo.png", Image.open("demo.png").size)
```

---

## Cell 6 — Build CIDEr-D cache from YOUR train captions

**Required.** `init_task.py` unconditionally builds `CiderD(df=args.eval_cider_cached_tokens)`
for caption tasks, and `CiderD` loads this pickle at startup -> without a valid file
training crashes before the first step. Format is what `metrics/ciderD_scorer.py`
expects: `ref_len` (float, log'd) + `document_frequency` (ngram->count). Built from
the **train** captions (col 2), punctuation-stripped exactly like evaluation does.

```python
import os, pickle, string
from collections import defaultdict

transtab = str.maketrans({key: None for key in string.punctuation})

refs = []
with open(TRAIN_TSV, encoding="utf-8") as f:
    for line in f:
        cap = line.rstrip("\n").split("\t")[2].translate(transtab).strip()
        if cap:
            refs.append([cap])

df = defaultdict(float)
for caps in refs:
    for cap in caps:
        words = cap.split()
        for k in range(1, 5):
            for i in range(len(words) - k + 1):
                df[tuple(words[i:i + k])] += 1

os.makedirs("cider_cached_tokens", exist_ok=True)
out = {"ref_len": float(sum(len(c) for c in refs)), "document_frequency": df}
with open("cider_cached_tokens/coco-valid-words.p", "wb") as f:
    pickle.dump(out, f)

print("ref_len:", out["ref_len"], "| distinct ngrams:", len(df))
```

---

## Cell 7 — Download OFA-Large + smoke-load (validates patches before a long run)

Weights-only download via `snapshot_download`; loaded with the vendored `OFAModel`.
The "Some weights ... not initialized" warning is expected (5 dropped buffers are
rebuilt at init) - do NOT worry about it.

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="OFA-Sys/ofa-large", local_dir="ofa-large")
print("ofa-large/pytorch_model.bin:", os.path.exists("ofa-large/pytorch_model.bin"))
```

```python
from ofa.tokenization_ofa import OFATokenizer
from ofa.modeling_ofa import OFAModel

tok = OFATokenizer.from_pretrained("tokenizer")
m = OFAModel.from_pretrained("ofa-large", use_cache=False)
n = sum(p.numel() for p in m.parameters())
print(f"OFA-Large loaded: {n/1e6:.0f}M params")
```

---

## Cell 8 — Fine-tune OFA-Large (stage-1, cross-entropy only)

Mirrors `scripts/finetune/caption_finetune.sh` with Colab-safe changes:
`--generator-version=hf` (script default is the vendored fairseq generator),
`--batch-size=4 --gradient-accumulation-steps=2` (effective 8; batch 8 at patch 480
can OOM a T4). Checkpoints land under `finetune_out/caption_stage1/<timestamp>/`,
with `saved_mode/` = best-by-CIDEr HF dir and `saved_mode_step/` = last eval.

`pipeline.initialize_distributed` uses `MASTER_PORT` from the env (default `6000`).
If a leftover/stale listener holds it you get
`DistNetworkError ... EADDRINUSE ... port: 6000` right after `device id: 0`. Fix:
run the cell below to pick a free port (it also kills any stray earlier
`main_train.py`/`main_distill.py` run), then start training.

```python
import os, socket, subprocess
subprocess.run("pkill -f main_train.py; pkill -f main_distill.py",
               shell=True, capture_output=True)
s = socket.socket()
s.bind(("127.0.0.1", 0))
os.environ["MASTER_PORT"] = str(s.getsockname()[1])
s.close()
os.environ["MASTER_ADDR"] = "localhost"
print("MASTER_ADDR/PORT:", os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
```

```python
# idempotent fix for the num_train_steps bug (committed in the fork @ cda4bb9;
# patched here so a stale clone works too). len(train_loader) is a list length 1,
# so with --gradient-accumulation-steps=2 the LR scheduler got num_train_steps=0
# -> ZeroDivisionError in get_polynomial_decay_schedule_with_warmup.
for _p in ["main_train.py", "main_distill.py"]:
    s = open(_p, encoding="utf-8").read()
    n = s.replace("train_loader) // args.gradient_accumulation_steps",
                  "train_loader[0]) // args.gradient_accumulation_steps")
    if n != s:
        open(_p, "w", encoding="utf-8").write(n)
        print("patched", _p, "(num_train_steps bug)")
    else:
        print("ok", _p, "(num_train_steps already fixed)")
```

```python
# idempotent fix for the save_pretrained shared-tensor error (committed in the
# fork @ 58a9fd5). transformers >= 4.40 rejects the OFA model's tied embedding
# (encoder/decoder embed_tokens + output_projection share one tensor) unless saved
# with safe_serialization=False -> RuntimeError at the first best-model save.
for _p in ["utils.py", "evaluation.py"]:
    s = open(_p, encoding="utf-8").read()
    n = s.replace('model.save_pretrained(os.path.join(args.output_dir, "saved_mode"))',
                  'model.save_pretrained(os.path.join(args.output_dir, "saved_mode"), safe_serialization=False)')
    n = n.replace('model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step"))',
                  'model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step"), safe_serialization=False)')
    n = n.replace('model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step_%d" % idx))',
                  'model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step_%d" % idx), safe_serialization=False)')
    if n != s:
        open(_p, "w", encoding="utf-8").write(n)
        print("patched", _p, "(save_pretrained safe_serialization)")
    else:
        print("ok", _p, "(save_pretrained already fixed)")
```

```python
!python main_train.py \
    --tables=$TRAIN_TSV,$VAL_TSV \
    --selected-cols=0,4,2 \
    --task=caption_stage1 \
    --generator-version=hf \
    --schedule=polynomial_decay \
    --warmup-proportion=0.06 \
    --lr=2e-5 \
    --lr-end=1e-7 \
    --label-smoothing=0.1 \
    --max-src-length=80 \
    --max-tgt-length=20 \
    --patch-image-size=480 \
    --eval-cider-cached-tokens=cider_cached_tokens/coco-valid-words.p \
    --beam=5 \
    --max-len-a=0 \
    --max-len-b=16 \
    --no-repeat-ngram-size=3 \
    --weight-decay=0.01 \
    --clip-grad=1.0 \
    --batch-size=4 \
    --micro-batch-size=4 \
    --gradient-accumulation-steps=2 \
    --num-epochs=5 \
    --best-score=10e10 \
    --metric=cider \
    --do-train \
    --do-predict \
    --ckpt-frequency=10 \
    --init-method=load_pretrain \
    --load=/content/ofa-repo/ofa-large \
    --student-model-config=ofa-large \
    --output-dir=/content/ofa-repo/finetune_out \
    --worker-cnt=1 \
    --gpus-per-node=1
```

> If `init_process_group` still fails with nccl, add `--distributed-backend gloo`.
> If you OOM, drop to `--batch-size=2`. Do NOT add `--fp16` (needs NVIDIA Apex).

---

## Cell 9 — Verify the fine-tuned model

Captions `demo.png` (from Cell 5) with the best `saved_mode/` checkpoint. Uses the
same 480px transform the model was trained with.

```python
import glob, os
from ofa.tokenization_ofa import OFATokenizer
from ofa.modeling_ofa import OFAModel
from PIL import Image
from torchvision import transforms
import torch

base = "finetune_out/caption_stage1"
latest = sorted(glob.glob(os.path.join(base, "*/")))[-1]
ckpt = os.path.join(latest, "saved_mode")
print("best checkpoint:", ckpt)

def caption(ckpt_dir, img_path, patch_size=480, device="cuda"):
    tok = OFATokenizer.from_pretrained("tokenizer")
    model = OFAModel.from_pretrained(ckpt_dir).to(device).eval()
    tfm = transforms.Compose([
        lambda im: im.convert("RGB"),
        transforms.Resize((patch_size, patch_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = tfm(Image.open(img_path)).unsqueeze(0).to(device)
    prompt = " what does the image describe?"
    src = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
    src = torch.cat([torch.tensor([tok.bos_token_id]), src,
                     torch.tensor([tok.eos_token_id])]).unsqueeze(0).to(device)
    with torch.no_grad():
        gen = model.generate(input_ids=src, patch_images=img,
                             patch_masks=torch.tensor([True]).to(device),
                             num_beams=5, max_length=16, min_length=1,
                             no_repeat_ngram_size=3)
    return tok.batch_decode(gen, skip_special_tokens=True)[0]

print("caption:", caption(ckpt))
```

---

## Cell 10 — Teacher + student paths, then download OFA-Tiny

Two traps in the fork are avoided by overriding `model_paths.py` in the working copy
(not committed): `pipeline.get_teacher_model` ignores `--load-teacher-model` when the
task is in `teacher_model_paths` and uses the `/home/xxx/...` placeholder instead;
`pipeline.get_student_model` raises unless the name is in the student dict. This
points both at the Colab paths used below.

```python
teacher_model_paths = {
    "pretrain": "/content/ofa-repo/teacher_finetuned",
    "caption_stage1": "/content/ofa-repo/teacher_finetuned",
    "refcoco": "/content/ofa-repo/teacher_finetuned",
    "refcocog": "/content/ofa-repo/teacher_finetuned",
    "refcocoplus": "/content/ofa-repo/teacher_finetuned",
    "snli_ve": "/content/ofa-repo/teacher_finetuned",
    "vqa_gen": "/content/ofa-repo/teacher_finetuned",
}
student_model_paths = {
    "load_pretrain": {
        "ofa-tiny": "/content/ofa-repo/ofa-tiny",
    }
}

src = "model_paths.py"
with open(src, "w", encoding="utf-8") as f:
    f.write("teacher_model_paths = " + repr(teacher_model_paths) + "\n\n")
    f.write("student_model_paths = " + repr(student_model_paths) + "\n")
print("model_paths.py overridden")
```

```python
import glob, os, shutil

latest = sorted(glob.glob(os.path.join("finetune_out/caption_stage1", "*/")))[-1]
src = os.path.join(latest, "saved_mode")
shutil.copytree(src, "teacher_finetuned", dirs_exist_ok=True)
print("teacher ->", "teacher_finetuned")
print("  pytorch_model.bin:", os.path.exists("teacher_finetuned/pytorch_model.bin"))
```

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="OFA-Sys/ofa-tiny", local_dir="ofa-tiny")
print("ofa-tiny/pytorch_model.bin:", os.path.exists("ofa-tiny/pytorch_model.bin"))
```

---

## Cell 11 — TextBrewer distill OFA-Large(finetuned) -> OFA-Tiny

Mirrors `scripts/distill/caption_distill.sh` (logit-KD `ce_with_mask`, weight 10000,
plus encoder+decoder attention-MSE intermediate matches on layers [0..3], since the
tiny student has 4+4 layers vs the teacher's 12+12). Teacher = your Cell 8 best
checkpoint (`teacher_finetuned`); student init = `OFA-Sys/ofa-tiny`. Batch 4 +
grad-accum 2 keeps a T4 alive.

Same free-port guard as Cell 8 (re-picks a port; also kills a stale run if any),
plus the idempotent `num_train_steps` bug fix (needed if the clone predates
fork commit `cda4bb9`):

```python
import os, socket, subprocess
subprocess.run("pkill -f main_train.py; pkill -f main_distill.py",
               shell=True, capture_output=True)
s = socket.socket()
s.bind(("127.0.0.1", 0))
os.environ["MASTER_PORT"] = str(s.getsockname()[1])
s.close()
os.environ["MASTER_ADDR"] = "localhost"
print("MASTER_ADDR/PORT:", os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
```

```python
for _p in ["main_train.py", "main_distill.py"]:
    s = open(_p, encoding="utf-8").read()
    n = s.replace("train_loader) // args.gradient_accumulation_steps",
                  "train_loader[0]) // args.gradient_accumulation_steps")
    if n != s:
        open(_p, "w", encoding="utf-8").write(n)
        print("patched", _p, "(num_train_steps bug)")
    else:
        print("ok", _p, "(num_train_steps already fixed)")
```

```python
# idempotent fix for the save_pretrained shared-tensor error (committed in the
# fork @ 58a9fd5) - same as Cell 8.
for _p in ["utils.py", "evaluation.py"]:
    s = open(_p, encoding="utf-8").read()
    n = s.replace('model.save_pretrained(os.path.join(args.output_dir, "saved_mode"))',
                  'model.save_pretrained(os.path.join(args.output_dir, "saved_mode"), safe_serialization=False)')
    n = n.replace('model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step"))',
                  'model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step"), safe_serialization=False)')
    n = n.replace('model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step_%d" % idx))',
                  'model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step_%d" % idx), safe_serialization=False)')
    if n != s:
        open(_p, "w", encoding="utf-8").write(n)
        print("patched", _p, "(save_pretrained safe_serialization)")
    else:
        print("ok", _p, "(save_pretrained already fixed)")
```

```python
!python main_distill.py \
    --generator-version=hf \
    --tables=$TRAIN_TSV,$VAL_TSV \
    --selected-cols=0,4,2 \
    --task=caption_stage1 \
    --schedule=polynomial_decay \
    --warmup-proportion=0.06 \
    --lr=2e-5 \
    --lr-end=1e-7 \
    --label-smoothing=0.1 \
    --kd-loss-weight=10000 \
    --kd-loss-type=ce_with_mask \
    --intermediate-matches=first:attention_mse_sum:encoder,first:attention_mse_sum:decoder \
    --max-src-length=80 \
    --max-tgt-length=20 \
    --patch-image-size=480 \
    --eval-cider-cached-tokens=cider_cached_tokens/coco-valid-words.p \
    --beam=5 \
    --max-len-a=0 \
    --max-len-b=16 \
    --no-repeat-ngram-size=3 \
    --weight-decay=0.01 \
    --clip-grad=1.0 \
    --batch-size=4 \
    --micro-batch-size=4 \
    --gradient-accumulation-steps=2 \
    --num-epochs=5 \
    --best-score=10e10 \
    --metric=cider \
    --do-train \
    --do-predict \
    --ckpt-frequency=10 \
    --init-method=load_pretrain \
    --student-model-config=ofa-tiny \
    --load-student-model=ofa-tiny \
    --load-teacher-model=/content/ofa-repo/teacher_finetuned \
    --output-dir=/content/ofa-repo/distill_out \
    --worker-cnt=1 \
    --gpus-per-node=1
```

> Same fallbacks as Cell 8 (gloo backend / batch 2 / no `--fp16`).

---

## Cell 12 — Verify the distilled student

Side-by-side caption of `demo.png`: original OFA-Tiny (256px, as `infer_caption.py`
does) vs the distilled student's best `saved_mode/` (256px). A slightly different
phrase from the baseline is expected and fine - the point is that it still produces
a sensible caption after the 12->4 layer distillation.

```python
import glob, os
import torch
from PIL import Image
from torchvision import transforms
from ofa.tokenization_ofa import OFATokenizer
from ofa.modeling_ofa import OFAModel

def caption(ckpt_dir, img_path, patch_size=256, device="cuda"):
    tok = OFATokenizer.from_pretrained("tokenizer")
    model = OFAModel.from_pretrained(ckpt_dir).to(device).eval()
    tfm = transforms.Compose([
        lambda im: im.convert("RGB"),
        transforms.Resize((patch_size, patch_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = tfm(Image.open(img_path)).unsqueeze(0).to(device)
    prompt = " what does the image describe?"
    src = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
    src = torch.cat([torch.tensor([tok.bos_token_id]), src,
                     torch.tensor([tok.eos_token_id])]).unsqueeze(0).to(device)
    with torch.no_grad():
        gen = model.generate(input_ids=src, patch_images=img,
                             patch_masks=torch.tensor([True]).to(device),
                             num_beams=5, max_length=16, min_length=1,
                             no_repeat_ngram_size=3)
    return tok.batch_decode(gen, skip_special_tokens=True)[0]

latest = sorted(glob.glob(os.path.join("distill_out/caption_stage1", "*/")))[-1]
ckpt = os.path.join(latest, "saved_mode")
print("baseline OFA-Tiny :", caption("ofa-tiny"))
print("distilled student :", caption(ckpt))
```

---

## Cell 13 — Record results (optional)

The best CIDEr for each run is in the run log (`log_0.txt` under the timestamped
`finetune_out/...` and `distill_out/...` dirs). Copy the two best checkpoint paths
here to keep a note for HANDOFF.md.

```python
import glob, os
for base in ["finetune_out/caption_stage1", "distill_out/caption_stage1"]:
    runs = sorted(glob.glob(os.path.join(base, "*/")))
    if runs:
        print(base, "->", runs[-1])
```
