# Handoff — OFA-Tiny Image Captioning on OFA-Compress (HF)

## Goal

Build a minimal OFA-Tiny image captioning pipeline on the **OFA-Compress fork**
(`https://github.com/dragoon4890/OFA-Compress`, cloned at
`C:\Users\harsh\Desktop\curr\github\OFA-Compress`), then fine-tune on COCO, and
eventually deploy on-device (ONNX/NCNN → Snapdragon 660 — **deferred**).

Working model: **OFA-Tiny** (~72M real params incl. ResNet50 backbone; the "33M"
claim excludes backbone/embeddings). Weights come from HF repo
`OFA-Sys/ofa-tiny` (`pytorch_model.bin`, ~322 MB fp32).

## Current status

- **Phase A DONE and verified in Google Colab**: single-image caption inference
  works on CPU, ~1.5 s/caption, model loads in ~1 s.
  Sample output for a two-cat photo: `a pair of cats laying on a bed`.
- Working environment is **Colab**, not Windows (local Windows box has no
  `transformers`; the old fairseq path is broken there by numpy 2.x).
- The old fairseq `OFA` repo at `C:\Users\harsh\Desktop\curr\OFA` is **redundant** —
  delete-able. Keep only for reference (its `datasets.md`/`checkpoints.md`/`run_scripts`
  have the exact upstream data URLs/configs).
- **Fork git state**: `main` @ `58a9fd5` "Save checkpoints with safe_serialization=False"
  (pushed to `origin/main`). `ofa/` carries the 2 compat patches (50f2285).
  `QUANTIZE.md` has uncommitted Cells 14–18 (fp16 build + SVD rank check).
  `HANDOFF.md` is untracked by design. Remotes: `origin` = `dragoon4890/OFA-Compress`,
  `upstream` = `OFA-Sys/OFA-Compress`.

## Files in the fork

- `infer_caption.py` — the working inference script (entry point). Args:
  `--generator hf|fairseq` (default `hf`), `--device`, `--num-beams` (5),
  `--max-length` (16), `--no-repeat-ngram-size` (3). Expects the checkpoint in
  `./ofa-tiny`.
- `scripts/finetune/caption_finetune.sh`, `scripts/distill/caption_distill.sh`,
  `scripts/evaluate/caption_evaluate.sh`, `scripts/pretrain.sh` — Phase B/C scripts.
- `ofa/` — vendored HF model (`modeling_ofa.py`, `configuration_ofa.py`,
  `tokenization_ofa.py`, `resnet.py`). **Carries the 2 compat patches (commit 50f2285).**
- `FINETUNE.md` — **Colab workbook** (Cells 1–13): fine-tune OFA-Large on your own
  TSVs then TextBrewer-distill into OFA-Tiny. Copy-paste cells, run top to bottom.
- `generate/` — vendored fairseq-style `SequenceGenerator` (HF fallback path).

## CRITICAL: compat patches are COMMITTED in the fork (commit 50f2285, pushed)

The vendored OFA code was written for `transformers 4.16`. With modern
`transformers` (4.44.2 verified) two patches were required; they are now **committed
in the fork** and are applied automatically on `git clone`. Commit `50f2285`
(2 files, 4 lines changed):

1. `ofa/__init__.py` + `ofa/modeling_ofa.py`:
   `from transformers.file_utils import` → `from transformers.utils import`
   (removed in modern transformers).
2. `ofa/modeling_ofa.py`, method `_prepare_encoder_decoder_kwargs_for_generation`:
   `generation_config=None` added to the signature. transformers ≥ ~4.40 passes a
   4th positional arg; without it: `TypeError: takes from 3 to 4 positional
   arguments but 5 were given`.

FINETUNE.md Cell 3 verifies both anchors and tells you to restart the kernel if a
stale module is cached. (A Colab kernel that imported `ofa` before pulling a change
can hold a stale module → false "already patched"; restart the session.)

### Gotcha — do NOT use the `edit` tool on `ofa/modeling_ofa.py`
The edit tool rewrote the file and applied formatting (671-line diff vs HEAD,
2025 → 2314 lines). It was reverted via `git restore`. The committed patch was made
with a python string-replace preserving the file byte-for-byte (CRLF line endings —
match `\r\n` in multi-line anchors).

## Verified technical facts (Phase A)

- Image preprocessing (matches `data_utils/caption_dataset.py:59-64`):
  `RGB` → `Resize((256,256), BICUBIC)` → `ToTensor` → `Normalize(0.5, 0.5)`.
  Prompt: `" what does the image describe?"` (exact string, from `caption_dataset.py:88`).
- Input ids = `[BOS(0)] + tokenize(prompt, no special) + [EOS(2)]`.
- HF generate call:
  `model.generate(input_ids=src, patch_images=patch_img, patch_masks=torch.tensor([True]), num_beams=5, max_length=16, min_length=1, no_repeat_ngram_size=3)`.
- `OFA-Sys/ofa-tiny` config: `is_encoder_decoder=true`, `decoder_start_token_id=0`,
  `forced_eos_token_id=2`, `vocab_size=59457`, d_model 256, 4 layers, 4 heads,
  `resnet50`, `use_cache=false`.
- `prepare_inputs_for_generation` zeroes the attention mask and drops
  `patch_images` after encoding; `_prepare_encoder_decoder_kwargs_for_generation`
  defaults `patch_masks` to `torch.tensor([True])`. KV-cache is not propagated
  across decode steps in either generator path (old `past=` arg vs modern
  `past_key_values=`) — correct output, just not incremental.
- Fallback path: vendored `SequenceGenerator(tokenizer, beam_size=5, max_len_b=16,
  min_len=1, no_repeat_ngram_size=3)`; call `gen.generate([model],
  {"net_input": {"input_ids": src, "patch_images": patch_img, "patch_masks": ...}})`,
  then `tokenizer.batch_decode([out[0][0]["tokens"]], skip_special_tokens=True)`.
  Works via the `EnsembleModel` wrapper calling `model.encoder`/`model.decoder`.
- Colab stack used: `transformers==4.44.2`, Colab-bundled torch/torchvision,
  checkpoint at `./ofa-tiny` via `snapshot_download(repo_id="OFA-Sys/ofa-tiny", local_dir="ofa-tiny")`.

## Weight analysis (QUANTIZE.md)

`QUANTIZE.md` is the Colab weight-inspection workflow (cells 0–18 done; 19–23 int8
pending). Checkpoint `OFA-Sys/ofa-tiny` = **322.5 MB fp32, ~72.3M params**.
Findings:

- **3× token-embedding copies (182.7 MB)**: `encoder.embed_tokens` /
  `decoder.embed_tokens` / `decoder.output_projection` are one shared tensor
  (`shared = nn.Embedding(...)` at `ofa/modeling_ofa.py:1739`, tied via
  `output_projection.weight = embed_tokens.weight`), saved 3× (3 × 60.9 MB).
  Keeping 1 copy saves **121.8 MB**. Dedup is lossless: `from_pretrained` loads
  with `strict=False` and the single kept copy populates the shared tensor.
- **66.6 MB regenerable index buffers**: `encoder/decoder.token_rp_bucket`
  ([1026×1026], int64), `encoder/decoder.image_rp_bucket` ([1765×1765], int64),
  `decoder.image_position_idx`. Pure deterministic functions of config
  (`make_token_bucket_position`/`make_image_bucket_position`,
  `ofa/modeling_ofa.py:98,114`) — rebuilt identically at init, so dropping them is
  lossless. (The encoder/decoder copies of each bucket are also byte-identical.)
- **Real learned weights: 134.1 MB fp32 → 67.0 MB fp16** (~49 MB int8 Linear/Conv +
  fp16 rest; ~35 MB full int8 incl. embedding).

Size targets: **fp32-slim 134.1 / fp16 67.0 / int8 ~49 / full-int8 ~35 MB**.
The fp32-slim checkpoint (`ofa-tiny-slim/`) is built by QUANTIZE.md Cell 9 (drop
buffers + dedupe embedding) and verified by Cells 10 (structural) / 11 (inference).
**fp16 (`ofa-tiny-fp16/`, ~67 MB) is built by Cell 14 and verified by Cells 15–16.**
Cell 18 (SVD rank spectrum) found a **flat spectrum** → low-rank factorization has
~7 MB of headroom, not worth it → **int8 quantization (Cells 19–23) is the next
shrink step** (~35 MB full-int8, ~49 MB int8+fp16-embed variant).

- **Published to HF**: `dragoon49/ofa-tiny-slim-fp32` (weights-only, 134.3 MB,
  uploaded 2026-08-04) and `dragoon49/ofa-tiny-fp16` (weights-only, ~67 MB).
  Verified in Colab: fp32 vs slim caption both `a pair of cats laying on a bed`
  (`match: True`). Loading still needs the fork's `ofa/` + the two compat patches —
  the repo has no model code. Reuse path: QUANTIZE.md Cell 13 (`snapshot_download`
  + vendored `OFAModel`).
- Load-time warning "Some weights of OFAModel were not initialized ..."
  (`decoder.image_position_idx`, `*_rp_bucket`, etc.) is expected/harmless — those
  5 buffers are regenerated at init from config. Deliberately not silenced.

## Next steps

1. **Phase B/C — fine-tune + distill via `FINETUNE.md`** (Colab workbook, cells
   copy-paste ready). Code-verified facts to rely on when debugging:
   - `main_train.py`/`main_distill.py` use `transformers.AdamW` → pin
     `transformers==4.44.2` (removed in ≥ 4.50). AdamW emits a DeprecationWarning — fine.
   - Both entry points are single-process-friendly: `args.rank/world_size` come from
     env with defaults 0/1 (`arguments.py:604-605`), so run `!python main_train.py`
     directly — no `torch.distributed.launch` needed. Add `--distributed-backend gloo`
     only if nccl init fails.
   - **EADDRINUSE on port 6000**: `pipeline.initialize_distributed` uses env
     `MASTER_PORT` (default `6000`). A stale/leftover listener (e.g. a crashed or
     still-running earlier `main_train.py`) makes `init_process_group` fail right
     after `device id: 0`. FINETUNE.md Cells 8/11 pick a free port first
     (`socket.bind(("127.0.0.1",0))` → `os.environ["MASTER_PORT"]`) and `pkill`
     stray runs.
   - **num_train_steps bug (fixed, fork @ cda4bb9)**: both entry points computed
     `len(train_loader)` for the single-dataset case, but `get_data_loader` returns a
     **list** holding one DataLoader, so that is always `1`. With
     `--gradient-accumulation-steps=2` → `1 // 2 = 0` → `num_train_steps = 0` →
     `ZeroDivisionError` in `get_polynomial_decay_schedule_with_warmup` right after
     `Train_config:` prints. Fixed to `len(train_loader[0])` (1 line each).
     FINETUNE.md Cells 8/11 apply this idempotently to a stale clone.
     Note: `num_train_steps` is used only for the LR schedule; `BasicTrainer` still
     iterates `num_epochs` full epochs.
   - **save_pretrained shared-tensor RuntimeError (fixed, fork @ 58a9fd5)**: after the
     first eval beats the baseline, `utils.py:197`/`evaluation.py` call
     `model.save_pretrained("saved_mode"/"saved_mode_step")`. transformers ≥ 4.40 with
     `safe_serialization=True` raises
     `RuntimeError: ... shared tensors [{'encoder.embed_tokens.weight',
     'decoder.output_projection.weight', 'decoder.embed_tokens.weight'}] ...` because
     the vendored OFA model ties embeddings without declaring them. Fix: pass
     `safe_serialization=False` to every `model.save_pretrained(...)` call
     (writes `pytorch_model.bin`; pickle memo preserves the sharing, and
     `OFAModel.from_pretrained(..., strict=False)` loads it). FINETUNE.md Cells 8/11
     apply this idempotently to a stale clone. Note: `save_checkpoint`'s
     `torch.save(sd, checkpoint_name)` is commented out in `utils.py`, so
     `saved_mode/` (best) and `saved_mode_step/` (last) are the only checkpoints.
   - `--generator-version` default is **fairseq** (`arguments.py:397`). Pass
     `--generator-version=hf` everywhere so eval uses `model.generate` (hf) instead of
     the vendored fairseq generator; hf also forces `use_cache=False`.
   - **CIDEr cache is mandatory**: `init_task.py:97` builds
     `CiderD(df=args.eval_cider_cached_tokens)` for caption tasks unconditionally →
     a valid pickle must exist or training crashes at startup. FINETUNE.md Cell 6
     builds it from your train TSV (`ref_len` float + `document_frequency` as a
     **defaultdict**, so unseen hyp ngrams → df 0, no KeyError).
   - **Teacher/student path traps** (neutralized by FINETUNE.md Cell 10 writing a
     Colab-path `model_paths.py`): `pipeline.get_teacher_model` uses
     `teacher_model_paths[task]` and **ignores `--load-teacher-model`** when the task
     is in the dict; `pipeline.get_student_model` **raises** unless the name is in
     `student_model_paths["load_pretrain"]`. The cell rewrites those dicts to
     `/content/ofa-repo/teacher_finetuned` and `/content/ofa-repo/ofa-tiny`.
   - Finetune init = `--init-method=load_pretrain --load=<ofa-large snapshot dir>`;
     `--student-model-config=ofa-large` only matters for `init_method=random`.
   - Distill = `main_distill.py` + `OFADistiller` (vendored `textbrewer`): teacher
     wrapped in `torch.no_grad()`, `--kd-loss-weight=10000 --kd-loss-type=ce_with_mask
     --intermediate-matches=first:attention_mse_sum:encoder,first:attention_mse_sum:decoder`
     (teacher 12+12 layers vs student 4+4 → `first` matches layers [0..3]).
   - Best checkpoints: `evaluation.py` saves `saved_mode/` (best CIDEr) and
     `saved_mode_step/` (last) via `save_pretrained`, under
     `<output-dir>/<task>/<timestamp>/`.
   - Do NOT pass `--fp16` to train/distill — needs NVIDIA Apex
     (`train/basic_trainer.py:110-118`). OOM fallback: batch 4 → 2 (or patch 384).
   - Data: `--selected-cols=0,4,2`, `--tables=train.tsv,val.tsv` (last = val);
     images are base64 in col 4; val multi-ref captions joined with `&&`
     (evaluation.py:87). Prompt: `" what does the image describe?"`.
   - Tokenizer dir `./tokenizer` must be cwd (`init_task.py:12`) — the fork ships it.
2. **Phase D (deferred)** — int8 quantization of `ofa-tiny-slim` (QUANTIZE.md
   Cells 19–23): ~35 MB full-int8 / ~49 MB int8+fp16-embed variant, dequantize-once
   loader, structural + inference verify. Then ONNX export → NCNN/ORT Mobile →
   Snapdragon 660.
3. **Housekeeping** — delete `ofa-tiny/pytorch_model.bin` (7.3 MB partial, untracked)
   and add `ofa-tiny/` to `.gitignore`. `HANDOFF.md` stays uncommitted.


