"""OFA-Tiny image captioning inference (HF path).

Compatible with modern transformers (>= 4.36; 4.44.x verified). Requires:
  - transformers, torch, torchvision, pillow installed
  - the HF checkpoint downloaded to ./ofa-tiny (OFA-Sys/ofa-tiny):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="OFA-Sys/ofa-tiny", local_dir="ofa-tiny")
  - the two compatibility patches (already applied in this fork):
      1. `from transformers.file_utils import` -> `from transformers.utils import`
         in ofa/__init__.py and ofa/modeling_ofa.py
      2. `generation_config=None` added to the signature of
         OFAModel._prepare_encoder_decoder_kwargs_for_generation

Usage:
  python infer_caption.py <image.jpg>
  python infer_caption.py <image.jpg> --generator fairseq   # fallback path
"""

import argparse
import time

import torch
from PIL import Image
from torchvision import transforms

from ofa.tokenization_ofa import OFATokenizer
from ofa.modeling_ofa import OFAModel

CKPT = "ofa-tiny"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to image file")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--generator", choices=["hf", "fairseq"], default="hf")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    patch_resize_transform = transforms.Compose(
        [
            lambda image: image.convert("RGB"),
            transforms.Resize(
                (256, 256), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    t0 = time.time()
    tokenizer = OFATokenizer.from_pretrained(CKPT)
    model = OFAModel.from_pretrained(CKPT).to(device).eval()
    print(f"model loaded in {time.time() - t0:.1f}s")

    image = Image.open(args.image)
    patch_img = patch_resize_transform(image).unsqueeze(0).to(device)
    prompt = " what does the image describe?"
    src = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.squeeze(0)
    src = (
        torch.cat(
            [
                torch.tensor([tokenizer.bos_token_id]),
                src,
                torch.tensor([tokenizer.eos_token_id]),
            ]
        )
        .unsqueeze(0)
        .to(device)
    )
    patch_masks = torch.tensor([True]).to(device)

    t0 = time.time()
    with torch.no_grad():
        if args.generator == "hf":
            gen = model.generate(
                input_ids=src,
                patch_images=patch_img,
                patch_masks=patch_masks,
                num_beams=args.num_beams,
                max_length=args.max_length,
                min_length=1,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            caption = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        else:
            from generate.sequence_generator import SequenceGenerator

            seq_gen = SequenceGenerator(
                tokenizer=tokenizer,
                beam_size=args.num_beams,
                max_len_b=args.max_length,
                min_len=1,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            ).to(device)
            data = {
                "net_input": {
                    "input_ids": src,
                    "patch_images": patch_img,
                    "patch_masks": patch_masks,
                }
            }
            out = seq_gen.generate([model], data)
            caption = tokenizer.batch_decode(
                [out[0][0]["tokens"]], skip_special_tokens=True
            )[0]
    print(f"caption in {time.time() - t0:.1f}s")
    print("caption:", caption)


if __name__ == "__main__":
    main()
