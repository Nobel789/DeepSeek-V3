"""Multi-modal demo for image + text queries.

This script extracts visual signals (caption, OCR, or VQA) from an image
and optionally feeds them into DeepSeek-V3 for a final response.
"""

import json
import os
from argparse import ArgumentParser
from typing import List

import torch
from PIL import Image
from safetensors.torch import load_model
from transformers import AutoTokenizer, pipeline

from model import ModelArgs, Transformer
from generate import generate


DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_OCR_MODEL = "microsoft/trocr-base-printed"
DEFAULT_VQA_MODEL = "dandelin/vilt-b32-finetuned-vqa"


def _load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _resolve_vision_device(device: str) -> int:
    if device.lower() == "cpu":
        return -1
    if device.startswith("cuda") and torch.cuda.is_available():
        if ":" in device:
            _, index = device.split(":", 1)
            return int(index)
        return 0
    return -1


def build_vision_context(
    image_path: str,
    question: str,
    mode: str,
    caption_model: str,
    ocr_model: str,
    vqa_model: str,
    device: int,
) -> List[str]:
    image = _load_image(image_path)
    context_lines: List[str] = []

    if mode in {"caption", "all"}:
        captioner = pipeline("image-to-text", model=caption_model, device=device)
        caption_output = captioner(image, max_new_tokens=64)[0]["generated_text"]
        context_lines.append(f"Image caption: {caption_output}")

    if mode in {"ocr", "all"}:
        ocr_engine = pipeline("image-to-text", model=ocr_model, device=device)
        ocr_output = ocr_engine(image, max_new_tokens=256)[0]["generated_text"]
        if ocr_output.strip():
            context_lines.append(f"Image OCR: {ocr_output}")

    if mode in {"vqa", "all"}:
        if not question:
            raise ValueError("A question is required for visual Q&A mode.")
        vqa_engine = pipeline("visual-question-answering", model=vqa_model, device=device)
        vqa_output = vqa_engine(image, question=question)[0]["answer"]
        context_lines.append(f"Visual Q&A answer: {vqa_output}")

    return context_lines


def load_text_model(ckpt_path: str, config: str, device: str) -> Transformer:
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(device):
        model = Transformer(args)
    load_model(model, os.path.join(ckpt_path, "model0-mp1.safetensors"))
    return model


def answer_with_text_model(
    model: Transformer,
    tokenizer: AutoTokenizer,
    context_lines: List[str],
    question: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt_parts = []
    if context_lines:
        prompt_parts.append("\n".join(context_lines))
    if question:
        prompt_parts.append(f"User question: {question}")
    else:
        prompt_parts.append("Describe the image and summarize any visible text.")
    prompt = "\n\n".join(prompt_parts)
    messages = [{"role": "user", "content": prompt}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    completion_tokens = generate(
        model,
        [prompt_tokens],
        max_new_tokens,
        tokenizer.eos_token_id,
        temperature,
    )
    return tokenizer.decode(completion_tokens[0], skip_special_tokens=True)


def main() -> None:
    parser = ArgumentParser(description="Multi-modal image + text demo.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--question", default="", help="Question to ask about the image.")
    parser.add_argument(
        "--mode",
        choices=["caption", "ocr", "vqa", "all"],
        default="all",
        help="Vision extraction mode.",
    )
    parser.add_argument("--caption-model", default=DEFAULT_CAPTION_MODEL)
    parser.add_argument("--ocr-model", default=DEFAULT_OCR_MODEL)
    parser.add_argument("--vqa-model", default=DEFAULT_VQA_MODEL)
    parser.add_argument("--device", default="cuda", help="Device for vision pipelines.")
    parser.add_argument("--ckpt-path", default="", help="Text model checkpoint path.")
    parser.add_argument("--config", default="", help="Text model config path.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    vision_device = _resolve_vision_device(args.device)
    context_lines = build_vision_context(
        image_path=args.image,
        question=args.question,
        mode=args.mode,
        caption_model=args.caption_model,
        ocr_model=args.ocr_model,
        vqa_model=args.vqa_model,
        device=vision_device,
    )

    print("\n".join(context_lines))
    if not args.ckpt_path or not args.config:
        return

    model = load_text_model(args.ckpt_path, args.config, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    answer = answer_with_text_model(
        model,
        tokenizer,
        context_lines,
        args.question,
        args.max_new_tokens,
        args.temperature,
    )
    print("\nDeepSeek-V3 Response:\n")
    print(answer)


if __name__ == "__main__":
    main()
