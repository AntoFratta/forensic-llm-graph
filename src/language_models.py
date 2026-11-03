"""CUDA model loading, raw-text perplexity and bounded entity generation.

Imports are deferred so passage preparation and offline tests do not require
PyTorch, Transformers, model weights or a GPU.
"""

import gc
import math
import os
import time

from src.llm_filter import parse_entities, review_messages


# This runner uses PyTorch exclusively; do not initialize an unrelated TF backend.
os.environ.setdefault("USE_TF", "0")


def require_cuda():
    """Require a CUDA runtime; CPU offloading is not part of this experiment."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before running the models.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Select the T4 runtime on Colab.")
    return torch


def load_model(settings: dict, quantized: bool):
    """Load a fixed model revision onto one GPU with FP16 computation.

    NF4 is used only for Qwen's weights. Minerva remains unquantized. Both
    tokenizers load from the same revision as their model. HF_TOKEN is read by
    Hugging Face from the environment and is never included in result metadata.
    """
    torch = require_cuda()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(settings["model"], revision=settings["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    kwargs = {"revision": settings["revision"], "dtype": torch.float16,
              "device_map": {"": 0}, "low_cpu_mem_usage": True,
              "attn_implementation": "sdpa", "use_safetensors": True}
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(settings["model"], **kwargs)
    model.eval()
    metadata = {"model": settings["model"], "revision": settings["revision"],
                "load_seconds": time.perf_counter() - started,
                "weight_memory_bytes": model.get_memory_footprint(),
                "compute_dtype": "float16", "quantization": "nf4_double" if quantized else None}
    return model, tokenizer, metadata


def perplexity_batches(passages: list, settings: dict, model, tokenizer):
    """Yield per-passage losses, excluding padding and the unpredicted first token.

    Each original passage is scored without chat prompts or generated text.
    Token limits are checked before inference. Oversized passages are reported,
    never silently truncated. Per-example losses avoid a padding-length bias.
    """
    torch = require_cuda()
    import torch.nn.functional as functional

    maximum = settings["max_tokens"]
    for offset in range(0, len(passages), settings["batch_size"]):
        batch = passages[offset:offset + settings["batch_size"]]
        encodings = tokenizer([passage["text"] for passage in batch], add_special_tokens=True,
                              padding=True, truncation=False, return_tensors="pt")
        lengths = encodings["attention_mask"].sum(dim=1).tolist()
        accepted = []
        results = {}
        for index, (passage, length) in enumerate(zip(batch, lengths, strict=True)):
            if not 2 <= length <= maximum:
                results[passage["id"]] = {"status": "token_limit" if length > maximum else "too_short",
                                           "token_count": length, "mean_nll": None, "perplexity": None}
            else:
                accepted.append((index, passage, length))
        if accepted:
            indices = [index for index, _, _ in accepted]
            width = max(length for _, _, length in accepted)
            inputs = {name: values[indices, :width].to(model.device) for name, values in encodings.items()}
            started = time.perf_counter()
            with torch.inference_mode():
                logits = model(**inputs, use_cache=False).logits
                targets = inputs["input_ids"][:, 1:].clone()
                targets.masked_fill_(inputs["attention_mask"][:, 1:] == 0, -100)
                losses = functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    targets.reshape(-1), ignore_index=-100, reduction="none",
                ).reshape(targets.shape)
                counts = (targets != -100).sum(dim=1)
                means = (losses.sum(dim=1) / counts).tolist()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            for (_, passage, length), loss in zip(accepted, means, strict=True):
                finite = math.isfinite(loss)
                value = math.exp(loss) if finite and loss < 700 else None
                results[passage["id"]] = {
                    "status": "ok" if value is not None else "nonfinite_loss",
                    "token_count": length, "scored_tokens": length - 1,
                    "mean_nll": loss if finite else None, "perplexity": value,
                    "batch_inference_seconds": elapsed,
                }
            del inputs, logits, targets, losses, counts
        yield results


def review_passage(passage: dict, settings: dict, model, tokenizer) -> dict:
    """Generate one bounded JSON response and preserve failures for inspection."""
    torch = require_cuda()
    prompt = tokenizer.apply_chat_template(review_messages(passage), tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    length = inputs["input_ids"].shape[1]
    if length > settings["max_input_tokens"]:
        return {"status": "token_limit", "input_tokens": length, "entities": [],
                "error": "Prompt exceeds max_input_tokens; first-pass candidates retained."}
    inputs = inputs.to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, do_sample=False, max_new_tokens=settings["max_new_tokens"],
            temperature=None, top_p=None, top_k=None,
            pad_token_id=tokenizer.pad_token_id, use_cache=True,
        )
    torch.cuda.synchronize()
    generated = output[0, length:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    truncated = len(generated) >= settings["max_new_tokens"] and int(generated[-1]) not in eos_ids
    result = {"status": "truncated" if truncated else "ok", "raw_response": response,
              "input_tokens": length, "output_tokens": len(generated),
              "inference_seconds": time.perf_counter() - started, "entities": []}
    if not truncated:
        try:
            result["entities"] = parse_entities(response, passage, settings["model"])
        except ValueError as exc:
            result["status"] = "invalid_response"
            result["error"] = str(exc)
    return result


def release_cuda_memory() -> None:
    """Release cached allocations after the caller deletes model references."""
    gc.collect()
    torch = require_cuda()
    torch.cuda.empty_cache()
