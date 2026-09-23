"""
Capability probe for the `ekvachan-vision` plan (PRD 5.2b). Not a training
script and not a benchmark -- it answers one question with real evidence:

    does Qwen3.5-4B's tokenizer/processor/model path support an image content
    block under *this project's exact* restricted-logit LoRA setup, without
    changing the mechanism?

Every claim in PRD 5.2b's "what was actually verified" table is produced by
this file, so the next agent can re-derive them instead of trusting the prose
(dev-guidelines rule 3). Writes a manifest to `results/` (rule 10).

Run (CPU-only checks, ~1 min, no GPU needed):
    uv run python3 -u -m training.probe_vision_path

Run (adds a real bf16 forward pass on the GPU, ~10 GB VRAM, ~2 min):
    uv run python3 -u -m training.probe_vision_path --gpu-forward

Requires `pillow` and `torchvision` -- transformers 5.x makes torchvision a
hard dependency of *any* image processor, which is itself one of the findings.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen3.5-4B"
MAX_OPTIONS = 26
LETTERS = [chr(ord("A") + i) for i in range(MAX_OPTIONS)]

# The same prompt shape `train_decoder_lora_benchcorpus.build_prompt()` emits,
# with an image content block prepended. Nothing else about it changes.
VISION_INSTRUCTIONS = (
    "You are a computer-use agent. Given the screenshot of the current screen "
    "state, choose which of the following actions to take next."
)
VISION_STATE = "The browser is showing a sign-in form."
VISION_OPTIONS = ["click the login button", "scroll down", "type into the search box"]

TEXT_PROMPT = (
    "Given a premise and a hypothesis, choose the correct relationship.\n\n"
    "Premise: a dog runs.\nHypothesis: an animal moves.\n\n"
    "Options:\nA) entailment\nB) neutral\nC) contradiction\n\n"
    "Answer with a single letter (A/B/C)."
)

# Sizes that matter for this project: ViZDoom-scale frames, Atari-HEAD frames,
# a browser-use DOM screenshot, and a document page.
PROBE_SIZES = [
    (160, 210, "atari_native"),
    (320, 240, "vizdoom_scale"),
    (448, 448, "square_448"),
    (640, 480, "vga_game_frame"),
    (768, 768, "doc_page"),
    (1280, 720, "screenshot_720p"),
    (1920, 1080, "screenshot_1080p"),
]


def _fake_image(w: int, h: int):
    from PIL import Image

    rng = np.random.default_rng(0)
    return Image.fromarray((rng.random((h, w, 3)) * 255).astype("uint8"))


def build_vision_prompt_messages():
    option_lines = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(VISION_OPTIONS))
    text = (
        f"{VISION_INSTRUCTIONS}\n\n{VISION_STATE}\n\nOptions:\n{option_lines}\n\n"
        f"Answer with a single letter ({'/'.join(LETTERS[:len(VISION_OPTIONS)])})."
    )
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--gpu-forward", action="store_true", help="also load the VL model and run a real forward pass")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import transformers
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    )

    findings: dict = {
        "base_model": args.base_model,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
    }
    try:
        import torchvision

        findings["versions"]["torchvision"] = torchvision.__version__
    except ImportError:
        findings["versions"]["torchvision"] = None

    # --- 1. config: is the checkpoint actually multimodal? -----------------
    cfg = AutoConfig.from_pretrained(args.base_model)
    findings["config"] = {
        "class": type(cfg).__name__,
        "architectures": list(cfg.architectures or []),
        "has_vision_config": hasattr(cfg, "vision_config"),
        "image_token_id": getattr(cfg, "image_token_id", None),
        "video_token_id": getattr(cfg, "video_token_id", None),
        "vision_start_token_id": getattr(cfg, "vision_start_token_id", None),
        "vision_end_token_id": getattr(cfg, "vision_end_token_id", None),
    }

    # --- 2. which auto-class does this project's loader actually get? ------
    mt = cfg.model_type
    findings["auto_class_mapping"] = {
        "model_type": mt,
        "AutoModelForCausalLM": MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(mt),
        "AutoModelForImageTextToText": MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.get(mt),
    }

    # --- 3. param split: how much does the vision tower cost? -------------
    from accelerate import init_empty_weights

    with init_empty_weights():
        vl = AutoModelForImageTextToText.from_config(cfg)
        txt = AutoModelForCausalLM.from_config(cfg)
    n_vl = sum(p.numel() for p in vl.parameters())
    n_vis = sum(p.numel() for n, p in vl.named_parameters() if "visual" in n)
    n_txt = sum(p.numel() for p in txt.parameters())
    findings["params"] = {
        "vl_class": type(vl).__name__,
        "vl_total": n_vl,
        "vl_visual": n_vis,
        "visual_fraction": round(n_vis / n_vl, 4),
        "causal_lm_class": type(txt).__name__,
        "causal_lm_total": n_txt,
        "causal_lm_has_visual": any("visual" in n for n, _ in txt.named_parameters()),
    }

    # --- 4. does this project's existing LoRA config still work unchanged? -
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    with init_empty_weights():
        vl2 = AutoModelForImageTextToText.from_config(cfg)
    peft_vl = get_peft_model(vl2, lora)
    lora_mods = [n for n, _ in peft_vl.named_modules() if n.endswith("lora_A")]
    findings["lora"] = {
        "target_modules": lora.target_modules if isinstance(lora.target_modules, list) else sorted(lora.target_modules),
        "n_lora_modules": len(lora_mods),
        "any_in_vision_tower": any("visual" in n for n in lora_mods),
        "example_key": lora_mods[0] if lora_mods else None,
        "trainable_params": sum(p.numel() for p in peft_vl.parameters() if p.requires_grad),
        "note": (
            "The vision tower is frozen by name mismatch, not by an explicit freeze: its blocks "
            "use attn.qkv / mlp.linear_fc{1,2}, none of which match this project's target_modules. "
            "Adapter keys gain a `.language_model` path segment under the VL class, so a CausalLM-"
            "trained adapter does NOT load into the VL model without a key remap."
        ),
    }

    # --- 5. the chat template + processor path ---------------------------
    tok = AutoTokenizer.from_pretrained(args.base_model)
    tmpl_only = tok.apply_chat_template(
        build_vision_prompt_messages(), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    findings["chat_template"] = {
        "tokenizer_class": type(tok).__name__,
        "accepts_image_content_block": "<|image_pad|>" in tmpl_only,
        "image_pad_occurrences_in_rendered_string": tmpl_only.count("<|image_pad|>"),
        "rendered_head": tmpl_only[:160],
        "rendered_tail": tmpl_only[-40:],
    }

    proc = AutoProcessor.from_pretrained(args.base_model)
    proc.tokenizer.padding_side = "left"
    if proc.tokenizer.pad_token is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    ip = proc.image_processor
    ipd = ip.to_dict()
    merge = ipd.get("merge_size", 2)
    findings["processor"] = {
        "class": type(proc).__name__,
        "image_processor_class": type(ip).__name__,
        "patch_size": ipd.get("patch_size"),
        "merge_size": merge,
        "default_size": ipd.get("size"),
    }

    # --- 6. how many sequence tokens does an image actually cost? ---------
    token_costs = {}
    for (w, h, label) in PROBE_SIZES:
        o = ip(images=[_fake_image(w, h)], return_tensors="pt")
        g = o["image_grid_thw"][0].tolist()
        token_costs[label] = {
            "w": w, "h": h, "grid_thw": g,
            "image_tokens": int(np.prod(g)) // (merge * merge),
            "pixel_values_shape": list(o["pixel_values"].shape),
        }
    capped = {}
    for cap in (256, 128, 64):
        ip_c = AutoProcessor.from_pretrained(args.base_model, max_pixels=cap * 28 * 28).image_processor
        o = ip_c(images=[_fake_image(1920, 1080)], return_tensors="pt")
        g = o["image_grid_thw"][0].tolist()
        capped[f"max_pixels={cap}*28*28"] = {"grid_thw": g, "image_tokens_1080p": int(np.prod(g)) // (merge * merge)}
    findings["image_token_cost"] = {"uncapped": token_costs, "capped_1080p": capped}

    # --- 7. the load-bearing one: does the last-column trick survive? -----
    v_text = proc.apply_chat_template(
        build_vision_prompt_messages(), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    t_text = proc.apply_chat_template(
        [{"role": "user", "content": TEXT_PROMPT}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    img_a, img_b = _fake_image(448, 448), _fake_image(224, 336)
    b_img = proc(text=[v_text], images=[img_a], return_tensors="pt")
    b_txt = proc(text=[t_text], images=None, return_tensors="pt")
    b_mix = proc(text=[v_text, t_text], images=[img_a], return_tensors="pt", padding=True)
    b_two = proc(text=[v_text, v_text], images=[img_a, img_b], return_tensors="pt", padding=True)

    img_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    last_col_ids = b_mix["input_ids"][:, -1].tolist()
    findings["batching"] = {
        "image_row_keys": sorted(b_img.keys()),
        "text_only_row_keys": sorted(b_txt.keys()),
        "text_only_emits_pixel_values": "pixel_values" in b_txt,
        "image_pad_expanded_by_processor": int((b_img["input_ids"][0] == img_pad_id).sum()),
        "mixed_batch_works": True,
        "mixed_batch_shapes": {k: list(v.shape) for k, v in b_mix.items()},
        "two_image_batch_shapes": {k: list(v.shape) for k, v in b_two.items()},
        "mixed_last_column_ids": last_col_ids,
        "mixed_last_column_identical": len(set(last_col_ids)) == 1,
        "mixed_last_column_attention": b_mix["attention_mask"][:, -1].tolist(),
        "note": (
            "Left padding keeps the final template token in the last column for every row, image "
            "or not, so `labels[:, last_col]` masking and the `logits[:, -1, :]` restricted-logit "
            "read in train_decoder_lora_benchcorpus.py work unchanged."
        ),
    }

    # --- 8. optional: a real forward pass -------------------------------
    if args.gpu_forward:
        assert torch.cuda.is_available(), "--gpu-forward requires CUDA"
        free, total = torch.cuda.mem_get_info(0)
        if free / 1e9 < 12:
            raise RuntimeError(f"only {free/1e9:.1f} GB free -- refusing to start on a shared GPU this tight")
        t0 = time.time()
        model = AutoModelForImageTextToText.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map="cuda")
        load_s = time.time() - t0
        letter_ids = torch.tensor([tok.encode(l, add_special_tokens=False)[0] for l in LETTERS[:3]], device="cuda")
        fwd = {}
        for name, enc in (("image_row", b_img), ("text_only_row", b_txt), ("mixed_batch", b_mix)):
            enc_c = {k: v.to("cuda") for k, v in enc.items()}
            t1 = time.time()
            with torch.no_grad():
                out = model(**enc_c)
            probs = F.softmax(out.logits[:, -1, :][:, letter_ids].float(), dim=-1).cpu().numpy()
            fwd[name] = {
                "ok": True,
                "logits_shape": list(out.logits.shape),
                "forward_ms": round((time.time() - t1) * 1000, 1),
                "restricted_probs_ABC": probs.round(4).tolist(),
            }
        findings["gpu_forward"] = {
            "device_name": torch.cuda.get_device_name(0),
            "load_seconds": round(load_s, 1),
            "weights_vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "runs": fwd,
            "note": "Untrained base model -- the probabilities prove the mechanism runs, nothing about accuracy.",
        }

    print(json.dumps(findings, indent=2))
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "results" / f"vision-path-probe-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.manifest.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(findings, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
