"""
ekVachan Hugging Face Space demo -- Gradio SDK (no Docker), built to the
explicit cost-minimization brief given for this deployment:

  1. No Docker -- plain `gradio` SDK Space (see README.md's front matter),
     requirements.txt for deps. Source files this app imports (serve/,
     benchmarks/common/, training/decoder_lora_lib.py, eval/) are vendored
     into this Space repo directly by scripts/publish_to_huggingface.py --
     no `git clone` anywhere in this file or at request time.
  2. @spaces.GPU wraps ONLY the actual model call (`_run_inference` below),
     never the Gradio callback that does input parsing/validation/base64
     encoding -- that all happens in `predict()`, outside the decorator.
  3. The model is loaded ONCE at import time (module scope, i.e. at Space
     cold start, never per-request) -- but constructed with device="cpu"
     deliberately: moving a model onto an actual GPU device has to happen
     inside a real GPU context, and the only place this Space is
     guaranteed to have one is inside `@spaces.GPU`. So `_run_inference`
     moves it to "cuda" exactly ONCE, on the first real call, guarded by a
     module-level flag -- never re-loaded, never re-moved, on every
     subsequent request. This is safe and correct whether the Space ends
     up running on true ZeroGPU (ephemeral GPU attachment per call, where
     `spaces.GPU` is not a no-op) or on dedicated hardware like a T4 (where
     `spaces.GPU` is a documented no-op passthrough and the GPU is simply
     always there) -- see the handoff notes for why this repo's hardware
     tier is still an open question at the time this file was written.
  4. No torch.compile, no CUDA graphs (EKVACHAN_USE_CUDA_GRAPHS=0, forced
     below) -- both deliberately off for this Space specifically: CUDA
     graphs were only ever validated (PRD.md 13a.29-13a.34) on this
     project's own training-server GPU, never on a T4 (Turing, limited/no
     native bf16 tensor-core support) or under ZeroGPU's ephemeral
     GPU-attachment model, where a captured graph's validity across calls
     is an open, unverified question this session has no way to test.
     Simplicity over an unverified ~15% win, for a cost-minimized demo.
  5. No background process, no polling loop, no persistent worker thread --
     this file defines a Gradio app and nothing else runs on its own.
"""

import base64
import os
import time
from io import BytesIO

import gradio as gr
import spaces
from huggingface_hub import snapshot_download

MODEL_REPO = os.environ.get("EKVACHAN_HF_MODEL_REPO", "abhi6168/ekvachan-decoder")
CHECKPOINTS_DIR = "checkpoints"

# Model *weights* (LoRA adapters, a few hundred MB, not the base model) --
# downloaded once at cold start, outside any @spaces.GPU function. This is
# a "model download" (requirement 5 names this explicitly) and belongs
# exactly here: module scope, not inside the decorated inference function.
snapshot_download(repo_id=MODEL_REPO, local_dir=CHECKPOINTS_DIR)

# Same production adapter-routing default as serve/server.py (PRD.md
# 13a.20/13a.33): the "vision" slot holds the strongest (stage3) checkpoint
# and answers text-only requests too. CUDA graphs forced off -- see the
# module docstring's point 4.
os.environ.setdefault("EKVACHAN_TEXT_ADAPTER", "vision")
os.environ["EKVACHAN_USE_CUDA_GRAPHS"] = "0"

from serve.inference import RoutingDecoderModel  # noqa: E402 -- after env vars are set

# Loaded once, on CPU, at import time -- the expensive part (reading
# safetensors off disk, constructing both LoRA adapters, building the code
# table) happens here, exactly once, never inside a request or inside
# @spaces.GPU. device="cpu" is deliberate: see module docstring point 3.
_model = RoutingDecoderModel(checkpoints_dir=CHECKPOINTS_DIR, device="cpu")
_model_on_gpu = False


@spaces.GPU(duration=30)
def _run_inference(state: str, options: list[str], image_b64: str | None) -> dict:
    """The ONLY function in this file that touches the GPU. Moves the
    already-loaded model onto cuda exactly once (first call only, guarded
    by `_model_on_gpu`), then runs one real forward pass. Nothing else --
    no parsing, no formatting, no base64 work -- happens in here, to keep
    GPU-attached time (what actually gets billed/quota-metered) as close
    to "just the forward pass" as possible (requirements 14/15)."""
    global _model_on_gpu
    if not _model_on_gpu:
        _model.model.to("cuda")
        _model.device = "cuda"
        _model_on_gpu = True
    return _model.predict_choice(state, options, image_b64=image_b64)


def predict(state: str, options_text: str, image):
    """Gradio callback -- all CPU-only work (validation, parsing, base64
    encoding, error formatting) lives here, outside @spaces.GPU, per
    requirement 14. Only the single call to `_run_inference` below ever
    touches the GPU."""
    if not state or not state.strip():
        raise gr.Error("Enter a state / question.")
    options = [o.strip() for o in options_text.split(",") if o.strip()]
    if len(options) < 2:
        raise gr.Error("Enter at least 2 comma-separated options.")

    image_b64 = None
    if image is not None:
        buf = BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    t0 = time.time()
    try:
        result = _run_inference(state, options, image_b64)
    except Exception as e:
        raise gr.Error(str(e))
    elapsed_ms = (time.time() - t0) * 1000

    return (
        result["choice"],
        {opt: float(p) for opt, p in result["probabilities"].items()},
        f"{elapsed_ms:.0f}ms this call (includes any queueing/allocation overhead -- "
        f"not our benchmarked ~112ms p50 on dedicated hardware, see PRD.md 13a.33/13a.34)",
    )


with gr.Blocks(title="ekVachan") as demo:
    gr.Markdown(
        "# ekVachan\n"
        "Open, self-hostable alternative to TypeSafe's Jev System-One decision "
        "model. Give it a `state` and a set of `options`; it returns a "
        "calibrated probability distribution over them in one forward pass "
        "(restricted-logit read, no autoregressive generation).\n\n"
        "[GitHub](https://github.com/asp616848/better-jev-for-all) -- real, "
        "verified accuracy/latency numbers (JevBench, jabr-v2, ViZDoom, our "
        "own corpus) live in the README and PRD.md there, measured against "
        "our own dedicated server, not necessarily this Space's hardware."
    )
    with gr.Row():
        with gr.Column():
            state_in = gr.Textbox(
                label="State",
                placeholder="The customer says the package never arrived.",
                lines=3,
            )
            options_in = gr.Textbox(
                label="Options (comma-separated, 2-26)",
                placeholder="refund, replace, escalate",
            )
            image_in = gr.Image(label="Optional image", type="pil")
            btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            choice_out = gr.Textbox(label="Choice")
            probs_out = gr.Label(label="Probabilities")
            latency_out = gr.Textbox(label="Latency (this call, see caveat above)")

    btn.click(predict, inputs=[state_in, options_in, image_in], outputs=[choice_out, probs_out, latency_out])
    gr.Examples(
        examples=[
            ["The customer says the package never arrived.", "refund, replace, escalate", None],
            ["A cat sitting on a windowsill in the sun.", "cat, dog, bird", None],
        ],
        inputs=[state_in, options_in, image_in],
    )

if __name__ == "__main__":
    demo.launch()
