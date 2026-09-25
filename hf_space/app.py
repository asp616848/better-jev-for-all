"""
ekVachan Hugging Face Space demo.

Thin Gradio UI directly over `serve.inference.RoutingDecoderModel` -- the
exact class our own `/v1/systemone` server runs in production (PRD.md
13a.29-13a.34), not a reimplementation. Runs on ZeroGPU: the actual forward
pass happens inside the `@spaces.GPU`-decorated function, on a shared GPU
slice allocated per call.

**Honest latency caveat, stated in the UI too, not just here**: ZeroGPU adds
its own queueing/allocation/cold-start overhead on top of the model's own
forward-pass time. The number this demo reports is real (measured around
the actual `predict_choice` call), but it is NOT the ~112ms p50 / ~131ms p95
figure in the README/PRD -- that number comes from our own dedicated
server with a warm, persistently-loaded model (PRD.md 13a.33/13a.34). This
demo exists to let people try the model, not to reproduce the benchmark.
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

# Pulled once at Space startup (cold start), not per-request. The repo must
# lay out its files as <MODEL_REPO>/ekvachan-decoder-qwen-benchcorpus/{...}
# and .../ekvachan-decoder-qwen-vision/{...}, each with its own
# manifest.json + adapter weights -- the same shape
# RoutingDecoderModel.__init__ already expects locally (see
# serve/inference.py's TEXT_CHECKPOINT_NAME/VISION_CHECKPOINT_NAME).
snapshot_download(repo_id=MODEL_REPO, local_dir=CHECKPOINTS_DIR)

# Same production defaults as serve/server.py: the "vision" slot holds the
# strongest (stage3) checkpoint and answers text-only requests too
# (PRD.md 13a.20/13a.33), CUDA graphs on (PRD.md 13a.34).
os.environ.setdefault("EKVACHAN_TEXT_ADAPTER", "vision")
os.environ.setdefault("EKVACHAN_USE_CUDA_GRAPHS", "1")

from serve.inference import RoutingDecoderModel  # noqa: E402 -- after env vars are set

_model = None


def get_model():
    global _model
    if _model is None:
        _model = RoutingDecoderModel(checkpoints_dir=CHECKPOINTS_DIR)
    return _model


@spaces.GPU(duration=30)
def predict(state: str, options_text: str, image):
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

    model = get_model()
    t0 = time.time()
    try:
        result = model.predict_choice(state, options, image_b64=image_b64)
    except Exception as e:
        raise gr.Error(str(e))
    elapsed_ms = (time.time() - t0) * 1000

    return (
        result["choice"],
        {opt: float(p) for opt, p in result["probabilities"].items()},
        f"{elapsed_ms:.0f}ms this call (ZeroGPU queue+cold-start included -- "
        f"not our benchmarked ~112ms p50, see PRD.md 13a.33/13a.34)",
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
        "our own dedicated server, not this Space."
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
