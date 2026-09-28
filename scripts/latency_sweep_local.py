"""
Local latency-vs-token-size sweep for the 0.8B full-corpus checkpoint on
Apple Silicon (MPS, eager path -- CUDA graphs are CUDA-only).

Loads RoutingDecoderModel once, then times predict_choice across prompt
token buckets (measured with the real tokenizer, not guessed). Writes
results/latency_08b_local.json + results/latency_08b_local.png.

Run: uv run python scripts/latency_sweep_local.py [--device mps]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUCKETS = [128, 256, 512, 1024, 2048, 4096]
N_WARMUP = 3
N_TIMED = 15
OPTIONS = ["refund", "replace", "escalate", "callback", "store_credit"]

FILLER = ("The customer contacted support regarding order status and delivery "
          "confirmation, requesting a full review of the shipment timeline. ")


def build_state_for_tokens(tok, target: int) -> tuple[str, int]:
    state = "The customer says the package never arrived."
    text, n = "", 0
    for _ in range(200):
        text = f"Decide the resolution. Context: {FILLER * 8}\n\nCase: {state}"
        n = len(tok(text)["input_ids"])
        if n >= target:
            break
        state = state + " " + FILLER * 20
    # Trim trailing words to land close to target (keep >= target).
    words = text.split(" ")
    while len(words) > 10:
        cand = " ".join(words[:-10])
        if len(tok(cand)["input_ids"]) < target:
            break
        words = words[:-10]
        n = len(tok(" ".join(words))["input_ids"])
    text = " ".join(words)
    return text, len(tok(text)["input_ids"])


def percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(q * len(xs))))
    return xs[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--checkpoints-dir", default=str(REPO_ROOT / "checkpoints-0.5b-fullcorpus-serve"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    import torch
    from serve.inference import RoutingDecoderModel

    print(f"loading model on device={args.device}...", flush=True)
    t0 = time.perf_counter()
    model = RoutingDecoderModel(checkpoints_dir=Path(args.checkpoints_dir),
                                device=args.device, use_cuda_graphs=False)
    print(f"loaded in {time.perf_counter() - t0:.1f}s (device={model.device})", flush=True)
    print(f"backend describe: {model.describe()}", flush=True)

    tok = model.tokenizer if hasattr(model, "tokenizer") else None
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

    rows = []
    for target in BUCKETS:
        state, actual = build_state_for_tokens(tok, target)
        for _ in range(N_WARMUP):
            model.predict_choice(state, OPTIONS)
        if args.device == "mps":
            torch.mps.synchronize()
        times = []
        for _ in range(N_TIMED):
            s = time.perf_counter()
            out = model.predict_choice(state, OPTIONS)
            if args.device == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - s) * 1000)
        row = {"target_tokens": target, "actual_tokens": actual,
               "n": N_TIMED, "p50_ms": percentile(times, 0.5),
               "p95_ms": percentile(times, 0.95),
               "mean_ms": statistics.mean(times),
               "choice": out["choice"]}
        print(f"tokens~{actual}: p50={row['p50_ms']:.1f}ms p95={row['p95_ms']:.1f}ms", flush=True)
        rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"device": model.device, "checkpoint": "ekvachan-decoder-qwen-fullcorpus (0.8B)",
               "options_n": len(OPTIONS), "rows": rows}
    (out_dir / "latency_08b_local.json").write_text(json.dumps(payload, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["actual_tokens"] for r in rows]
    p50 = [r["p50_ms"] for r in rows]
    p95 = [r["p95_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, p50, "o-", label="p50")
    ax.plot(xs, p95, "s--", label="p95")
    ax.set_xlabel("prompt tokens (measured)")
    ax.set_ylabel("latency ms (e2e, in-process)")
    ax.set_title(f"ekVachan 0.8B full-corpus latency vs prompt size ({model.device}, eager)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_08b_local.png", dpi=120)
    print(f"wrote {out_dir / 'latency_08b_local.json'} + .png", flush=True)


if __name__ == "__main__":
    main()
