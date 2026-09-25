# ekVachan HF Space (ZeroGPU). Runs the actual repo, not a vendored copy --
# clones a pinned commit so the Space always serves real, reviewed code
# (see hf_space/README.md for the update procedure).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Pin to a specific commit for reproducibility -- bump this on purpose when
# shipping a new model/serving change, not automatically on every push.
ARG EKVACHAN_COMMIT=main
RUN git clone --depth 1 https://github.com/asp616848/better-jev-for-all.git . \
    && (test "$EKVACHAN_COMMIT" = "main" || git fetch --depth 1 origin "$EKVACHAN_COMMIT" && git checkout "$EKVACHAN_COMMIT")

# Trimmed dependency set for serving + the Gradio UI -- not the full
# training/benchmark pyproject.toml (no vizdoom/scikit-learn/datasets needed
# here).
RUN pip install --no-cache-dir \
    torch \
    "transformers>=4.45" \
    "peft>=0.13" \
    pillow \
    torchvision \
    numpy \
    "huggingface-hub>=0.25" \
    gradio \
    spaces

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["python", "hf_space/app.py"]
