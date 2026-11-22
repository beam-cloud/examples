"""
Qwen 27B behind an OpenAI-compatible API, served by vLLM on one B200.

Deploy:

    beam secret create VLLM_API_KEY "$(openssl rand -hex 24)"
    beam deploy app.py:qwen --json
    beam deployment scale <deployment_id> --replicas 1

To serve your own fine-tune, change MODEL to a Hugging Face repo (add
"HF_TOKEN" to `secrets` if it is private) or to a directory on the volume,
such as "/models/my-finetune". See the README.
"""

from beam import Image, Pod, Volume

MODEL = "Qwen/Qwen3.8-27B"

# The name clients pass as `model`. Keeping it fixed lets you swap the
# weights behind it without changing client code.
SERVED_MODEL_NAME = "qwen-27b"

# Weights are downloaded here once, then reused by every later container.
MODELS_PATH = "/models"

qwen = Pod(
    name="qwen-27b",
    image=Image(base_image="vllm/vllm-openai:v0.34.0"),
    gpu="B200",  # or "B300"
    cpu=16,
    # Enough RAM to hold the whole checkpoint in the page cache while it loads.
    memory="120Gi",
    ports=[8000],
    volumes=[Volume(name="qwen-27b-weights", mount_path=MODELS_PATH)],
    # vLLM reads VLLM_API_KEY and requires it as the Bearer token on /v1.
    secrets=["VLLM_API_KEY"],
    env={"HF_HOME": f"{MODELS_PATH}/hf"},
    # How long an idle container stays up when the deployment has no fixed
    # replica count. See the README: set one with `beam deployment scale`.
    keep_warm_seconds=300,
    entrypoint=[
        "vllm",
        "serve",
        MODEL,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        # Read the checkpoint from the volume into RAM before loading it.
        "--safetensors-load-strategy",
        "prefetch",
        # Requests beyond this many wait in vLLM's queue.
        "--max-num-seqs",
        "256",
        # Return thinking in `reasoning_content`, separate from the answer.
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
    ],
)
