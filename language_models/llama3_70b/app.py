import logging
import os
from logging import getLogger

from beam import Image, Volume, endpoint
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
LOGGER = getLogger(__name__)

CACHE_PATH = "./model-weights"
model_name = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"


class PredictionRequest(BaseModel):
    prompt: str
    max_length: int


def download_models():
    from huggingface_hub import login
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    login(
        os.environ["HF_TOKEN"]
    )  # Add your HF API key to Beam using `beam secret create` CLI command

    LOGGER.info(f"Loading model: {model_name}")
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)
    model = LLM(
        model=model_name,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.80,
        max_model_len=4096,
        trust_remote_code=True,
        max_num_seqs=16,
        quantization="awq_marlin",
        download_dir=CACHE_PATH,
        cpu_offload_gb=20,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_PATH)

    LOGGER.info("Model and tokenizer loaded successfully.")
    return model, tokenizer, sampling_params


@endpoint(
    name="llama70b",
    secrets=["HF_TOKEN"],
    keep_warm_seconds=30,
    on_start=download_models,
    volumes=[Volume(name="model-weights", mount_path=CACHE_PATH)],
    cpu="8000m",
    memory="32Gi",
    gpu="A100-40",
    image=Image(
        python_version="python3.10",
        python_packages=["vllm==0.5.4"],
        commands=[],
    ),
    timeout=3600,
)
def generate(context, **inputs):
    # Unpack the values returned by on_start
    model, tokenizer, sampling_params = context.on_start_value

    # Prompt passed to API
    prompt = inputs.get("prompt", "How do I bake a chocolate cake?")
    LOGGER.info(f"Received prompt: {prompt}")

    # Inference
    try:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        output = model.generate(formatted_prompt, sampling_params)
        generated_text = output[0].outputs[0].text
        LOGGER.info(f"Generated text: {generated_text}")
        return {"text": generated_text}
    except Exception as e:
        LOGGER.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}
