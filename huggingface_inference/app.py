from beam import Image, endpoint, env, Volume, QueueDepthAutoscaler

# Path to cache model weights
BEAM_VOLUME_CACHE_PATH = "./weights"

# This ensures that these packages are only loaded when the script is running remotely on Beam
if env.is_remote():
    from transformers import AutoTokenizer, OPTForCausalLM
    import torch


# Function to download and cache models
def download_models():
    model = OPTForCausalLM.from_pretrained(
        "facebook/opt-125m", cache_dir=BEAM_VOLUME_CACHE_PATH
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/opt-125m", cache_dir=BEAM_VOLUME_CACHE_PATH
    )
    return model, tokenizer


@endpoint(
    name="inference-quickstart",
    on_start=download_models,
    volumes=[Volume(name="weights", mount_path=BEAM_VOLUME_CACHE_PATH)],
    cpu=1,
    memory="16Gi",
    gpu="T4",
    image=Image(python_version="python3.9")
    .add_python_packages(["transformers", "torch", "huggingface_hub[hf-transfer]"])
    .with_envs("HF_HUB_ENABLE_HF_TRANSFER=1"),
    autoscaler=QueueDepthAutoscaler(max_containers=5, tasks_per_container=1),
)
def predict(context, **inputs):
    # Retrieve cached model and tokenizer from on_start function
    model, tokenizer = context.on_start_value

    prompt = inputs.get("prompt", None)
    if not prompt:
        return {"error": "Please provide a prompt."}

    # Generate inference
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(result)

    return {"text": result}
