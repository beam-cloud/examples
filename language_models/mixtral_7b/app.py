from beam import endpoint, Image, Volume, env

# This ensures that these packages are only loaded when the script is running remotely on Beam
if env.is_remote():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT = "mistralai/Mistral-7B-v0.1"
BEAM_VOLUME_PATH = "./cached_models"


def load_models():
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=BEAM_VOLUME_PATH,
    )
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    return model, tokenizer


@endpoint(
    secrets=["HF_TOKEN"],
    on_start=load_models,
    name="mistral-7b",
    cpu=2,
    memory="32Gi",
    gpu="A10G",
    image=Image(
        python_version="python3.11",
        python_packages=[
            "transformers==4.42.3",
            "sentencepiece==0.1.99",
            "accelerate==0.23.0",
            "torch==2.0.1",
            "huggingface_hub[hf-transfer]",
        ],
    ).with_envs("HF_HUB_ENABLE_HF_TRANSFER=1"),
    volumes=[
        Volume(
            name="cached_models",
            mount_path=BEAM_VOLUME_PATH,
        )
    ],
)
def generate(context, **inputs):
    # Retrieve model and tokenizer from on_start
    model, tokenizer = context.on_start_value

    # Inputs passed to API
    prompt = inputs.get("prompt")
    if not prompt:
        return {"error": "Please provide a prompt."}

    generate_args = {
        "max_new_tokens": inputs.get("max_new_tokens", 128),
        "temperature": inputs.get("temperature", 1.0),
        "top_p": inputs.get("top_p", 0.95),
        "top_k": inputs.get("top_k", 50),
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "use_cache": True,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
        output = model.generate(inputs=input_ids, **generate_args)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {"generated_text": generated_text}
