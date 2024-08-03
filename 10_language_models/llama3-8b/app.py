"""
### Meta Llama 3 8B Instruct ###

Note: This is a gated Huggingface model and you must request access to it here:

https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

Retrieve your HF token from this page: https://huggingface.co/settings/tokens

After your access is granted, make sure to save your Huggingface token on Beam:

`$ beam secret create HF_TOKEN`

...and add the secret to your Beam function decorator:

@endpoint(secrets=["HF_TOKEN"])

After the endpoint is deployed, you can call it like this:

```
curl -X POST 'https://app.beam.cloud/endpoint/id/[ENDPOINT-ID]' \
-H 'Connection: keep-alive' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer [AUTH-TOKEN]' \
-d '{
    "messages": [
        {"role": "system", "content": "You are a yoda chatbot who always responds in yoda speak!"},
        {"role": "user", "content": "Who are you?"}
    ]
}'
```
"""

from beam import endpoint, Image, Volume, env

# This ensures that these packages are only loaded when the script is running remotely on Beam
if env.is_remote():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

# Model parameters
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_LENGTH = 512
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 40
REPETITION_PENALTY = 1.0
NO_REPEAT_NGRAM_SIZE = 0
DO_SAMPLE = True

BEAM_VOLUME_PATH = "./cached_models"


# This runs once when the container first starts
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return model, tokenizer


@endpoint(
    secrets=["HF_TOKEN"],
    on_start=load_models,
    name="meta-llama-3-8b-instruct",
    cpu=2,
    memory="32Gi",
    gpu="A100-40",
    image=Image(
        python_version="python3.9",
        python_packages=["torch", "transformers", "accelerate"],
    ),
    volumes=[
        Volume(
            name="cached_models",
            mount_path=BEAM_VOLUME_PATH,
        )
    ],
)
def generate_text(context, **inputs):
    # Retrieve model and tokenizer from on_start
    model, tokenizer = context.on_start_value

    # Inputs passed to API
    messages = inputs.pop("messages", None)
    if not messages:
        return {"error": "Please provide messages for text generation."}

    generate_args = {
        "max_length": inputs.get("max_tokens", MAX_LENGTH),
        "temperature": inputs.get("temperature", TEMPERATURE),
        "top_p": inputs.get("top_p", TOP_P),
        "top_k": inputs.get("top_k", TOP_K),
        "repetition_penalty": inputs.get("repetition_penalty", REPETITION_PENALTY),
        "no_repeat_ngram_size": inputs.get(
            "no_repeat_ngram_size", NO_REPEAT_NGRAM_SIZE
        ),
        "do_sample": inputs.get("do_sample", DO_SAMPLE),
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    model_inputs = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(model_inputs, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **generate_args
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"output": output_text}
