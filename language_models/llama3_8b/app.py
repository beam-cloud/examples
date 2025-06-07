from beam import endpoint, Image, Volume, env

# This ensures that these packages are only loaded when the script is running remotely on Beam
if env.is_remote():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

# Model parameters
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_LENGTH = 512
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.05
NO_REPEAT_NGRAM_SIZE = 2
DO_SAMPLE = True 
NUM_BEAMS = 1
EARLY_STOPPING = True

BEAM_VOLUME_PATH = "./cached_models"


# This runs once when the container first starts
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        cache_dir=BEAM_VOLUME_PATH,
        padding_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        cache_dir=BEAM_VOLUME_PATH,
        use_cache=True,
        low_cpu_mem_usage=True
    )
    model.eval()
    return model, tokenizer


image = (
    Image(python_version="python3.9")
    .add_python_packages(
        [
            "torch",
            "transformers",
            "accelerate",
            "huggingface_hub[hf-transfer]",
        ]
    )
    .with_envs({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "0",
    })
)


@endpoint(
    secrets=["HF_TOKEN"],
    on_start=load_models,
    name="meta-llama-3.1-8b-instruct",
    cpu=2,
    memory="16Gi",
    gpu="A10G",
    volumes=[
        Volume(
            name="cached_models",
            mount_path=BEAM_VOLUME_PATH,
        )
    ],
    image=image,
)
def generate_text(context, **inputs):
    # Retrieve model and tokenizer from on_start
    model, tokenizer = context.on_start_value

    # Inputs passed to API
    messages = inputs.pop("messages", None)
    if not messages:
        return {"error": "Please provide messages for text generation."}

    generate_args = {
        "max_new_tokens": inputs.get("max_tokens", MAX_LENGTH),
        "temperature": inputs.get("temperature", TEMPERATURE),
        "top_p": inputs.get("top_p", TOP_P),
        "top_k": inputs.get("top_k", TOP_K),
        "repetition_penalty": inputs.get("repetition_penalty", REPETITION_PENALTY),
        "no_repeat_ngram_size": inputs.get(
            "no_repeat_ngram_size", NO_REPEAT_NGRAM_SIZE
        ),
        "num_beams": inputs.get("num_beams", NUM_BEAMS),
        "early_stopping": inputs.get("early_stopping", EARLY_STOPPING),
        "do_sample": inputs.get("do_sample", DO_SAMPLE),
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    model_inputs_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    tokenized_inputs = tokenizer(
        model_inputs_str, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=2048
    )
    input_ids = tokenized_inputs["input_ids"].to("cuda")
    attention_mask = tokenized_inputs["attention_mask"].to("cuda")
    input_ids_length = input_ids.shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            **generate_args
        )
        new_tokens = outputs[0][input_ids_length:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {"output": output_text}
