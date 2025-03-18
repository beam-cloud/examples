from beam import Image, endpoint, Volume, env

if env.is_remote():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

image = (
    Image(python_version="python3.11")
    .add_python_packages(
        [
            "ninja",
            "packaging",
            "wheel",
            "torch",
            "xformers",
            "trl",
            "peft",
            "accelerate",
            "bitsandbytes",
        ]
    )
    .add_commands(
        [
            "pip uninstall unsloth -y",
            'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"',
        ]
    )
)

MAX_SEQ_LENGTH = 2048
VOLUME_PATH = "./model_storage"


@endpoint(
    name="unsloth-inference",
    image=image,
    cpu=12,
    memory="32Gi",
    gpu="A100-40",
    timeout=-1,
    volumes=[Volume(name="model-storage", mount_path=VOLUME_PATH)],
)
def generate(**inputs):
    prompt = inputs.pop("prompt", None)

    if not prompt:
        return {"error": "Please provide a prompt"}

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"{VOLUME_PATH}/fine_tuned_model",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    FastLanguageModel.for_inference(model)

    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs, max_new_tokens=64, use_cache=True, temperature=1.5, min_p=0.1
    )
    res = tokenizer.batch_decode(outputs)
    return {"output": res}
