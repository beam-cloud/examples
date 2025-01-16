# inference.py
# Deploy to beam by running `$ beam deploy inference.py:predict --name llama-ft` in the terminal

from beam import Image, endpoint, env, Volume, QueueDepthAutoscaler, experimental

MOUNT_PATH = "./llama-ft"
FINETUNE_PATH = "./llama-ft/llama-finetuned"
MODEL_PATH = "./llama-ft/weights"

# This ensures that these packages are only loaded when the script is running remotely on Beam
if env.is_remote():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel


def load_finetuned_model():
    global model, tokenizer, stop_token_ids
    print("Loading latest...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, attn_implementation="eager", device_map="auto", is_decoder=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # using our LORA result via the PEFT library
    model = PeftModel.from_pretrained(model, FINETUNE_PATH)
    print(model.config)

    stop_token = "<|im_end|>"
    stop_token_ids = tokenizer.encode(stop_token, add_special_tokens=False)


@endpoint(
    name="llama-inference",
    on_start=load_finetuned_model,
    volumes=[Volume(name="llama-ft", mount_path=MOUNT_PATH)],
    cpu=1,
    memory="16Gi",
    # We can switch to a smaller, more cost-effective GPU for inference rather than fine-tuning
    gpu="T4",
    image=Image(python_version="python3.9").add_python_packages(
        ["transformers==4.42.0", "torch", "peft"]
    ),
    # This autoscaler spawns new containers (up to 5) if the queue depth for tasks exceeds 1
    autoscaler=QueueDepthAutoscaler(max_containers=5, tasks_per_container=1),
)
def predict(**inputs):
    global model, tokenizer, stop_token_ids  # These will have the latest values

    prompt = inputs.get("prompt", None)
    if not prompt:
        return {"error": "Please provide a prompt."}

    # Now we will format the user provided prompt so that it is of the format that
    # the fine tuning dataset established.
    prompt = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"

    # We set the end of sequence token to the last token from <|im_end|>
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        use_cache=False,
        eos_token_id=stop_token_ids[-1],
        pad_token_id=tokenizer.eos_token_id,
    )
    # Here we are trimming the input length from the output so that only the newly generated text is returned.
    text = tokenizer.decode(output[0][len(inputs[0]) :])
    print(text)

    return {"text": text}


if __name__ == "__main__":
    predict.remote()
