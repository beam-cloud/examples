from beam import Image, endpoint, env, Volume, QueueDepthAutoscaler, experimental

# Path to cache model weights
VOLUME_PATH = "./gemma-ft"
FT_PATH = "./gemma-ft/gemma-2b-finetuned"
MODEL_PATH = "./gemma-ft/weights"

# This ensures that these packages are only loaded when the script is running remotely on Beam
if env.is_remote():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel


def load_finetuned_model():
    global model, tokenizer, stop_token_ids
    print("Loading latest...")

    # loading the model here
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, attn_implementation="eager", device_map="auto", is_decoder=True
    )

    # loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # using PEFT to load the LORA addition
    model = PeftModel.from_pretrained(model, FT_PATH)
    print(model.config)

    stop_token = "<|im_end|>"
    stop_token_ids = tokenizer.encode(stop_token, add_special_tokens=False)

    s.clear()  # Clear the signal so it doesn't fire again


s = experimental.Signal(
    name="reload-model",
    handler=load_finetuned_model,
)


@endpoint(
    name="gemma-inference",
    on_start=load_finetuned_model,
    volumes=[Volume(name="gemma-ft", mount_path=VOLUME_PATH)],
    cpu=1,
    memory="16Gi",
    gpu="T4",
    image=Image(
        python_version="python3.9",
        python_packages=["transformers==4.42.0", "torch", "peft"],
    ),
    autoscaler=QueueDepthAutoscaler(max_containers=5, tasks_per_container=1),
)
def predict(**inputs):
    global model, tokenizer, stop_token_ids  # These will have the latest values

    prompt = inputs.get("prompt", None)
    if not prompt:
        return {"error": "Please provide a prompt."}

    # now we will format the user provided prompt so that it is of the format that
    # the fine tuning dataset expects
    prompt = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"

    # we set the end of sequence token to the last token from <|im_end|>
    # could probably use stopping criteria to check the full stop matches if  you want to.
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        use_cache=False,
        eos_token_id=stop_token_ids[-1],
        pad_token_id=tokenizer.eos_token_id,
    )
    # note that here we are trimming the input length from the output so that
    # only the newly generated text is returned
    text = tokenizer.decode(output[0][len(inputs[0]) :])
    print(text)

    # note that the stop sequence is in the output. in a realistic scenario you would
    # probably have some service that is tracking the conversation and would remove
    # the stop sequence from the output before returning it to the user. it might also
    # be storing the conversation history so that the model can continue the conversation
    # (could potentially be a nice usecase for Maps if you want long conversations without
    # having the front end send the entire conversation history each time; assuming you
    # dont need long term durability of the conversation history)
    return {"text": text}


if __name__ == "__main__":
    predict.remote()
