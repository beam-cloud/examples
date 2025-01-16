"""
In inference.py we will define the inference endpoint for our model. This endpoint
will load the latest model weights and tokenizer from the volume and then
use the model to generate a response to the user provided prompt.

Note that the stop sequence is in the output. In a realistic scenario you would
probably have some service that is tracking the conversation and would remove
the stop sequence from the output before returning it to the user. It might also
be storing the conversation history so that the model can continue the conversation.
This could potentially be a nice use case for Maps (https://docs.beam.cloud/v2/getting-started/sdk#map-2) 
if you want long conversations without having the front end send the entire 
conversation history each time. Be careful, Maps are not long-term
persistent and the data will eventually be lost.
"""

from beam import Image, endpoint, env, Volume, QueueDepthAutoscaler, experimental

MOUNT_PATH = "./gemma-ft"
FINETUNE_PATH = "./gemma-ft/gemma-2b-finetuned"
MODEL_PATH = "./gemma-ft/weights"

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

    s.clear()  # Clear the signal so it doesn't fire again


# We are using the experimental Signal abstraction, which lets us hot reload the latest weights
# without having to restart the inference service. It can be triggered with: experimental.Signal(name="reload-model")
s = experimental.Signal(
    name="reload-model",
    handler=load_finetuned_model,
)


@endpoint(
    name="gemma-inference",
    on_start=load_finetuned_model,
    volumes=[Volume(name="gemma-ft", mount_path=MOUNT_PATH)],
    cpu=1,
    memory="16Gi",
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
