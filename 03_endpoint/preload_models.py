"""
** Preload Models **

Beam includes an `on_start` lifecycle hook which is useful for running operations
that only need to happen once when a container first starts.

As an example, loading model weights is an operation that only needs to happen once.

In the code below, a `download_models()` function is attached to `on_start`.

The `download_models` code runs exactly once when the container first starts. 
"""

from beam import Image, endpoint, Volume


CACHE_PATH = "./weights"


def download_models():
    from transformers import AutoTokenizer, OPTForCausalLM

    model = OPTForCausalLM.from_pretrained("facebook/opt-125m", cache_dir=CACHE_PATH)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", cache_dir=CACHE_PATH)

    return model, tokenizer


@endpoint(
    on_start=download_models,
    volumes=[Volume(name="weights", mount_path=CACHE_PATH)],
    cpu=1,
    memory="16Gi",
    gpu="T4",
    image=Image(
        python_version="python3.8",
        python_packages=[
            "transformers",
            "torch",
        ],
    ),
)
def predict(context, prompt):
    # Retrieve cached model from on_start function
    model, tokenizer = context.on_start_value

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(result)

    return {"prediction": result}
