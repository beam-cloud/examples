from beam import endpoint, Volume, experimental, Image

VOLUME_NAME = "trained-models"
CACHE_PATH = f"./{VOLUME_NAME}-cache"


def load_latest_model():
    # Preload and return the model and tokenizer
    global model, tokenizer
    print("Loading latest...")
    model = lambda x: x + 1  # This is just example code

    s.clear()  # Clear the signal so it doesn't fire again


# Set a signal handler - when invoked, it will run the handler function
s = experimental.Signal(
    name="reload-model",
    handler=load_latest_model,
)


@endpoint(
    name="inference",
    volumes=[Volume(name=VOLUME_NAME, mount_path=CACHE_PATH)],
    image=Image().add_python_packages(["transformers", "torch"]),
    on_start=load_latest_model,
)
def predict(**inputs):
    global model, tokenizer  # These will have the latest values

    return {"success": "true"}
