"""
*** Invoking functions in other apps ***

This example demonstrates how to invoke functions in other apps on Beam.
Specifically, we cover the scenario with an inference and a retraining function

The retraining function needs a way to tell the inference function to use the latest model.

We use a `experimental.Signal()`, which is a special type of event listener that can be triggered from the retrain function.

To test this, open two terminal windows:

In window 1, serve and invoke the inference function
In window 2, serve and invoke the retrain function

Look at the logs in window 1 -- you'll notice that the signal has fired, and load_latest_model ran again
"""

from beam import endpoint, Volume, experimental, Image

VOLUME_NAME = "brand_classifier"
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
    image=Image(python_packages=["transformers", "torch"]),
    on_start=load_latest_model,
)
def predict(**inputs):
    global model, tokenizer  # These will have the latest values

    return {"success": "true"}
