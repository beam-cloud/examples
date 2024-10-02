from beam import Image, endpoint, env

if env.is_remote():
    import tensorflow as tf


@endpoint(
    name="tensorflow-gpu",
    cpu=1,
    memory="4Gi",
    gpu="A10G",
    # Make sure to use `tensorflow[and-cuda]` in order to access GPU resources
    image=Image().add_python_packages(["tensorflow[and-cuda]"]),
)
def predict():
    # Show available GPUs
    gpus = tf.config.list_physical_devices("GPU")

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

    print("🚧 Is built with CUDA:", tf.test.is_built_with_cuda())
    print("🚧 Is GPU available:", tf.test.is_gpu_available())
    print("🚧 GPUs available:", tf.config.list_physical_devices("GPU"))
