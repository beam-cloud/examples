from beam import Image, endpoint, Volume


def init_model():
    from paddleocr import PaddleOCR

    import subprocess

    print(subprocess.check_output(["nvidia-smi"], shell=True))

    # Initialize PaddleOCR with local model paths
    return PaddleOCR(
        use_gpu=True,
        lang="en",
    )


@endpoint(
    on_start=init_model,
    cpu=1,
    memory="1Gi",
    gpu="T4",
    image=Image(
        python_version="python3.8",
        commands="apt-get update && \
            apt-get install -y \
            python3 \
            python3-pip \
            python3-dev \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libgl1-mesa-glx \
            libglib2.0-0 \
            && rm -rf /var/lib/apt/lists/*",
        base_image="nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04",  # Use CUDA 11.7 with cuDNN 8
        python_packages=[
            "paddlepaddle-gpu==2.4.0rc0",  # Specify the correct version for CUDA 11.7
            "paddleocr",
        ],
    ),
)
def predict(context):
    import paddle

    is_cuda_available = paddle.device.is_compiled_with_cuda()
    print("Is PaddlePaddle compiled with CUDA support?", is_cuda_available)

    # Check the number of available GPUs
    gpu_count = paddle.device.cuda.device_count()
    print("Number of GPUs available:", gpu_count)

    # Optionally, get details about each GPU (e.g., name)
    if is_cuda_available and gpu_count > 0:
        gpu_name = paddle.device.cuda.get_device_name()
        print("GPU name:", gpu_name)
    else:
        print("No GPU found or CUDA is not available.")
