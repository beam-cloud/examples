"""
** Custom Images **

You can customize the remote container that runs your code with 
Python packages, a specific version of Python, shell commands, and a custom base image.
"""

from beam import endpoint, Image


image = Image(
    python_version="python3.9",
    python_packages=[
        "torch",
    ],
    commands=["apt-get update -y && apt-get install neovim -y"],
    base_image="docker.io/nvidia/cuda:12.3.1-runtime-ubuntu20.04",
)


@endpoint(image=image)
def handler():
    import torch

    return {"torch_version": torch.__version__}
