from beam import endpoint, Image, function


image = (
    Image(
        base_image="docker.io/nvidia/cuda:12.3.1-runtime-ubuntu20.04",
        python_version="python3.10",
    )
    .add_commands(["apt-get update -y", "apt-get install neovim -y"])
    .add_python_packages(["torch"])
)


@endpoint(app="examples", image=image)
def handler():
    import torch

    return {"torch_version": torch.__version__}