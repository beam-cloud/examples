import subprocess
from beam import endpoint


@endpoint(app="examples", gpu="T4")
def handler():
    print(subprocess.check_output(["nvidia-smi"]))
    return "This container has a GPU attached 📡!"


if __name__ == "__main__":
    handler.serve()
