from beam import task_queue, Image


@task_queue(
    app="examples",
    cpu=1.0,
    memory=128,
    gpu="T4",
    image=Image().add_python_packages(["torch"]),
    keep_warm_seconds=1000,
)
def multiply(**inputs):
    result = inputs["x"] * 2
    return {"result": result}


if __name__ == "__main__":
    # Interactively enqueue a task without deploying
    multiply.put(x=1)
