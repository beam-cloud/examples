"""
** Async Tasks **

Task Queues are great for deploying resource-intensive functions on Beam. 
Instead of processing tasks immediately, the task queue enables you to add 
tasks to a queue and process them later, either sequentially or concurrently.

Task queues are deployed the same way as web endpoints. 

As a recap, this is the CLI command to deploy a task queue or endpoint:

```
beam deploy [file.py]:[function] --name [name]
```

For example, if you wanted to deploy this file, the command would be:

```
beam deploy async_task.py:multiply --name my-app
```

"""

from beam import task_queue, Image


@task_queue(
    cpu=1.0,
    memory=128,
    gpu="T4",
    image=Image(python_packages=["torch"]),
    keep_warm_seconds=1000,
)
def multiply(**inputs):
    result = inputs["x"] * 2
    return {"result": result}


if __name__ == "__main__":
    # Interactively enqueue a task without deploying
    multiply.put(x=1)
