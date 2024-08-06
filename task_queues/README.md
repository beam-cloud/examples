# Async Tasks

Task Queues are great for deploying resource-intensive functions on Beam.
Instead of processing tasks immediately, the task queue enables you to add
tasks to a queue and process them later, either sequentially or concurrently.

Task queues are deployed the same way as web endpoints.

As a recap, this is the CLI command to deploy a task queue or endpoint:

```sh
beam deploy [file.py]:[function] --name [name]
```

For example, if you wanted to deploy this file, the command would be:

```sh
beam deploy app.py:multiply --name my-app
```
