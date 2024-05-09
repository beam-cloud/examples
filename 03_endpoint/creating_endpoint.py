"""
** Web Endpoints **

Any code can be deployed as a web endpoint by adding an `endpoint()` decorator

This is the CLI command to deploy an endpoint:

```
beam deploy [file.py]:[function] --name [name]
```

For example, if you wanted to deploy this file, the command would be:

```
beam deploy creating_endpoint.py:multiply --name my-app
```
"""

from beam import endpoint


@endpoint(
    cpu=1.0,
    memory=128,
)
def multiply(**inputs):
    result = inputs["x"] * 2
    return {"result": result}
