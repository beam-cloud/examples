# Web Endpoints

Any code can be deployed as a web endpoint by adding an `endpoint()` decorator

This is the CLI command to deploy an endpoint:

```
beam deploy [file.py]:[function] --name [name]
```

For example, if you wanted to deploy the `app.py` file in this directory, the command would be:

```
beam deploy app.py:multiply --name my-app
```

"""
