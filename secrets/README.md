# Managing Secrets and Environment Variables

You can read and write secrets that will be available to your apps.

To create a secret, use the Beam CLI:

```python
beam secret create [KEY] [VALUE]
```

Once the secret is created, it can be accessed as an environment variable:

```python
import os
from beam import function

os.environ["FOO"] = "bar"


@function(secrets=["FOO"])
def handler():
    import os

    my_secret = os.environ["FOO"]
    return f"secret {my_secret}"
```
