import os
from beam import function

os.environ["FOO"] = "bar"


@function(secrets=["FOO"])
def handler():
    import os

    my_secret = os.environ["FOO"]
    return f"secret {my_secret}"


if __name__ == "__main__":
    handler()
