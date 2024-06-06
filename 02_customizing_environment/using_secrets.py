"""
Managing Secrets and Environment Variables in Beam

You can read and write secrets that will be available to your apps.

To create a secret, use the Beam CLI: `beam secret create [KEY] [VALUE]`

Once the secret is created, it can be accessed as an environment variable (see below).
"""

from beam import function


@function(secrets=["AWS_ACCESS_KEY"])
def handler():
    import os

    my_secret = os.environ["AWS_ACCESS_KEY"]
    print(f"Secret: {my_secret}")


if __name__ == "__main__":
    handler()
