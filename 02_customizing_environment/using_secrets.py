"""
Warning: This is not implemented yet!
"""

from beam import function
import os

MY_SECRET = os.environ["MY_SECRET"]


@function()
def handler():
    print(f"Secret: {MY_SECRET}")
