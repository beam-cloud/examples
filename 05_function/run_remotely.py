"""
** Remote Functions **

You can add a decorator to any Python function to run it remotely on Beam.

By adding a `function()` decorator to `handler`, the code in `handler` will run 
on the cloud -- not your laptop -- when invoked.

Functions can be invoked in the module directly, using the `.remote()` method
"""

from beam import function, Image


@function(cpu=2, memory=128, image=Image(python_packages=["numpy"]))
def handler():
    import numpy as np

    return {"arr": np.zeros((2, 3))}


if __name__ == "__main__":
    print(handler.remote())
