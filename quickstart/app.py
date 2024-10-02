from beam import function, Image


@function(image=Image(python_packages=["numpy"]))
def hello_world():
    import numpy
    return "I'm running in a remote container!"