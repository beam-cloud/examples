from beam import endpoint, Image


@endpoint(
    name="inference",
    cpu=1,
    memory="1Gi",
    image=Image().add_python_packages(["numpy"]),
)
def predict(**inputs):
    x = inputs.get("x", 256)
    return {"result": x**2}
