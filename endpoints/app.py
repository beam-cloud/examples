from beam import endpoint


@endpoint(
    cpu=1.0,
    memory=128,
)
def multiply(**inputs):
    result = inputs["x"] * 2
    return {"result": result}
