from beam import function


@function(callback_url="https://www.beam.cloud/")
def handler(x):
    return {"result": x}


if __name__ == "__main__":
    handler.remote(x=10)
