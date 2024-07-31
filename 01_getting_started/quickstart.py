from beam import endpoint


@endpoint(name="quickstart", cpu="100m", memory="100Mi")
def run():
    print("🔮 This is running remotely on Beam!")
    return {
        "success": "Nice work! Check out the docs to continue: https://docs.beam.cloud/v2/getting-started/quickstart"
    }
