from beam import endpoint, experimental


@endpoint(name="trainer")
def train():
    # Send a signal to another other app letting it know to reload the models
    s = experimental.Signal(name="reload-model")
    s.set(ttl=60)
