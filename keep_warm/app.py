from beam import endpoint


# Each container will stay up for 5 min before shutting down automatically
@endpoint(keep_warm_seconds=300)
def handler():
    return "warm"
