"""
** Configure Keep Warm ** 

You can add a `keep_warm_seconds` argument to your functions to control
how long the container should stay up before shutting down.
"""

from beam import endpoint


# Each container will stay up for 5 min before shutting down automatically
@endpoint(keep_warm_seconds=300)
def handler():
    return {}
