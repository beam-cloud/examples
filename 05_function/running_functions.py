"""
** Beam Quickstart **

This app demonstrates a few things:

1. You can wrap any code in a `function()` decorator to run it on the cloud 
2. You can invoke a function locally with `.local()`
3. You can invoke a function _remotely_ with `.remote()`
4. You can fan out workloads to hundreds or thousands of containers with `.map()`

In your shell, run this script by running `python [app.py]`. 

The logs from the remote container runtime will stream back to your shell, and you can watch as
the remote code is executed in the cloud. 

The `.map()` method in `main()` spawns 5 remote containers, but that number is arbitrary. 

You can change `5` to `500`, and invoke this function again -- you'll see 500 containers spin up almost instantly.
"""

from beam import function


@function(cpu="100m", memory="100Mi")  # Each function runs on 100 millicores of CPU
def square(x):
    sum = 0

    for i in range(x):
        sum += i**2

    return {"sum": sum}


def main():
    print(square.local(x=10))
    print(square.remote(x=10))

    for i in square.map(range(5)):  # Spin up 5 containers
        print(i)


if __name__ == "__main__":
    main()
