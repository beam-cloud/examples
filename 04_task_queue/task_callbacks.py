"""
** Task Callbacks **

You can fire callbacks to your own web server when a Beam task finishes running. 

If you supply a `callback_url` argument to your function decorator, 
Beam will make a POST request to your server whenever a task finishes running. 

Callbacks fire for both successful and failed tasks.
"""

from beam import function


@function(callback_url="https://www.beam.cloud/")
def handler(x):
    return {"result": x}


if __name__ == "__main__":
    handler.remote(x=10)
