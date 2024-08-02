"""
** Sharing State ** 

You have the option of sharing state between tasks using the `Queue()` abstraction.

`Queue()` is a concurrency-safe distributed queue, accessible both locally and within remote containers.
"""

from beam import Queue, function


@function(cpu=0.1)
def access_queue(): 
    q = Queue(name="myqueue")
    return q.pop()

if __name__ == "__main__":
    val = ["eli", "luke", "john", "nick"]

    # Initialize the Queue
    q = Queue(name="myqueue")

    for i in val:
        # Insert something to the queue
        q.put(i)

    while not q.empty():
        # Remove something from the queue
        val = q.pop()
        print(val)

    q.put("daniel")
 
    print(access_queue.remote())