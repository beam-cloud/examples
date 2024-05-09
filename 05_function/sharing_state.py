"""
** Sharing State ** 

You have the option of sharing state between tasks using the `Queue()` abstraction.

`Queue()` is a concurrency-safe distributed queue, accessible both locally and within remote containers.
"""

from beam import Queue

val = [1, 2, 3]

# Initialize the Queue
q = Queue(name="myqueue")

for i in range(100):
    # Insert something to the queue
    q.put(val)
while not q.empty():
    # Remove something from the queue
    val = q.pop()
    print(val)
