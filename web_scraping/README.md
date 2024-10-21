# Web Scraping

The code in this example demonstrates how Beam functions can be used to scrape Wikipedia pages in parallel.

## Mapped Crawl

The mapped crawl uses the `map` method from the `function` decorator to launch a remote function calls for each element of the input list. 

Run this example: `python mapped_crawl.py`

## Threaded Crawl

The threaded crawl uses a thread pool to launch remote function calls in parallel. The main difference between this example and the mapped crawl is that the threaded crawl launches a new remote function as soon as a slot opens up, rather than waiting for all the remote function calls to complete before launching the next batch.

Run this example: `python threaded_crawl.py`
