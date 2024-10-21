# Web Scraping

The code in this example demonstrates how Beam functions can be used to scrape Wikipedia pages in parallel.

## Batched Web Scraping with Beam Functions

The batched crawling example uses the `map` method from the `function` decorator to launch a remote function calls for each element of the input list. 

Run this example: `python mapped_crawl.py`

## Continuous Web Scraping with Beam Functions and Threads

The continuous crawling example uses a thread pool to launch remote function calls in parallel. The main difference between this example and the batched crawl is that this one launches a new remote function as soon as a slot opens up, rather than waiting for all the remote function calls to complete before launching the next batch.

Run this example: `python threaded_crawl.py`
