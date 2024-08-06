# Invoking functions in other apps

This example demonstrates how to invoke functions in other apps on Beam.
Specifically, we cover the scenario with an inference and a retraining function

The retraining function needs a way to tell the inference function to use the latest model.

We use a `experimental.Signal()`, which is a special type of event listener that can be triggered from the retrain function.

To test this, open two terminal windows:

In window 1, serve and invoke the inference function
In window 2, serve and invoke the retrain function

Look at the logs in window 1 -- you'll notice that the signal has fired, and load_latest_model ran again.
