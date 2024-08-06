# Preload Models

Beam includes an `on_start` lifecycle hook which is useful for running operations
that only need to happen once when a container first starts.

As an example, loading model weights is an operation that only needs to happen once.

In the code below, a `download_models()` function is attached to `on_start`.

The `download_models` code runs exactly once when the container first starts.
