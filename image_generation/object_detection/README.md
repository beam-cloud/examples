# Object Detection on Beam

This guide demonstrates how to use Beam to perform object detection on a base64-encoded image string. We will cover how to set up the API endpoint, load the pre-trained model, and visualize the results.

## Overview

We'll create an endpoint that:

- Accepts a base64-encoded image string
- Decodes the image
- Performs object detection using a pre-trained Faster R-CNN model
- Visualizes the detection results
- Returns the resulting image and bounding box coordinates

## Pre-requisites

- An active Beam account
- Basic knowledge of Python
- An image file to test the endpoint

## Test the endpoint

```
beam serve app.py:predict
```

This will print a URL in your shell. Be sure to update `request.py` with your unique URL and auth token:

```
url = 'https://app.beam.cloud/endpoint/id/[ENDPOINT-ID]'
headers = {
    'Connection': 'keep-alive',
    'Content-Type': 'application/json',
    'Authorization': 'Bearer [AUTH_TOKEN]'
}
```

You can run `python request.py` to send a request to the API.

It returns a pre-signed URL with the bounding boxes added to the image:

<img src="https://app.beam.cloud/output/id/95ea6071-2c4a-4618-9397-117345f3e8f2" alt="beam image"/>

## Deploy the endpoint

To deploy a persistent endpoint for production use, run this command:

```
beam deploy app.py:predict
```
