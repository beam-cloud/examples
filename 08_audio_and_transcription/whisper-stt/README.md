# Using Whisper on Beam

## Prerequisites

1. Create a [Beam account](https://platform.beam.cloud) (if you are added to waitlist, email eli [at] beam [dot] cloud to approve your registration)
2. Click the `Onboarding` button on the top right of the page and follow the instructions.

## Using this example

- `app.py` is the inference function that will be deployed to Beam
- `request.py` is a simple benchmarking script that will call the deployed API and measure inference and cold boot time

### Deploy the app

```
beam deploy app.py:transcribe
```

### Run the benchmarking script

```
python request.py
```
