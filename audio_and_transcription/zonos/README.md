# Zonos TTS API Example

A text-to-speech API service powered by Zonos.

## Deployment

```bash
beam deploy app.py:generate
```

## API Usage

Send a `POST` request with the following JSON payload:

```json
{
    "text": "Your text to convert to speech",
}
```

curl -X POST 'https://ff6e671a-c43d-468e-ab21-df87c8d87afb.app.beam.cloud' \
-H 'Connection: keep-alive' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer dERxHlMCz4TDU9k7cxZte_FrILTCvs0nf3KSVe8oZumoTAOa4OIkpJiyGOq_hS9nyangjUG6GC9VmswWd_Rt4g==' \
-d '{ "text": "On Beam run AI workloads anywhere with zero complexity. One line of Python, global GPUs, full control"}'

### Example Request:

```json
{
    "text": "On Beam run AI workloads anywhere with zero complexity. One line of Python, global GPUs, full control",
}
```

### Example Response:

A generated audio file will be returned. 

## Audio Example

Here’s an example of the generated audio output:

<audio controls>
  <source src="https://app.beam.cloud/output/id/704defd0-9370-4499-9124-677925e64961" type="audio/mpeg">
</audio>

