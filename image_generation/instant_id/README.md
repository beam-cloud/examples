# InstantID

## Overview

This repository contains the implementation of InstantID as a Beam endpoint.

In your shell, serve this by running:

`beam serve app.py:generate_image
`

To use this endpoint with curl:

```bash
curl -X POST 'https://your-beam-endpoint.app.beam.cloud' \
-H 'Connection: keep-alive' \
-H 'Content-Type: application/json' \
-H 'Authorization: YOU_BEAM_API' \
-d '{ "image": "https://hips.hearstapps.com/hmg-prod/images/screen-shot-2024-05-22-at-3-00-35-pm-664e410d9114a.png",
    "prompt": "analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette"
  }'
```

Response:

```json
{
  "output_url": "https://app.beam.cloud/output/id/f43f5411-d96b-48b1-bab6-28b2defb9b36"
}
```

## Input Parameters

| Parameter | Description                         | Default Value        |
| --------- | ----------------------------------- | -------------------- |
| `image`   | Input reference image               | URL to sample image  |
| `prompt`  | Text prompt describing output style | "film noir style..." |
