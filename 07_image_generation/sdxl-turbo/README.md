## sdxl-turbo

This app generates images with SDXL Turbo.

- GPU inference powered by [Beam](https://beam.cloud)
- Frontend powered by [Reflex](https://reflex.dev/)

![](/static/reflex-ui.png)

# Quickstart

0. Create a free account on [Beam](https://beam.cloud) and follow the instructions in the onboarding to add your credentials to your local environment
1. Create a virtual environment: `python3 -m virtualenv .venv && source .venv/bin/activate`
2. Install Beam and Reflex: `pip3 install reflex beam-client`

# Usage

## Start Backend

1. `cd backend && beam deploy app.py:generate`
2. Paste the URL returned in the previous step, as well as your Beam auth token, in `./frontend/sdxl_frontend/beam_service.py`

## Start Frontend

1. `cd frontend && reflex run`
