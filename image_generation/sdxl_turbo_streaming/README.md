## sdxl-turbo-streaming

This app generates images with SDXL Turbo in streaming mode.

- GPU inference powered by [Beam](https://beam.cloud)
- Frontend powered by [Nextjs](https://nextjs.org/)

# Quickstart

0. Create a free account on [Beam](https://beam.cloud) and follow the instructions in the onboarding to add your credentials to your local environment
1. Create a virtual environment: `python3 -m virtualenv .venv && source .venv/bin/activate`
2. Install Beam: `pip3 install beam-client`
3. Insall nodejs `https://nodejs.org/en/download/package-manager` (Pay attention, it must be at least the minimal version supported by nextjs)
4. Install nextjs: `https://nextjs.org/docs/getting-started/installation`

# Usage

## Start Backend

1. `cd backend && beam deploy app.py:generate`

## Start Frontend

Configuration steps:

1. `cd frontend`
2. Create a `.env` file with your beam key in the enviornment variables `NEXT_PUBLIC_BEAM_AUTH_TOKEN` and `NEXT_PUBLIC_BEAM_API_URL`

Running steps:

1. `cd frontend`
2. `npm install`
3. `npm run dev`
