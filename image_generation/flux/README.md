## Serverless Inference with Flux

Here's how to run serverless inference with Flux.

> Flux is a gated model on Huggingface. To use it, you must request access [here](https://huggingface.co/black-forest-labs/FLUX.1-dev).

0. Setup Beam on your computer by following the steps [here](platform.beam.cloud/onboarding).
1. Save your [Huggingface API key](https://huggingface.co/settings/tokens) to Beam: `beam secret create HF_TOKEN [YOUR-HF-TOKEN]`
2. Download this example to your computer: `beam example download image_generation/flux && cd mage_generation/flux`
3. Deploy it to Beam: `beam deploy app.py:generate`
