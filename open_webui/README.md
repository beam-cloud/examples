# Open WebUI Server

We'll deploy [Open WebUI](https://github.com/open-webui/open-webui) using a Beam Pod. Useful for running a self-hosted chat UI connected to a custom LLM backend.

## Usage

Edit and run the script below to deploy your Open WebUI pod:

```bash
python app.py
```

Once deployed, the server URL will be printed to your console. Open it in your browser to access the WebUI.

## Notes

- Update your `BEAM_LLM_API_BASE_URL` and `BEAM_API_KEY` before running.
- The base URL must be an OpenAI-compatible endpoint.  
  For example, you can follow this [guide for using Qwen2.5-7B with SGLang](https://docs.beam.cloud/v2/examples/sglang).