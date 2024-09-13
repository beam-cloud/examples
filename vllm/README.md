# vLLM

This example demonstrates how to use the vLLM library to run inference on Beam. 

## Running simple inference
The simplest way to get started with vLLM on Beam is to run the `inference.py` script. This will download the Yi-Coder-9B-Chat model and use it to run inference on a simple prompt.

```bash
python inference.py
```

## Deploying an API server

The `yicoder.py` script demonstrates how to deploy an vLLM server for the Yi-Coder-9B-Chat model that can be used with the OpenAI sdk.

```bash
beam deploy yicoder.py:yicoder_server --name yicoder-server
```

Once the server is deployed, you can use the `yichat.py` script to chat with the model. However, make sure to replace the `YOUR_BEAM_TOKEN` with your actual Beam token. Also, if you don't have the OpenAI python package installed, you can install it with `pip install openai`.

```bash
python yichat.py
```
