# vLLM

This example demonstrates how to use the vLLM library to run inference on Beam. 

## Running simple inference
The `inference.py` script demonstrates how to do ["one-off inference"](https://docs.vllm.ai/en/latest/getting_started/examples/offline_inference.html) with vLLM on the Beam platform. This will download the Yi-Coder-9B-Chat model and use it to run inference on a simple prompt.

```bash
python inference.py
```

## Deploying an API

The `yicoder.py` script demonstrates how to deploy a vLLM openai-compatible server. It uses the Yi-Coder-9B-Chat model. The Beam SDK includes a special wrapper class (`VLLM`) that makes it easy to deploy vLLM servers.

```bash
beam deploy yicoder.py:yicoder
```

Once the server is deployed, you can use the `yichat.py` script to chat with the model. However, make sure to replace the `YOUR_BEAM_TOKEN` with your actual Beam token. Also, if you don't have the OpenAI python package installed, you can install it with `pip install openai`.

```bash
python yichat.py
```

## Multi-modal inference

We can deploy a multi-modal model like Phi-3.5-vision-instruct. The `phi.py` script demonstrates how to do this.

```bash
beam deploy phi.py:phi
```

Once the server is deployed, you can use the `phichat.py` script to chat with the model.

```bash
python phichat.py
```

You can have a conversation like this:

```bash
Welcome to the CLI Chat Application!
Type 'quit' to exit the conversation. Image link is optional.
Model is ready
Question: What kind've animal is this? 
Image link: https://upload.wikimedia.org/wikipedia/commons/b/b2/Endangered_Red_Panda.jpg
Assistant:  It's a red panda.
```