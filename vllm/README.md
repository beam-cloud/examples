# vLLM

This example demonstrates how to use the vLLM library to run inference on Beam. 

## Running simple inference
The `inference.py` script demonstrates how to do ["one-off inference"](https://docs.vllm.ai/en/latest/getting_started/examples/offline_inference.html) with vLLM on the Beam platform. This will download the Yi-Coder-9B-Chat model and use it to run inference on a simple prompt.

```bash
python inference.py
```

## Deploying OpenAI-compatible APIs

Using `model.py`, we can deploy OpenAI-compatible APIs for three different LLMs. To do this, we use the `VLLM` wrapper class from the Beam SDK. Any command line argument that could be passed when calling `vllm serve` can be passed to the wrapper class via the `vllm_args` field. 

The models can be deployed like this: 

```bash
beam deploy model.py:phi_vision_instruct
beam deploy model.py:yicoder_chat
beam deploy model.py:mistral_instruct
```

Each of these models supports different features. The `phi_vision_instruct` model supports multi-modal inference, the `yicoder_chat` model supports chat inference, and the `mistral_instruct` model supports chat inference with tool calling. 

To demonstrate how to use these APIs, we have provided a simple chat client in `chat.py`. This script will allow you to chat with each of the deployed models. 

```bash
python chat.py
```

When you run this script, you will be prompted to enter the url of beam deployment. After you've entered the url, the container will start up and you will be able to chat with whatever model you chose. 

```bash
Welcome to the CLI Chat Application!
Type 'quit' to exit the conversation.
Enter the app URL: https://phi-3-5-vision-instruct-15c4487-v1.app.beam.cloud
Model microsoft/Phi-3.5-vision-instruct is ready
Question: What is in this image?
Image link (press enter to skip): https://upload.wikimedia.org/wikipedia/commons/8/86/Wood.duck.arp.jpg
Assistant:  The image captures a vibrant wood duck in mid-flight, its wings spread wide as it soars through a lush field dotted with yellow flowers. The duck's head is adorned with striking red and black markings, while its body is a mix of green, white, and brown feathers. The perspective of the photo is from below, placing the duck in the center and giving a sense of its impressive wingspan. The background is a vivid green, filled with various shades of green and yellow flowers, providing a stark contrast to the duck's colorful plumage. The image is a beautiful representation of wildlife in its natural habitat
```