# Fine-Tuning Gemma 2B on Beam

## Fine-Tuning
In `finetune.py`, we are using Low-Rank Adaption (LoRA) to fine-tune the Gemma language model using the Open Assistant dataset. The goal is to use this dataset to improve Gemma's ability to engage in helpful conversations, making it more suitable for assistant-like apps. 

### LoRA
You can read more about LoRA [here](https://arxiv.org/abs/2106.09685). However, let's briefly discuss what exactly it does and why we chose to use it here. At a high level, LoRA introduces a new small set of weights to the model that we will be training. By limiting our training to these additional weights, we can fine-tune the model much quicker. Additionally, since we are not touching the original weights, the model's initial knowledge base should intact. 

### Training our model
In this example, we are using an [A100-40](https://www.nvidia.com/en-us/data-center/a100/). We are using mixed precision (FP16) to optimize for speed and memory usage. In this example, we are only training for one epoch. In practice, you can probably train longer and continue to see improvemed results. 

No surprise here, but we are getting our compute via Beam. We are using the `function` decorator so that we can run our fine-tuning application as if it were on our local machine. 
```python
@function(
    volumes=[Volume(name="gemma-ft", mount_path=MOUNT_PATH)],
    image=Image(
        python_packages=["transformers", "torch", "datasets", "peft", "bitsandbytes"]
    ),
    gpu="A100-40",
    cpu=4,
)
```
One interesting thing to note above is that we are mounting a volume to our container. This volume is where we have uploaded our intial weights from hugging face and our training dataset. It is also where we will store our additional fine-tuned weights. 

We can start our training by running `python finetune.py`. After beginning training, you should see something like the following in your terminal: 
```bash
=> Building image 
=> Syncing files 
...
=> Running function: <finetune:gemma_fine_tune> 
Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
...
Generating train split: 12947 examples [00:00, 114393.80 examples/s]
...
Map:  93%|#########2| 12000/12947 [00:13<00:01, 921.12 examples/s]
...
  1%|          | 6/809 [00:08<16:35,  1.24s/it]
...
{'loss': 1.617, 'grad_norm': 0.4805833399295807, 'learning_rate': 0.00019752781211372064, 'epoch': 0.01}
...
```

Once it is finished, we can use the beam cli to look at the resulting files. You should see something like this:
```bash
❯ beam ls gemma-ft/gemma-2b-finetuned

  Name                                                Size   Modified Time   IsDir
 ──────────────────────────────────────────────────────────────────────────────────
  gemma-2b-finetuned/README.md                    4.97 KiB   Aug 10 2024     No
  gemma-2b-finetuned/adapter_config.json          644.00 B   Aug 10 2024     No
  gemma-2b-finetuned/adapter_model.safetensors   12.20 MiB   Aug 10 2024     No
  gemma-2b-finetuned/checkpoint-700              36.70 MiB   Aug 01 2024     Yes
  gemma-2b-finetuned/checkpoint-800              36.70 MiB   Aug 01 2024     Yes
  gemma-2b-finetuned/checkpoint-809              36.70 MiB   Aug 01 2024     Yes
  gemma-2b-finetuned/special_tokens_map.json      555.00 B   Aug 10 2024     No
  gemma-2b-finetuned/tokenizer.json              16.71 MiB   Aug 10 2024     No
  gemma-2b-finetuned/tokenizer_config.json       45.21 KiB   Aug 10 2024     No

  9 items | 139.06 MiB used
```

## Inference
In `inference.py`, we are loading up our model with the additional fine-tuned weights and setting up an endpoint to send it requests. Note, that we make use of the Beam's `on_start` functionality so that we only load the model when the container starts instead of every time we receive a request. Let's explore the `endpoint` decorator below. 
```python
@endpoint(
    name="gemma-inference",
    on_start=load_finetuned_model,
    volumes=[Volume(name="gemma-ft", mount_path=MOUNT_PATH)],
    cpu=1,
    memory="16Gi",
    gpu="T4",
    image=Image(
        python_version="python3.9",
        python_packages=["transformers==4.42.0", "torch", "peft"],
    ),
    autoscaler=QueueDepthAutoscaler(max_containers=5, tasks_per_container=1),
)
```
Once again, we are mounting our storage volume named "gemma-ft". Since we have already run training, this volume will now contain our fine-tuned weights alongside the base weights we got from huggingface. 

### Choosing a GPU for inference
Now that we've trained the model, we can run it on a machine with a weaker GPU. Training requires more memory than inference because it must store gradients and optimizer states for all parameters, in addition to activations, whereas inference only needs to maintain the current layer's activations during a forward pass. Be sure to keep this in mind as you work on your own applications. You can use the [Beam dashboard](https://platform.beam.cloud/) to get a sense of GPU utilization in real-time. With this information, you can make a more informed choice about how much compute you require. 

### Using signals to reload model weights
We are also making use of a new experimental feature calls `Signal`. This allows us to communicate between apps on Beam. In this example, we have it setup to re-run our on-start method when a signal is received. This way, if we re-train our model, we can load the newest weights without restarting the container.  

### Deploying our endpoint
Let's deploy our endpoint! We can do this easily with the `beam` cli. 
```
beam deploy inference.py:predict --name gemma-ft
```
The output will look something like this: 
```bash
=> Building image
=> Syncing files 
=> Deploying 
=> Deployed 🎉 
=> Invocation details 
curl -X POST 'https://app.beam.cloud/endpoint/gemma-ft/v2' \
-H 'Connection: keep-alive' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer {YOUR_AUTH_TOKEN}' \
-d '{}'
```
When calling our inference endpoint, we'll need to include a prompt. For example, we can call the deployed endpoint with `-d '{"prompt": "hi"}`. The response we get back will be in the following format: 
```bash
{"text":"Hello! How can I help you today?<|im_end|>"}
```
Note that the returned response includes the stop tokens `<|im_end|>`. You could strip this token in the endpoint logic if you would like, but it is worth keeping around if you will be appending this response to a longer running conversation. 
