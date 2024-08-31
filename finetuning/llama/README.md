# Fine-Tuning Llama3 with LoRA and Beam

Let’s say you want to fine-tune llama3. By running PEFT LoRA fine-tuning on a dataset with a few thousand rows in it, you can dramatically increase the accuracy of your model’s responses, and if you use Beam for compute, you can accomplish it in 1 hour rather than 10.

## Setup Beam and Llama3 model

1. Set up the Beam API token

First, run pip install beam-client in your terminal. It’ll ask you for an API key.

Go to beam.cloud and create a Pay-As-You-Go Developer account. You’ll receive the API key and instructions on how to save it to your environment. Follow these instructions.

1. Access Llama3 weights

Next, you need the weights of the Llama3 base model to begin fine-tuning them. Go to [llama.meta.com/llama-downloads/](https://llama.meta.com/llama-downloads/) to request access, which is typically granted immediately, along with a custom URL and instructions on using it to download the weights. Follow the instructions.

1. Upload the weights and dataset to a Beam Volume

Finally, upload the weights and your fine-tuning dataset to a Beam Volume.

To create a fine-tuning dataset, you need a collection of input-output pairs, formatted as a CSV, JSON, or text file, where each entry provides an example of what the model should predict given a specific input. Reference [these examples](https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/finetuning/datasets) from Meta for Llama3 formatting. 

Once you have your fine-tuning dataset, use the following CLI commands.

```bash
// using weights as the Volume name
$ beam volume create llama-ft

// assuming your weights are saved locally to local_weights
$ beam cp local_weights llama-ft/weights

// assuming your fine-tuning dataset is saved locally as local_dataset
$ beam cp local_dataset llama-ft/data
```

## Fine-tune using LoRA

1. Configure LoRA

We will be leveraging the `transfomers` and `peft` packages to run LoRA PEFT fine-tuning, with the same Python as you would for a local run. Here is the default configuration we will be working with.

```bash
# finetune.py
def llama_fine_tune():
    import os
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not torch.cuda.is_available():
        return "CUDA is not available"

    torch.set_float32_matmul_precision("high")

    # Load the Llama3 model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHT_PATH, device_map="auto", attn_implementation="eager", use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(WEIGHT_PATH, use_fast=False)
    
    # Set the pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token


    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Load the Yelp Reviews dataset from Hugging Face
    dataset = load_dataset(DATASET_PATH)

    def prepare_dataset(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(prepare_dataset, batched=True)

    training_args = TrainingArguments(
        # This output directory is on our mounted volume
        output_dir="./llama-ft/llama-finetuned",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        gradient_checkpointing=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    # Saving the LORA model and tokenizer to our mounted volume so that our inference endpoint can access it.
    model.save_pretrained("./llama-ft/llama-finetuned")
    tokenizer.save_pretrained("./llama-ft/llama-finetuned")
```

1. Configure Beam

That’s all set to run locally, but it would take hours and hours on consumer hardware. Fortunately, we can run it on Beam’s serverless GPUs by adding a few lines of code.

```bash
# finetune.py
# Deploy to beam by running `$ python finetune.py` in the terminal
from beam import Volume, Image, function, env

# The mount path is the location on the beam volume that we will access. 
MOUNT_PATH = "./llama-ft"
WEIGHT_PATH = "./llama-ft/weights"
DATASET_PATH = "./llama-ft/data"

@function(
    secrets=["HF_TOKEN"],
    volumes=[Volume(name="llama-ft", mount_path=MOUNT_PATH)],
    image=Image(
        python_packages=["transformers", "torch", "datasets", "peft", "bitsandbytes"]
    ),
    gpu="A100-40",
    cpu=4,
)
def llama_fine_tune():
   # this is unchanged

if __name__ == "__main__":\
    llama_fine_tune.remote()
```

Within the `@function` decorator, we’ve done three things. First, we set paths to the Volume we created earlier. Second, we mounted an Image, which gives Beam access to the Python packages we describe. And third, we selected the GPU and CPU specs for our run. 

Outside of the decorator, we modified the script to call llama_fine_tune() with the remote() method, which will send the compute to the Beam CLI.

1. Run the fine-tuning function through the Beam CLI

The moment of truth! Simply run `python [finetune.py](http://finetune.py)` in the terminal, and you should receive output like this:

```bash
=> Building image
=> Syncing files
...
=> Running function: <finetune:llama_fine_tune>
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

Once this is complete, use the Beam CLI to check on the output files:

```bash
$ beam ls llama-ft/llama-finetuned

  Name                                                Size   Modified Time   IsDir
 ──────────────────────────────────────────────────────────────────────────────────
  llama-finetuned/README.md                    4.97 KiB   Aug 10 2024     No
  llama-finetuned/adapter_config.json          644.00 B   Aug 10 2024     No
  llama-finetuned/adapter_model.safetensors   12.20 MiB   Aug 10 2024     No
  llama-finetuned/checkpoint-700              36.70 MiB   Aug 01 2024     Yes
  llama-finetuned/checkpoint-800              36.70 MiB   Aug 01 2024     Yes
  llama-finetuned/checkpoint-809              36.70 MiB   Aug 01 2024     Yes
  llama-finetuned/special_tokens_map.json      555.00 B   Aug 10 2024     No
  llama-finetuned/tokenizer.json              16.71 MiB   Aug 10 2024     No
  llama-finetuned/tokenizer_config.json       45.21 KiB   Aug 10 2024     No

  9 items | 139.06 MiB used

```

## Using your Fine-Tuned Model

1. Write an inference function

Start off by writing a local function to call inference on the model, customizing it to the format of your fine-tuning dataset:

```bash
# inference.py
def predict(**inputs):
    global model, tokenizer, stop_token_ids  # These will have the latest values

    prompt = inputs.get("prompt", None)
    if not prompt:
        return {"error": "Please provide a prompt."}

    # Now we will format the user provided prompt so that it is of the format that
    # the fine tuning dataset established.
    prompt = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"

    # We set the end of sequence token to the last token from <|im_end|>
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        use_cache=False,
        eos_token_id=stop_token_ids[-1],
        pad_token_id=tokenizer.eos_token_id,
    )
    # Here we are trimming the input length from the output so that only the newly generated text is returned.
    text = tokenizer.decode(output[0][len(inputs[0]) :])
    print(text)

    return {"text": text}
```

1. Add the endpoint decorator and a function to load the model

Like before, we’ll add a decorator to this function to allow Beam to access it, but we’ll also write another function, `load_finetuned_model()` that we pass into the decorator so Beam knows how to load the model. 

```bash
# inference.py
from beam import Image, endpoint, env, Volume, QueueDepthAutoscaler, experimental

MOUNT_PATH = "./llama-ft"
FINETUNE_PATH = "./llama-ft/llama-finetuned"
MODEL_PATH = "./llama-ft/weights"

# This ensures that these packages are only loaded when the script is running remotely on Beam
if env.is_remote():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

def load_finetuned_model():
    global model, tokenizer, stop_token_ids
    print("Loading latest...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, attn_implementation="eager", device_map="auto", is_decoder=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # using our LORA result via the PEFT library
    model = PeftModel.from_pretrained(model, FINETUNE_PATH)
    print(model.config)

    stop_token = "<|im_end|>"
    stop_token_ids = tokenizer.encode(stop_token, add_special_tokens=False)

@endpoint(
    name="llama-inference",
    on_start=load_finetuned_model,
    volumes=[Volume(name="llama-ft", mount_path=MOUNT_PATH)],
    cpu=1,
    memory="16Gi",
    # We can switch to a smaller, more cost-effective GPU for inference rather than fine-tuning
    gpu="T4",
    image=Image(
        python_version="python3.9",
        python_packages=["transformers==4.42.0", "torch", "peft"],
    ),
    # This autoscaler spawns new containers (up to 5) if the queue depth for tasks exceeds 1
    autoscaler=QueueDepthAutoscaler(max_containers=5, tasks_per_container=1),
)
def predict(**inputs):
    # This function is unchanged

if __name__ == "__main__":
    predict.remote()
```

1. Deploy the endpoint and make API calls

Finally, it’s time to deploy our endpoint! Just run `$ beam deploy [inference.py](http://inference.py/):predict --name llama-ft` and you will get an output like:

```bash
=> Building image
=> Syncing files
=> Deploying
=> Deployed 🎉
=> Invocation details
curl -X POST 'https://app.beam.cloud/endpoint/llama-ft/v2' \
-H 'Connection: keep-alive' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer {YOUR_AUTH_TOKEN}' \
-d '{}'
```

Congrats! Your very own fine-tuned model is now running and accessible through POST requests like:

```bash
response = requests.post(
    "https://app.beam.cloud/endpoint/llama-ft/v2", 
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_AUTH_TOKEN"
    }, 
    json={
        "prompt": "hi"
    }
)
```

which could return:

```bash
{"text":"Hello! How can I help you today?<|im_end|>"}

```

### Useful Links
https://huggingface.co/docs/transformers/en/training

https://llama.meta.com/docs/how-to-guides/fine-tuning/

https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/finetuning/datasets

https://docs.beam.cloud/v2/examples/gemma-fine-tune
