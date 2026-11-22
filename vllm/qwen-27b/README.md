# Serve Qwen 27B on a B200

Run Qwen 27B, or your own fine-tune of it, behind an OpenAI-compatible API on Beam. One B200 served 200 concurrent requests at over 5,500 output tokens per second in our test, with a p95 time to first token under 2 seconds.

This example uses [vLLM](https://github.com/vllm-project/vllm) and starts with the base `Qwen/Qwen3.8-27B` weights. Once it's running, switching to your own weights is a one-line change.

| File | What it is |
| --- | --- |
| `app.py` | The deployment: image, GPU, volume and the `vllm serve` command. |
| `load_test.py` | Sends a batch of concurrent requests and reports throughput and latency. |

## 1. Deploy

Install the CLI and sign in:

```bash
pip install beam-client
beam login
```

Create the API key that clients will send, then deploy:

```bash
export QWEN_KEY=$(openssl rand -hex 24)
beam secret create VLLM_API_KEY "$QWEN_KEY"
beam deploy app.py:qwen --json
```

The output includes a `deployment_id` and an `invoke_url`. Use the ID to keep one replica running around the clock:

```bash
beam deployment scale <deployment_id> --replicas 1
```

The first start downloads the weights to the `qwen-27b-weights` volume, and every later start reuses them. Follow the startup with:

```bash
beam logs qwen-27b --follow
```

The server is ready when the log shows `Application startup complete` and `/health` returns 200.

## 2. Call it

Use the `invoke_url` from the deploy output. It belongs to that version of the deployment; for a URL that follows every new deploy, see [Pod URLs](https://docs.beam.cloud/v2/pod/web-service#pod-urls). `beam secret show VLLM_API_KEY` prints the key again if you need it later.

```bash
export QWEN_URL=https://<invoke-url-host>

curl $QWEN_URL/v1/chat/completions \
  -H "Authorization: Bearer $QWEN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen-27b", "messages": [{"role": "user", "content": "Hello"}]}'
```

With the OpenAI client:

```python
import os
from openai import OpenAI

client = OpenAI(base_url=os.environ["QWEN_URL"] + "/v1", api_key=os.environ["QWEN_KEY"])

response = client.chat.completions.create(
    model="qwen-27b",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

Any OpenAI-compatible client works. Keep its automatic retries on, as you would for any production API.

Thinking is on by default, and the reasoning comes back in `reasoning_content`, separate from the answer. To turn it off for a request, pass `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`. Tool calling is enabled.

## 3. Serve your own fine-tune

Change `MODEL` in `app.py`, then deploy again. Clients keep using the model name `qwen-27b`, so nothing changes on their side.

**From a private Hugging Face repo.** Store a token and add it to the app's secrets:

```bash
beam secret create HF_TOKEN hf_...
```

```python
MODEL = "your-org/your-finetune"
...
    secrets=["VLLM_API_KEY", "HF_TOKEN"],
```

**From local files.** Upload the checkpoint directory to the volume, then point `MODEL` at its path:

```bash
beam cp ./my-finetune beam://qwen-27b-weights/my-finetune
```

```python
MODEL = "/models/my-finetune"
```

The directory needs the same files as a Hugging Face model repo: `config.json`, the tokenizer files and the `.safetensors` shards. A LoRA adapter can be merged into the base weights first, or served on top of the base model with vLLM's `--lora-modules` flag.

## 4. Load test

```bash
pip install openai
python load_test.py --requests 200 --concurrency 100
```

Run it from a cloud machine, so the results measure the model rather than your local connection.

### Results

One B200, `Qwen/Qwen3.8-27B` in BF16, thinking off, 512 max output tokens (about 435 used per request):

| Concurrent requests | Output tokens/s | First token, p50 | First token, p95 | 200 requests finish in |
| --- | --- | --- | --- | --- |
| 1 | 74 | 1.3s | 2.4s | |
| 100 | 3,300 to 4,400 | 0.6s | 1.2s | 20 to 26s |
| 200 | 5,550 to 5,940 | 1.3s | 1.9s | 15 to 16s |

The 100 and 200 rows come from a client running in the cloud, 10 and 6 runs each. The 100 runs shared the GPU with a second test client, so they understate what 100 concurrent requests get on their own. The single-request row was measured from a laptop.

The weights use 51 GB of the B200's 178 GB. The rest holds 1.86 million tokens of KV cache at the model's full 262,144-token context.

## 5. Scale past one GPU

Each replica is one B200 running its own copy of the model. Add replicas with one command:

```bash
beam deployment scale <deployment_id> --replicas 2
```

For steady production traffic, reserve your B200s in a [compute pool](https://docs.beam.cloud/v2/scaling/pools) so that capacity is always yours, then pass the pool name to the `Pod` in `app.py`:

```python
    pool="your-pool-name",
```

## Settings worth knowing

| Setting | Where | Notes |
| --- | --- | --- |
| `gpu` | `app.py` | `"B200"` or `"B300"`. The results above are from a B200. |
| `--max-num-seqs` | `app.py` | Requests processed at once. Extra requests queue inside vLLM. |
| `--max-model-len` | add to `app.py` | Defaults to the model's 262,144 tokens. Lower it to cap the size of a single request. |
| `memory` | `app.py` | 120 GB lets the checkpoint be read into RAM before loading. Raise it for a larger checkpoint. |
| FP8 weights | `MODEL` | `Qwen/Qwen3.8-27B-FP8` is 31 GB instead of 56 GB, if you want more room for KV cache. |

## Stop it

```bash
beam deployment stop <deployment_id>
```

The weights stay on the volume, so the next start skips the download.
