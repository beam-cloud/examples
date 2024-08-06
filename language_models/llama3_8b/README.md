# Meta Llama 3 8B Instruct

> Note: This is a gated Huggingface model and you must request access to it here: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

Retrieve your HF token from this page: https://huggingface.co/settings/tokens

After your access is granted, make sure to save your Huggingface token on Beam:

```sh
$ beam secret create HF_TOKEN
```

...and add the secret to your Beam function decorator:

```python
@endpoint(secrets=["HF_TOKEN"])
```

After the endpoint is deployed, you can call it like this:

```sh
curl -X POST 'https://app.beam.cloud/endpoint/id/[ENDPOINT-ID]' \
-H 'Connection: keep-alive' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer [AUTH-TOKEN]' \
-d '{
    "messages": [
        {"role": "system", "content": "You are a yoda chatbot who always responds in yoda speak!"},
        {"role": "user", "content": "Who are you?"}
    ]
}'
```
