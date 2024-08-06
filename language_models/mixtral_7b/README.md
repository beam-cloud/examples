# Mixtral 7B

> Note: This is a gated Huggingface model and you must request access to it here: https://huggingface.co/mistralai/Mistral-7B-v0.1

Retrieve your HF token from this page: https://huggingface.co/settings/tokens

After your access is granted, make sure to save your Huggingface token on Beam:

```sh
$ beam secret create [SECRET]
```

...and add the secret to your Beam function decorator:

```python
@endpoint(secrets=["HF_TOKEN"])
```
