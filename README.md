<p align="center">
<img alt="Logo" src="https://slai-demo-datasets.s3.amazonaws.com/beam-banner.svg"/ width="500">
</p>

<p align="center">
<a href="https://join.slack.com/t/beam-89x5025/shared_invite/zt-1ye1jzgg2-cGpMKuoXZJiT3oSzgPmN8g"><img src="https://img.shields.io/badge/join-Slack-yellow"/></a>
<a href="https://docs.beam.cloud"><img src="https://img.shields.io/badge/docs-quickstart-blue"/></a>

# Get Started

This repo includes various code examples that demonstrate the functionality of Beam.

## Running examples
Some of the examples showcase local usecases and others are examples of full deployments. For the examples that can 
be run locally, you can use poetry to get python setup correctly. 

```bash 
poetry install 
poetry shell
```

---

**Attention Beta9 users**: These examples are for the [beam.cloud](beam.cloud) product. If you are coming from the open-source [Beta9](https://github.com/beam-cloud/beta9/) repo, any of these examples can be run by changing the Python imports from **beam** to **beta9**:

|              | [beam.cloud](https://beam.cloud) | [Beta9](https://github.com/beam-cloud/beta9/) |
| ------------ | -------------------------------- | --------------------------------------------- |
| Imports      | `from beam import endpoint`      | `from beta9 import endpoint`                  |
| CLI Commands | `beam serve app.py:function`     | `beta9 serve app.py:function`                 |
