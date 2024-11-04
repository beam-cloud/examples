from beam.abstractions.integrations import VLLM, VLLMEngineConfig

MODEL_NAME = "01-ai/Yi-Coder-9B-Chat"

yicoder = VLLM(
    name="yicoder-vllm",
    cpu=8,
    memory="16Gi",
    gpu="A100-40",
    vllm_engine_config=VLLMEngineConfig(
        model=MODEL_NAME,
        served_model_name=[MODEL_NAME],
        task="chat",
        trust_remote_code=True,
        max_model_len=8096,
    ),
)
