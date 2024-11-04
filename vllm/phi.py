from beam.abstractions.integrations import VLLM, VLLMEngineConfig

MODEL_NAME = "microsoft/Phi-3.5-vision-instruct"

phi = VLLM(
    name="phi-vllm",
    cpu=8,
    memory="16Gi",
    gpu="A100-40",
    vllm_engine_config=VLLMEngineConfig(
        model=MODEL_NAME,
        served_model_name=[MODEL_NAME],
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 2},
    ),
)
