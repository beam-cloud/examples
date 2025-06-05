from beam.integrations import VLLM, VLLMArgs
from beam import Image

INTERNVL3_AWQ = "OpenGVLab/InternVL3-8B-AWQ"
YI_CODER_CHAT = "01-ai/Yi-Coder-9B-Chat"
MISTRAL_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.3"
DEEPSEEK_R1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

internvl = VLLM(
    name=INTERNVL3_AWQ.split("/")[-1],
    cpu=4,
    memory="16Gi",
    gpu="A10G",
    gpu_count=1,
    image=(Image(python_version="python3.12")).add_python_packages(
        ["vllm==0.6.4.post1"]
    ),
    vllm_args=VLLMArgs(
        model=INTERNVL3_AWQ,
        served_model_name=[INTERNVL3_AWQ],
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        limit_mm_per_prompt={"image": 2},
        quantization="awq",
    ),
)

yicoder_chat = VLLM(
    name=YI_CODER_CHAT.split("/")[-1],
    cpu=8,
    memory="16Gi",
    gpu="A100-40",
    vllm_args=VLLMArgs(
        model=YI_CODER_CHAT,
        served_model_name=[YI_CODER_CHAT],
        task="chat",
        trust_remote_code=True,
        max_model_len=8096,
    ),
)

mistral_instruct = VLLM(
    name=MISTRAL_INSTRUCT.split("/")[-1],
    cpu=8,
    memory="16Gi",
    gpu="A100-40",
    secrets=["HF_TOKEN"],
    vllm_args=VLLMArgs(
        model=MISTRAL_INSTRUCT,
        served_model_name=[MISTRAL_INSTRUCT],
        chat_template="./tool_chat_template_mistral.jinja",
        enable_auto_tool_choice=True,
        tool_call_parser="mistral",
    ),
)

deepseek_r1 = VLLM(
    name=DEEPSEEK_R1.split("/")[-1],
    cpu=8,
    memory="32Gi",
    gpu="A10G",
    gpu_count=2,
    vllm_args=VLLMArgs(
        model=DEEPSEEK_R1,
        served_model_name=[DEEPSEEK_R1],
        task="generate",
        trust_remote_code=True,
        max_model_len=8096,
    ),
)
