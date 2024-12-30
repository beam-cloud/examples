from beam.integrations import VLLM, VLLMArgs
from beam import Image

INTERNVL2_5 = "OpenGVLab/InternVL2_5-8B"
YI_CODER_CHAT = "01-ai/Yi-Coder-9B-Chat"
MISTRAL_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.3"

internvl = VLLM(
    name=INTERNVL2_5.split("/")[-1],
    cpu=8,
    memory="32Gi",
    gpu="A10G",
    gpu_count=2,
    image=(Image(python_version="python3.12", python_packages=["vllm==0.6.4.post1"])),
    vllm_args=VLLMArgs(
        model=INTERNVL2_5,
        served_model_name=[INTERNVL2_5],
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        limit_mm_per_prompt={"image": 2},
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
