from beam import endpoint, Image

MODEL_NAME = "OpenGVLab/InternVL2_5-8B"

vllm_image = Image(python_version="python3.10", python_packages=["vllm==0.6.3.post1", "fastapi[standard]==0.115.4"])

@endpoint(
    name="internvl2_5-1b",
    image=vllm_image,
    cpu=4,
    memory="32Gi",
    gpu="A10G",
    gpu_count=2,
)
def generate():
    import fastapi
    import vllm.entrypoints.openai.api_server as api_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.logger import RequestLogger
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import (
        OpenAIServingCompletion,
    )
    from vllm.entrypoints.openai.serving_engine import BaseModelPath
    from vllm.usage.usage_lib import UsageContext

    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible server",
        description="Run an OpenAI-compatible LLM server with vLLM on Beam",
        version="0.0.1",
        docs_url="/docs",
    )

    router = fastapi.APIRouter()
    router.include_router(api_server.router)
    web_app.include_router(router)

    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        max_model_len=8096,
        trust_remote_code=True,
        enforce_eager=False,  
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    model_config = engine.get_model_config()

    request_logger = RequestLogger(max_log_len=2048)

    base_model_paths = [
        BaseModelPath(name=MODEL_NAME.split("/")[1], model_path=MODEL_NAME)
    ]

    api_server.chat = OpenAIServingChat(
        engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        chat_template=None,
        response_role="assistant",
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )
    api_server.completion = OpenAIServingCompletion(
        engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )

    return web_app
 
def get_model_config(engine):
    import asyncio

    try:  # adapted from vLLM source -- https://github.com/vllm-project/vllm/blob/507ef787d85dec24490069ffceacbd6b161f4f72/vllm/entrypoints/openai/api_server.py#L235C1-L247C1
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    return model_config    