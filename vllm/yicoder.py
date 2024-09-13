from beta9 import Image, Volume, asgi, env

# These imports are only available in the remote environment
if env.is_remote():
    import asyncio

    import fastapi

    import vllm.entrypoints.openai.api_server as api_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.logger import RequestLogger
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import (
        OpenAIServingCompletion,
    )
    from vllm.usage.usage_lib import UsageContext

MODEL_NAME = "01-ai/Yi-Coder-9B-Chat"

# This beam volume is mounted as a file system and used to cache the downloaded model
vllm_cache = Volume(name="yicoder", mount_path="./yicoder")


@asgi(
    image=Image().add_python_packages(["vllm"]),
    volumes=[vllm_cache],
    gpu="A100-40",
    memory="8Gi",
    cpu=1,
    keep_warm_seconds=360,
)
def yicoder_server():
    app = fastapi.FastAPI(
        title=f"{MODEL_NAME} server",
        docs_url="/docs",
    )

    # Health check is required as it will be checked during setup for vllm
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    app.include_router(api_server.router)

    # Create the engine client
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        max_model_len=8096,
        download_dir=vllm_cache.mount_path,
    )
    async_engine_client = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    model_config = asyncio.run(async_engine_client.get_model_config())

    # Optionally setup a request logger
    request_logger = RequestLogger(max_log_len=2048)

    # Setup the OpenAI serving chat and completion endpoints
    api_server.openai_serving_chat = OpenAIServingChat(
        async_engine_client,
        model_config=model_config,
        served_model_names=[MODEL_NAME],
        chat_template=None,
        lora_modules=[],
        prompt_adapters=[],
        response_role="assistant",
        request_logger=request_logger,
    )

    api_server.openai_serving_completion = OpenAIServingCompletion(
        async_engine_client,
        model_config=model_config,
        served_model_names=[MODEL_NAME],
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )

    return app
