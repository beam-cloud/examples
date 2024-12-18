from beam import Image, endpoint, env, Volume
from pydantic import BaseModel
from uuid import uuid4
import requests

PATH = "meta-llama/Llama-3.2-11B-Vision-Instruct"
TOKENIZER = "meta-llama/Meta-Llama-3.1-8B-Instruct",
CHAT_TEMPLATE = "llama_3_vision"

VOLUME_PATH = "./llama3.2",
GPU_COUNT = 1,
LOG_LEVEL = "error"

class ImageRequest(BaseModel):
    image_url: str = "https://images.squarespace-cdn.com/content/v1/5c7f5f60797f746a7d769cab/1708063049157-NMFAB7KBRBY2IG2BWP4E/the+golden+gate+bridge+san+francisco.jpg"
    question: str = "What is this?"

image = (
    Image(python_version="python3.11")
    .add_python_packages([
        "sglang[all]==0.1.17",
        "transformers==4.40.2",
        "numpy<2",
        "fastapi[standard]==0.115.4",
        "pydantic==2.9.2",
        "starlette==0.41.2",
    ])
)


def init_model():
    if env.is_remote():
        from huggingface_hub import snapshot_download
        import transformers

        snapshot_download(
            PATH,
            ignore_patterns=["*.pt", "*.bin"],
            cache_dir=VOLUME_PATH
        )
        transformers.utils.move_cache()

@endpoint(
    name="sglang",
    image=image,
    on_start=init_model,
    cpu=2,
    memory="32Gi",
    gpu="A100-40",
    volumes=[Volume(name="llama-vision", mount_path=VOLUME_PATH)],
)
def generate():
    import sglang as sgl

    runtime = sgl.Runtime(
        model_path=PATH,
        tokenizer_path=TOKENIZER,
        tp_size=GPU_COUNT,
        log_level=LOG_LEVEL
    )
    runtime.endpoint.chat_template = sgl.lang.chat_template.get_chat_template(
        CHAT_TEMPLATE
    )
    sgl.set_default_backend(runtime)

    async def process_image(image_url: str) -> str:
        response = requests.get(image_url)
        response.raise_for_status()

        image_filename = f"/tmp/{uuid4()}-{image_url.split('/')[-1]}"
        with open(image_filename, "wb") as file:
            file.write(response.content)
        return image_filename

    @sgl.function
    def image_qa(s, image_path: str, question: str):
        s += sgl.user(sgl.image(image_path) + question)
        s += sgl.assistant(sgl.gen("answer"))

    image_path = process_image("https://images.squarespace-cdn.com/content/v1/5c7f5f60797f746a7d769cab/1708063049157-NMFAB7KBRBY2IG2BWP4E/the+golden+gate+bridge+san+francisco.jpg")
    state = image_qa.run(
            image_path=image_path,
            question="What is this?",
            max_new_tokens=128
        )
    return {"answer": state["answer"]}