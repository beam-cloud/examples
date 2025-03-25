from beam import function, Image, Volume, env

if env.is_remote():
    import torch
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from datasets import load_dataset
    import os

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
MAX_SEQ_LENGTH = 2048
VOLUME_PATH = "./model_storage"
TRAIN_CONFIG = {
    "batch_size": 2,
    "grad_accumulation": 4,
    "max_steps": 60,
    "learning_rate": 2e-4,
    "seed": 3407,
}

image = (
    Image(python_version="python3.11")
    .add_python_packages(
        [
            "ninja",
            "packaging",
            "wheel",
            "torch",
            "xformers",
            "trl",
            "peft",
            "accelerate",
            "bitsandbytes",
        ]
    )
    .add_commands(
        [
            "pip uninstall unsloth -y",
            'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"',
        ]
    )
)


@function(
    name="unsloth-fine-tune",
    cpu=12,
    memory="32Gi",
    gpu="A100-40",
    image=image,
    volumes=[Volume(name="model-storage", mount_path=VOLUME_PATH)],
    timeout=-1,
)
def fine_tune_model():
    output_dir = os.path.join(VOLUME_PATH, "fine_tuned_model")
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True
    )

    def format_alpaca_prompt(instruction, input_text, output):
        template = (
            "Below is an instruction that describes a task, paired with an input that "
            "provides further context. Write a response that appropriately completes the request.\n"
            "### Instruction:\n{}\n### Input:\n{}\n### Response:\n{}"
        )
        return template.format(instruction, input_text, output) + tokenizer.eos_token

    def format_dataset(examples):
        texts = [
            format_alpaca_prompt(instruction, input_text, output)
            for instruction, input_text, output in zip(
                examples["instruction"], examples["input"], examples["output"]
            )
        ]
        return {"text": texts}

    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset = dataset.map(format_dataset, batched=True)

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
        random_state=TRAIN_CONFIG["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=TRAIN_CONFIG["batch_size"],
            gradient_accumulation_steps=TRAIN_CONFIG["grad_accumulation"],
            max_steps=TRAIN_CONFIG["max_steps"],
            learning_rate=TRAIN_CONFIG["learning_rate"],
            fp16=False,
            bf16=True,
            logging_steps=1,
            output_dir=output_dir,
            seed=TRAIN_CONFIG["seed"],
        ),
    )

    with torch.autograd.set_detect_anomaly(True):
        trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "status": "success",
        "message": "Fine-tuning complete",
        "model_path": output_dir,
    }

if __name__ == "__main__":
    fine_tune_model()