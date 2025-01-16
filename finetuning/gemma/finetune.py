from beam import Volume, Image, function


# The mount path is the location on the beam volume that we will access.
MOUNT_PATH = "./gemma-ft"
WEIGHT_PATH = "./gemma-ft/weights"
OPEN_ASSISTANT_DATASET_PATH = "./gemma-ft/data/oa.jsonl"


@function(
    volumes=[Volume(name="gemma-ft", mount_path=MOUNT_PATH)],
    image=Image().add_python_packages(
        ["transformers", "torch", "datasets", "peft", "bitsandbytes"]
    ),
    gpu="A100-40",
    cpu=4,
)
def gemma_fine_tune():
    import os
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not torch.cuda.is_available():
        return "CUDA is not available"

    torch.set_float32_matmul_precision("high")

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHT_PATH, device_map="auto", attn_implementation="eager", use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(WEIGHT_PATH)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    dataset = load_dataset("json", data_files=OPEN_ASSISTANT_DATASET_PATH)

    def prepare_dataset(examples):
        conversations = examples["text"]
        tokenized = tokenizer(
            conversations, truncation=True, padding="max_length", max_length=512
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        for i, labels in enumerate(tokenized["labels"]):
            tokenized["labels"][i] = [-100] + labels[
                :-1
            ]  # -100 is the ignore index for CrossEntropyLoss
        return tokenized

    tokenized_dataset = dataset.map(
        prepare_dataset, batched=True, remove_columns=dataset["train"].column_names
    )

    training_args = TrainingArguments(
        # This output directory is on our mounted volume
        output_dir="./gemma-ft/gemma-2b-finetuned",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        gradient_checkpointing=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    # Saving the LORA model and tokenizer to our mounted volume so that our inference endpoint can access it.
    model.save_pretrained("./gemma-ft/gemma-2b-finetuned")
    tokenizer.save_pretrained("./gemma-ft/gemma-2b-finetuned")


if __name__ == "__main__":
    gemma_fine_tune.remote()
