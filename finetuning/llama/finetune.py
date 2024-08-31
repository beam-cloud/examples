# finetune.py
# Deploy to beam by running `$ python finetune.py` in the terminal# finetune.py
from beam import Volume, Image, function, env

# The mount path is the location on the beam volume that we will access. 
MOUNT_PATH = "./llama-ft"
WEIGHT_PATH = "meta-llama/Meta-Llama-3.1-8B"

@function(
    secrets=["HF_TOKEN"],
    volumes=[Volume(name="llama-ft", mount_path=MOUNT_PATH)],
    image=Image(
        python_packages=["transformers", "torch", "datasets", "peft", "bitsandbytes"]
    ),
    gpu="A100-40",
    cpu=4,
)
def llama_fine_tune():
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

    # Load the Llama3 model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHT_PATH, device_map="auto", attn_implementation="eager", use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(WEIGHT_PATH, use_fast=False)
    
    # Set the pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token


    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Load the Yelp Reviews dataset from Hugging Face
    dataset = load_dataset("yelp_review_full")

    def prepare_dataset(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(prepare_dataset, batched=True)

    training_args = TrainingArguments(
        # This output directory is on our mounted volume
        output_dir="./llama-ft/llama-finetuned",
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
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    # Saving the LORA model and tokenizer to our mounted volume so that our inference endpoint can access it.
    model.save_pretrained("./llama-ft/llama-finetuned")
    tokenizer.save_pretrained("./llama-ft/llama-finetuned")


if __name__ == "__main__":\
    llama_fine_tune.remote()
