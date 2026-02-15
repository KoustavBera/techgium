
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# Configuration
MODEL_NAME = "BioMistral/BioMistral-7B"
DATASET_NAME = "lavita/ChatDoctor-HealthCareMagic-100k"
NEW_MODEL_NAME = "BioMistral-7B-ChatDoctor-FinteTuned"
OUTPUT_DIR = "./results"

def format_instruction(sample):
    """
    Format the dataset into a prompt structure.
    Adjust this based on the actual column names of the dataset.
    Commonly: 'instruction', 'input', 'output' or similar.
    """
    # Check dataset columns to ensure correct keys
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')
    output = sample.get('output', '')
    
    # Construct prompt
    if input_text:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return {"text": text}

def main():
    print(f"Loading model: {MODEL_NAME}")
    
    # QLoRA Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Dataset
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")

    # Format dataset BEFORE training (memory efficient)
    print("Formatting dataset...")
    def process_data(sample):
        # Check keys (handles potential variations)
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')
        
        # Construct prompt
        if input_text:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        return {"text": text}

    # Map the dataset to create the 'text' column
    dataset = dataset.map(process_data, remove_columns=dataset.column_names)
    
    # Shuffle and trim if needed to fit in memory (optional, but good for stability)
    # dataset = dataset.shuffle(seed=42).select(range(10000)) # Uncomment to train on a smaller subset if OOM persists

    # LoRA Configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=16, # Reduced rank could also save memory if needed (try 32 or 16 if OOM)
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,  # EXTREMELY LOW for 6GB VRAM
        gradient_accumulation_steps=4,  # Accumulate to simulate larger batch (4 * 1 = 4)
        optim="paged_adamw_8bit",       # Use 8-bit optimizer to save memory
        save_steps=250,
        logging_steps=25,
        learning_rate=1e-4,
        weight_decay=0.001,
        fp16=True,                      # Use fp16 (faster/less memory)
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        gradient_checkpointing=True,    # Crucial for saving memory (trades compute for memory)
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text", # Now we use the pre-computed 'text' column
        max_seq_length=512, # Limited to 512 for 6GB VRAM.
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.model.save_pretrained(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME)
    print(f"Model saved to {NEW_MODEL_NAME}")

if __name__ == "__main__":
    main()
