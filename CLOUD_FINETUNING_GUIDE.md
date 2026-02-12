# Cloud Fine-Tuning Guide (No Downloads Required)

## The Problem with BioMistral-7B
❌ BioMistral-7B is **14GB** - not available on serverless Inference API  
❌ Requires dedicated GPU to run  

## Better Alternative: Fine-Tune Mistral-7B on Medical Data
✅ Use **Hugging Face AutoTrain** - trains in the cloud  
✅ You get a medical-specialized model  
✅ No local GPU needed  
✅ Cost: ~$5-15 for full fine-tune  

---

## Option 1: Hugging Face AutoTrain (Easiest) ⭐

### Step 1: Prepare Your Dataset
Create a CSV file with this format:

```csv
instruction,input,output
"Patient has fever and headache","What should I do?","You may have viral fever. Take Paracetamol 500mg, rest, drink fluids. See doctor if >3 days."
"Mujhe pet mein dard hai","","Pet dard acidity se ho sakta hai. Digene lein, halka khana khayein."
```

**Download medical datasets:**
- [MedDialog](https://huggingface.co/datasets/medical_dialog) - 3.4M dialogues
- [ChatDoctor](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) - 100K Q&A

### Step 2: Create AutoTrain Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Select **Docker** → **AutoTrain**
4. Name it: `medical-chatbot-train`
5. Select **CPU** (for setup) - we'll use GPU later

### Step 3: Configure Training

In AutoTrain UI:
```yaml
Task: LLM SFT (Supervised Fine-Tuning)
Base Model: mistralai/Mistral-7B-Instruct-v0.2
Training Data: Upload your CSV
Text Column: instruction + input + output

# Training Config
use_peft: true            # LoRA - efficient fine-tuning
quantization: int4        # 4-bit - reduces memory
epochs: 3
learning_rate: 2e-4
batch_size: 2
```

### Step 4: Start Training

1. Click **"Start Training"**
2. Select **T4 GPU** (~$0.60/hr) or **A10G** (~$1.50/hr)
3. Training takes **2-4 hours** for 10K examples

### Step 5: Use Your Fine-Tuned Model

After training, model is pushed to your Hub:
```javascript
// Update in MedicalChatService.js
this.models = {
    medical: "YOUR_USERNAME/medical-mistral-7b",  // Your model!
    translate: "CohereLabs/command-a-translate-08-2025:cohere"
};
```

---

## Option 2: Google Colab (Free GPU)

### Colab Notebook Setup
```python
# Install dependencies
!pip install transformers peft trl datasets accelerate bitsandbytes

# Login to Hugging Face
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")

# Load dataset
from datasets import load_dataset
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")

# Load base model with 4-bit quantization
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Setup LoRA
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Train
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=TrainingArguments(
        output_dir="./medical-mistral",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100
    ),
    dataset_text_field="text",
    max_seq_length=512
)

trainer.train()

# Push to Hub
model.push_to_hub("YOUR_USERNAME/medical-mistral-7b")
tokenizer.push_to_hub("YOUR_USERNAME/medical-mistral-7b")
```

---

## Option 3: Together AI Fine-Tuning

[Together.ai](https://together.ai) offers hosted fine-tuning:

1. Upload dataset to Together
2. Select base model (Mistral, Llama)
3. They handle training
4. You get API endpoint

**Cost**: ~$0.0008/token → ~$20 for 10K examples

---

## Datasets to Use

| Dataset | Size | Best For |
|---------|------|----------|
| ChatDoctor-100k | 100K | General medical Q&A |
| MedDialog | 3.4M | Conversation flow |
| HealthCareMagic | 200K | Symptom-based advice |
| MedQA | 60K | Medical reasoning |

---

## Cost Comparison

| Platform | GPU | Time | Cost |
|----------|-----|------|------|
| Google Colab | T4 (free) | 4-6 hrs | **$0** |
| Colab Pro | A100 | 1-2 hrs | $10/mo |
| AutoTrain | T4 | 3-4 hrs | ~$5-10 |
| RunPod | A40 | 1-2 hrs | ~$8 |
| Lambda Labs | A10 | 2-3 hrs | ~$5 |

---

## Recommended Approach

```
Week 1: Use current setup + few-shot examples
Week 2: Fine-tune on Colab (free) with 10K examples
Week 3: Test and iterate
```

**Start with ChatDoctor-100k dataset** - it's already formatted as Q&A conversations.
