from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import load_from_disk
import torch

# Auto-load processed data (no manual upload needed)
dataset = load_from_disk("data/processed/")

# Initialize model (Mistral 7B by default)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# LoRA setup
peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

# Training code (same as before)
# ...
