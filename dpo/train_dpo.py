# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "kpeng-05/dpo_dataset"
OUTPUT_DIR = "Qwen2.5-0.5B-DPO"

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
train_dataset = load_dataset(DATASET_PATH, split="train")

training_args = DPOConfig(output_dir=OUTPUT_DIR, logging_steps=10)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()