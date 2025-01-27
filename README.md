# AI-Preference-Optimization
AI Preference Optimization for Healthcare - centered around DPO and KTO training

## Getting Started
 - Start creating a Python project environment, I recommend a Conda environment
 - Miniconda install - Windows Powershell, run these commands separately
```
wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -outfile ".\miniconda.exe"
```
```
Start-Process -FilePath ".\miniconda.exe" -ArgumentList "/AddToPath=1 /S" -Wait
```
```
del .\miniconda.exe
```
 - Restart the shell, navigate to the file location where you intend to work, and create a conda environment:
```
conda create --name environment-name
```
 - Replace environment-name with your intended environment name
 - Install Huggingface dependencies
```
pip install datasets
pip install trl
pip install transformers
```
 - If using CUDA GPU, install that as well, make sure you have the correct CUDA toolkit also installed prior:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## DPO Training
 - Original instructions are found at https://huggingface.co/docs/trl/main/en/dpo_trainer
 - By utilizing the following Python script to run the DPO training, please make sure you have Python installed:
```
# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("kpeng-05/dpo_dataset", split="train")

training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO", logging_steps=10)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```
 - Train the model with the following command:
```
accelerate launch train_dpo.py
```
 - Note that if you are using a computer without a GPU, this training may take several hours
