# AI-Preference-Optimization
AI Preference Optimization for Healthcare - centered around DPO and KTO training

## Getting Started
 - Skip ahead to Longleaf if using that
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
MacOS Instructions:
```
pip3 install datasets
pip3 install trl
pip3 install transformers
```
 - If using CUDA GPU, install that as well, make sure you have the correct CUDA toolkit also installed prior. Replace `cu118` with your current version.
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

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
train_dataset = load_dataset("kpeng-05/dpo_dataset", split="train")

training_args = DPOConfig(output_dir="Qwen2.5-0.5B-DPO", logging_steps=10)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```
 - Train the model with the following command:
```
accelerate launch train_dpo.py
```
 - Note that if you are using a computer without a GPU, this training may take several hours
 - Training with RTX 4060 Ti 16GB took ~2 minutes
 - You can test the trained model with the following command (you do not need to have trained a model to do this, just make sure you have the correct libraries installed):
```
trl chat --model_name_or_path kpeng-05/Qwen2.5-0.5B-DPO
```

## KTO Training
 - Original instructions are found at https://huggingface.co/docs/trl/main/en/kto_trainer

 - Cleaning Dataset (Mihika's Code Below)
```
import pandas as pd

kto = pd.read_csv(<filepath>)

kto['Age'] = kto['Age'].astype(str)

kto['Prompt'] = kto['Age'] + "_" + kto['CancerStage']

kto['Completion']  = kto['Surgical Intervention_Thumbs'].str.cat(kto['Chemotherapy_Thumbs'], '_').str.cat(kto['Radiation Therapy_Thumbs'], '_').str.cat(kto['Palliative Care_Thumbs'], '_').str.cat(kto['Complementary Medicine_Thumbs'], '_')

kto = kto.drop(columns = ['Age', 'CancerStage', 'Surgical Intervention_Thumbs', 'Chemotherapy_Thumbs', 'Radiation Therapy_Thumbs', 'Palliative Care_Thumbs', 'Complementary Medicine_Thumbs'])

kto.head(5)
```
# Longleaf Login & Setup
Request access to Longleaf via the UNC Research Computing website. After receiving access, log into the account and navigate to Clusters, and then Longleaf Cluster Access.
 - [Set up a conda environment](https://help.rc.unc.edu/python-packages/#conda) and install the listed dependencies in the Getting Started section (datasets, transformers, trl), documentation on how to activate the environment is also listed
 - Use `module load cuda/12.9` before attempting to download torch (or you can just download the CUDA files, up to you)
 - Run the SLURM script (example below):
```
#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=20g
#SBATCH -t 2-
#SBATCH -p a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=<onyen>@email.unc.edu

module purge
## Replace 12.9 with whatever version you created the environment with
module load cuda/12.9
module load anaconda
conda activate <env_name>
python myscript.py
