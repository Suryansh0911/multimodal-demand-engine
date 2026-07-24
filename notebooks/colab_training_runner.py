import os
import subprocess

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

print("=== 1. Environment Setup ===")
IN_COLAB = False
try:
    from google.colab import drive
    IN_COLAB = True
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    CHECKPOINT_DIR = "/content/drive/MyDrive/models/checkpoints/"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print("Google Drive mounted successfully.")
except Exception:
    print("Not running in Google Colab (or Drive not mounted). Using local checkpoint directory.")
    CHECKPOINT_DIR = "models/checkpoints/"

print("=== 2. Repository Setup ===")
repo_url = "https://github.com/Suryansh0911/multimodal-demand-engine.git"

if os.path.basename(os.getcwd()) != "multimodal-demand-engine":
    if not os.path.exists("multimodal-demand-engine"):
        run_cmd(f"git clone {repo_url}")
    os.chdir("multimodal-demand-engine")

print("=== 3. Installing Dependencies ===")
run_cmd("pip install -r requirements.txt")

print("=== 4. Updating Configuration ===")
import yaml
with open("configs/train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

config['training']['model_dir'] = CHECKPOINT_DIR
config['training']['epochs'] = 20
config['dataset']['batch_size'] = 32

with open("configs/train_config.yaml", "w") as f:
    yaml.dump(config, f)

print("=== 5. Launching Distributed Training Pipeline ===")
run_cmd("python -m src.train")
print(f"Training complete! Best weights saved to: {CHECKPOINT_DIR}")