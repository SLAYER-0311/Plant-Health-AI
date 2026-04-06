"""
Upload model to Hugging Face Hub.
Run once locally: python scripts/upload_model_to_hf.py
"""
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_REPO_ID = "jsohamg/plant-health-ai-model"   # your HF username/repo-name
MODEL_PATH = Path("backend/models/plant_disease_model.pth")
CLASS_NAMES_PATH = Path("backend/models/class_names.json")

api = HfApi()

# Create the model repo (won't error if it already exists)
print(f"Creating repo: {HF_REPO_ID}")
create_repo(HF_REPO_ID, repo_type="model", exist_ok=True, private=False)

# Upload model weights
print(f"Uploading {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)...")
api.upload_file(
    path_or_fileobj=str(MODEL_PATH),
    path_in_repo="plant_disease_model.pth",
    repo_id=HF_REPO_ID,
    repo_type="model",
)

# Upload class names
print(f"Uploading {CLASS_NAMES_PATH}...")
api.upload_file(
    path_or_fileobj=str(CLASS_NAMES_PATH),
    path_in_repo="class_names.json",
    repo_id=HF_REPO_ID,
    repo_type="model",
)

print(f"\n✅ Done! Model hosted at: https://huggingface.co/{HF_REPO_ID}")
print("Update HF_MODEL_REPO in streamlit_app.py if you used a different repo name.")
