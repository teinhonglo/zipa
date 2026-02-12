
import os
import torch
from huggingface_hub import hf_hub_download, list_repo_files

MODELS = {
    "zipa-t-small-300k": "anyspeech/zipa-small-noncausal-300k",
    "zipa-t-large-300k": "anyspeech/zipa-large-noncausal-300k", # Inferred from context, user check required if fails
    "zipa-t-small-500k": "anyspeech/zipa-small-noncausal-500k",
    "zipa-t-large-500k": "anyspeech/zipa-large-noncausal-500k", # Inferred
    "zipa-cr-small-300k": "anyspeech/zipa-small-crctc-300k",
    "zipa-cr-large-300k": "anyspeech/zipa-large-crctc-300k",
    "zipa-cr-small-500k": "anyspeech/zipa-small-crctc-500k",
    "zipa-cr-large-500k": "anyspeech/zipa-large-crctc-500k",
    "zipa-cr-ns-small-700k": "anyspeech/zipa-small-crctc-ns-700k",
    "zipa-cr-ns-large-800k": "anyspeech/zipa-large-crctc-ns-800k",
    "zipa-cr-ns-small-nodiacritics-700k": "anyspeech/zipa-small-crctc-ns-no-diacritics-700k",
    "zipa-cr-ns-large-nodiacritics-780k": "anyspeech/zipa-large-crctc-ns-no-diacritics-780k",
}

OUTPUT_DIR = "checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_averaged_checkpoint(repo_id):
    try:
        files = list_repo_files(repo_id)
        # Priority: ends with _avg10.pth or similar
        candidates = [f for f in files if "avg" in f and f.endswith(".pth")]
        if candidates:
             # Sort candidates to ensure we get the "best" one if there are multiple?
             # Probably just picking one is fine as usually there is one final averaged one.
             return candidates[0]
        return None
    except Exception as e:
        print(f"Error accessing {repo_id}: {e}")
        return None

for name, repo_id in MODELS.items():
    print(f"Processing {name} from {repo_id}...")
    
    checkpoint_file = find_averaged_checkpoint(repo_id)
    if not checkpoint_file:
        print(f"Warning: No averaged checkpoint found for {name} in {repo_id}. Skipping.")
        continue
        
    print(f"Found checkpoint: {checkpoint_file}")
    
    try:
        path = hf_hub_download(repo_id=repo_id, filename=checkpoint_file)
        
        # Determine wrapping folder structure: checkpoints/<name>/exp/epoch-999.pt
        model_dir = os.path.join(OUTPUT_DIR, name)
        exp_dir = os.path.join(model_dir, "exp")
        os.makedirs(exp_dir, exist_ok=True)
        
        output_path = os.path.join(exp_dir, "epoch-999.pt")
        if os.path.exists(output_path):
            print(f"  Already exists at {output_path}, skipping download/wrap.")
            continue

        # Wrap and save
        print(f"  Wrapping and saving to {output_path}...")
        checkpoint = torch.load(path, map_location="cpu")
        wrapped = {"model": checkpoint}
        torch.save(wrapped, output_path)
    except Exception as e:
        print(f"Error downloading/wrapping {name}: {e}")

print("Done.")
