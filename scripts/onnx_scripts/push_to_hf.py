
import os
import argparse
from huggingface_hub import HfApi

MODELS = {
    "zipa-t-small-300k": "anyspeech/zipa-small-noncausal-300k",
    "zipa-t-large-300k": "anyspeech/zipa-large-noncausal-300k",
    "zipa-t-small-500k": "anyspeech/zipa-small-noncausal-500k",
    "zipa-t-large-500k": "anyspeech/zipa-large-noncausal-500k",
    "zipa-cr-small-300k": "anyspeech/zipa-small-crctc-300k",
    "zipa-cr-large-300k": "anyspeech/zipa-large-crctc-300k",
    "zipa-cr-small-500k": "anyspeech/zipa-small-crctc-500k",
    "zipa-cr-large-500k": "anyspeech/zipa-large-crctc-500k",
    "zipa-cr-ns-small-700k": "anyspeech/zipa-small-crctc-ns-700k",
    "zipa-cr-ns-large-800k": "anyspeech/zipa-large-crctc-ns-800k",
    "zipa-cr-ns-small-nodiacritics-700k": "anyspeech/zipa-small-crctc-ns-no-diacritics-700k",
    "zipa-cr-ns-large-nodiacritics-780k": "anyspeech/zipa-large-crctc-ns-no-diacritics-780k",
}

README_TEMPLATE = """
# ONNX Models for ZIPA

This repository contains optimized ONNX models for the ZIPA speech recognition system. 
The models are exported from the original PyTorch checkpoints and support FP32, FP16, and INT8 precision.

## Original Repository
[GitHub: lingjzhu/zipa](https://github.com/lingjzhu/zipa)

## Available Files
- `model.onnx`: FP32 ONNX model (or `encoder-*.onnx`, etc. for Transducer)
- `model.fp16.onnx`: FP16 Quantized model
- `model.int8.onnx`: INT8 Quantized model
- `tokens.txt`: Token vocabulary

## Usage

### Installation
```bash
pip install onnxruntime soundfile librosa lhotse
```

### Inference
Please refer to the [Inference Documentation](https://github.com/lingjzhu/zipa/tree/main/inference) in the official repository for detailed usage and scripts.

#### Quick Example (CTC)
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.fp16.onnx")
# ... preprocessing ...
```
"""

def main():
    parser = argparse.ArgumentParser(description="Push ONNX models to Hugging Face Hub")
    parser.add_argument("--token", help="Hugging Face API token (optional, can be inferred from env)")
    args = parser.parse_args()
    

    # Assuming script is run from project root, or handle relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # If run from root, checkpoints is processcwd() + "checkpoints"
    # If run from onnx_scripts/, checkpoints is ../checkpoints
    
    # Let's assume run from root as standard.
    if os.path.isdir("checkpoints"):
        base_checkpoints_dir = "checkpoints"
        tokens_file = "ipa_simplified/tokens.txt"
    elif os.path.isdir("../checkpoints"):
         base_checkpoints_dir = "../checkpoints"
         tokens_file = "../ipa_simplified/tokens.txt"
    else:
         print("Could not find 'checkpoints' directory. Please run from project root or onnx_scripts/.")
         return
    
    api = HfApi(token=args.token)
    
    for model_name, repo_id in MODELS.items():
        print(f"Processing {model_name} -> {repo_id}...")
        
        exp_dir = os.path.join(base_checkpoints_dir, model_name, "exp")
        if not os.path.exists(exp_dir):
            print(f"  Directory {exp_dir} not found. Skipping.")
            continue
            
        # 1. Identify files to upload
        files_to_upload = []
        
        # ONNX files
        for f in os.listdir(exp_dir):
            if f.endswith(".onnx"):
                files_to_upload.append(os.path.join(exp_dir, f))
                
        if not files_to_upload:
            print(f"  No ONNX files found in {exp_dir}. Skipping.")
            continue
            
        # Tokens file
        if os.path.exists(tokens_file):
             # We upload it as top-level tokens.txt
             pass 
        else:
             print(f"  Warning: {tokens_file} not found.")

        # 2. Upload Files
        print(f"  Uploading {len(files_to_upload)} ONNX files...")
        for file_path in files_to_upload:
            filename = os.path.basename(file_path)
            print(f"    Uploading {filename}...")
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename, # Root of repo
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Add ONNX model: {filename}"
                )
            except Exception as e:
                print(f"    Error uploading {filename}: {e}")
                
        # Upload tokens
        if os.path.exists(tokens_file):
            print("    Uploading tokens.txt...")
            try:
                api.upload_file(
                    path_or_fileobj=tokens_file,
                    path_in_repo="tokens.txt",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message="Add tokens.txt"
                )
            except Exception as e:
                print(f"    Error uploading tokens.txt: {e}")

        # 3. Upload README_ONNX.md
        print("    Uploading README_ONNX.md...")
        # Create temporary file
        readme_path = os.path.join(exp_dir, "README_ONNX.md")
        with open(readme_path, "w") as f:
            f.write(README_TEMPLATE)
            
        try:
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README_ONNX.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add ONNX documentation"
            )
        except Exception as e:
             print(f"    Error uploading README_ONNX.md: {e}")
             
        # Cleanup temp file
        if os.path.exists(readme_path):
            os.remove(readme_path)
            
    print("All uploads complete.")

if __name__ == "__main__":
    main()
