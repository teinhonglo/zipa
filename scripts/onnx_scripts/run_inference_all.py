
import os
import subprocess
import glob

def main():
    checkpoints_dir = "checkpoints"
    sample_audio = "sample.wav"
    inference_script = "onnx_scripts/inference_example.py"
    
    if not os.path.exists(sample_audio):
        print(f"Error: {sample_audio} not found.")
        return

    # Find all model directories
    # Structure: checkpoints/<model_name>/exp/
    
    models = []
    if os.path.exists(checkpoints_dir):
        for model_name in os.listdir(checkpoints_dir):
            model_path = os.path.join(checkpoints_dir, model_name)
            if os.path.isdir(model_path):
                exp_dir = os.path.join(model_path, "exp")
                if os.path.exists(exp_dir):
                    models.append(model_name)
    
    models.sort()
    
    print(f"Found {len(models)} models.")
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Running inference for: {model}")
        print(f"{'='*60}")
        
        # Determine type
        model_type = "ctc"
        if "zipa-t-" in model: # heuristic based on naming "zipa-t-..." vs "zipa-cr-..."
             model_type = "transducer"
        
        precisions = ["fp32", "fp16", "int8"]
        
        for prec in precisions:
            print(f"  [{prec}]")
            
            model_path_arg = ""
            search_suffix_arg = ".onnx"
            
            if model_type == "ctc":
                filename = "model.onnx"
                if prec == "fp16": filename = "model.fp16.onnx"
                elif prec == "int8": filename = "model.int8.onnx"
                
                model_path_arg = os.path.join(checkpoints_dir, model, "exp", filename)
                if not os.path.exists(model_path_arg):
                    print(f"    Skipping {prec}: {filename} not found.")
                    continue
            else:
                # Transducer: pass directory, use search suffix
                model_path_arg = os.path.join(checkpoints_dir, model, "exp") 
                if prec == "fp16": search_suffix_arg = ".fp16.onnx"
                elif prec == "int8": search_suffix_arg = ".int8.onnx"
                
            cmd = [
                "/home/slime-base/anaconda3/envs/zipa_export/bin/python",
                inference_script,
                sample_audio,
                "--model-path", model_path_arg,
                "--model-type", model_type
            ]
            
            if model_type == "transducer":
                cmd.extend(["--search-suffix", search_suffix_arg])
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    # Parse out the Predicted Phones line for cleaner output
                    for line in result.stdout.splitlines():
                        if "Predicted Phones:" in line:
                             print(f"    {line.strip()}")
                else:
                    print("    Error running inference:")
                    # print only last few lines of error
                    lines = result.stderr.splitlines()
                    for l in lines[-3:]:
                        print(f"      {l}")
            except Exception as e:
                print(f"    Failed to run command: {e}")

if __name__ == "__main__":
    main()
