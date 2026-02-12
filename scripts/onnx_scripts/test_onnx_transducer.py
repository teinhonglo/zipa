
import torch
import onnxruntime as ort
import numpy as np
import argparse
import sys
import os

def test_transducer(model_name, checkpoints_dir="checkpoints"):
    print(f"Testing {model_name}...")
    model_dir = os.path.join(checkpoints_dir, model_name)
    exp_dir = os.path.join(model_dir, "exp")
    
    suffixes = ["", ".fp16", ".int8"]
    
    for suffix in suffixes:
        search_suffix = suffix + ".onnx"
        enc_search = f"encoder-"
        # The export script generates: encoder-epoch-999-avg-1.onnx
        # My script uses --epoch 999 --avg 1
        # So filename is encoder-epoch-999-avg-1.onnx
        # But wait, my run_export.sh didn't specify filename output, the python script determins it.
        # Check export-onnx.py line 560:
        # suffix = f"epoch-{params.epoch}-avg-{params.avg}"
        # encoder_filename = ... / f"encoder-{suffix}.onnx"
        
        # So I need to find the files.
        # Note: if suffix is .int8, we look for encoder-epoch-999-avg-1.int8.onnx
        found_files = [f for f in os.listdir(exp_dir) if f.startswith("encoder-") and f.endswith(search_suffix)]
        if not found_files:
            if suffix == "": print(f"  No ONNX encoder found in {exp_dir}")
            continue
            
        file_suffix = found_files[0].replace("encoder-", "").replace(search_suffix, "")
        # e.g. epoch-999-avg-1
        
        enc_name = f"encoder-{file_suffix}{search_suffix}"
        dec_name = f"decoder-{file_suffix}{search_suffix}"
        join_name = f"joiner-{file_suffix}{search_suffix}"
        
        enc_path = os.path.join(exp_dir, enc_name)
        dec_path = os.path.join(exp_dir, dec_name)
        join_path = os.path.join(exp_dir, join_name)
        
        if not (os.path.exists(enc_path) and os.path.exists(dec_path) and os.path.exists(join_path)):
             print(f"  Missing one of the ONNX files for suffix '{suffix}'")
             continue

        print(f"  Testing {suffix if suffix else 'fp32'} ONNX models...")
        
        try:
            # 1. Encoder
            session_enc = ort.InferenceSession(enc_path)
            # Dummy Input
            B, T, C = 1, 100, 80
            x = np.random.randn(B, T, C).astype(np.float32)
            x_lens = np.array([T], dtype=np.int64)
            
            enc_out = session_enc.run(None, {"x": x, "x_lens": x_lens})
            encoder_out = enc_out[0]
            encoder_out_lens = enc_out[1]
            print(f"    Encoder output: {encoder_out.shape}")
            
            # 2. Decoder
            context_size = 2 
            y = np.zeros((B, context_size), dtype=np.int64)
            session_dec = ort.InferenceSession(dec_path)
            dec_out = session_dec.run(None, {"y": y})
            decoder_out = dec_out[0]
            print(f"    Decoder output: {decoder_out.shape}")
            
            # 3. Joiner
            encoder_out_frame = encoder_out[:, 0, :] # (N, joiner_dim)
            
            session_join = ort.InferenceSession(join_path)
            join_out = session_join.run(None, {
                "encoder_out": encoder_out_frame,
                "decoder_out": decoder_out
            })
            logit = join_out[0]
            print(f"    Joiner output: {logit.shape}")
            print("    Success.")
        except Exception as e:
             if suffix == ".fp16":
                 print(f"    [WARNING] FP16 Inference/Loading failed: {e}")
                 print("    This is often due to onnxruntime validation issues with float16 control flow.")
             else:
                 print(f"    [ERROR] Inference failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name to test")
    parser.add_argument("--all", action="store_true", help="Test all models")
    args = parser.parse_args()
    
    models = [d for d in os.listdir("checkpoints") if os.path.isdir(os.path.join("checkpoints", d))]
    
    if args.model:
        test_transducer(args.model)
    elif args.all:
        for m in models:
            if "zipa-t-" in m: # Only test Transducer models
                test_transducer(m)
