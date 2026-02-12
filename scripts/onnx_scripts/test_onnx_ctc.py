
import sys
import os

# Add zipformer_crctc to path to allow imports from it
# train.py uses "import optim" which requires the dir to be in path
sys.path.insert(0, os.path.abspath("zipformer_crctc"))

# Import train first to avoid RuntimeError: set_num_interop_threads
try:
    from train import get_model
except ImportError:
    # If it fails, maybe we need to rely on sys.path being set correctly
    pass

import argparse
import torch
import onnxruntime as ort
import numpy as np
from icefall.utils import AttributeDict
from zipa_ctc_inference import small_params, large_params
from icefall.utils import AttributeDict
from zipa_ctc_inference import small_params, large_params

def load_pt_model(model_name, pt_path):
    if "small" in model_name:
        params = small_params
    else:
        params = large_params
    
    # Ensure params match the initialization requirements
    # get_model expects AttributeDict with specific keys
    # zipa_ctc_inference.small_params seems to have them.
    # We might need to set device.
    params.device = torch.device("cpu")
    
    # Create model
    model = get_model(params)
    
    # Load checkpoint
    checkpoint = torch.load(pt_path, map_location="cpu")
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
        
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model

def test_ctc(model_name, checkpoints_dir="checkpoints"):
    print(f"Testing {model_name}...")
    model_dir = os.path.join(checkpoints_dir, model_name)
    exp_dir = os.path.join(model_dir, "exp")
    pt_path = os.path.join(exp_dir, "epoch-999.pt")
    onnx_path = os.path.join(exp_dir, "model.onnx")
    onnx_fp16_path = os.path.join(exp_dir, "model.fp16.onnx")
    
    if not os.path.exists(pt_path): 
        print(f"  PyTorch model not found at {pt_path}")
        return
    if not os.path.exists(onnx_path):
        print(f"  ONNX model not found at {onnx_path}")
        return

    # Load PyTorch model
    try:
        pt_model = load_pt_model(model_name, pt_path)
    except Exception as e:
        print(f"  Failed to load PyTorch model: {e}")
        return

    # Dummy input
    B, T, C = 1, 100, 80
    x = torch.randn(B, T, C)
    x_lens = torch.tensor([T], dtype=torch.int64)
    
    # PyTorch Inference
    # Model forward might need specific args
    # ZIPA_CTC.predict uses encoder.forward_encoder and encoder.ctc_output
    # let's look at export-onnx-ctc.py OnnxModel.forward logic
    # x, x_lens = self.encoder_embed(x, x_lens)
    # ...
    # But get_model returns the Zipformer2 model (encoder) directly?
    # zipa_ctc_inference.py says: self.encoder = get_model(params)
    # So pt_model IS the encoder (Zipformer2).
    
    # Wait, in export-onnx-ctc.py:
    # model = get_model(params) ... 
    # model = OnnxModel(encoder=model.encoder, encoder_embed=model.encoder_embed, ctc_output=model.ctc_output)
    
    # It seems get_model returns a wrapper that HAS .encoder, .encoder_embed, .ctc_output?
    # Let's check zipformer_crctc/train.py or infer from usage.
    # zipa_ctc_inference: self.encoder = get_model(params); 
    # encoder_out, _ = self.encoder.forward_encoder(feature, feature_lens)
    # ctc_output = self.encoder.ctc_output(encoder_out)
    
    # So pt_model has forward_encoder.
    
    with torch.no_grad():
        # forward_encoder expects (x, x_lens)
        # x: (N, T, C)
        encoder_out, encoder_out_lens = pt_model.forward_encoder(x, x_lens)
        # ctc_output: (N, T, Vocab)
        log_probs_pt = pt_model.ctc_output(encoder_out)
        
    print(f"  PyTorch output shape: {log_probs_pt.shape}")

    # ONNX Inference
    session = ort.InferenceSession(onnx_path)
    onnx_inputs = {
        "x": x.numpy().astype(np.float32),
        "x_lens": x_lens.numpy().astype(np.int64)
    }
    onnx_out = session.run(None, onnx_inputs)
    log_probs_onnx = onnx_out[0]
    
    print(f"  ONNX output shape: {log_probs_onnx.shape}")
    
    # Compare
    # Output shapes might differ slightly if ONNX handles padding differently?
    # But usually they match.
    diff = np.abs(log_probs_pt.numpy() - log_probs_onnx).max()
    print(f"  Max difference (PT vs ONNX FP32): {diff}")
    
    # FP16 Test
    if os.path.exists(onnx_fp16_path):
        try:
            session_fp16 = ort.InferenceSession(onnx_fp16_path)
            onnx_out_fp16 = session_fp16.run(None, onnx_inputs)
            log_probs_fp16 = onnx_out_fp16[0]
            
            diff_fp16 = np.abs(log_probs_onnx - log_probs_fp16).max()
            print(f"  Max difference (ONNX FP32 vs ONNX FP16): {diff_fp16}")
        except Exception as e:
            print(f"  [WARNING] FP16 Inference/Loading failed: {e}")
            print("  This is often due to onnxruntime validation issues with float16 control flow.")

    # INT8 Test
    onnx_int8_path = os.path.join(exp_dir, "model.int8.onnx")
    if os.path.exists(onnx_int8_path):
        try:
            session_int8 = ort.InferenceSession(onnx_int8_path)
            onnx_out_int8 = session_int8.run(None, onnx_inputs)
            log_probs_int8 = onnx_out_int8[0]
            
            diff_int8 = np.abs(log_probs_onnx - log_probs_int8).max()
            print(f"  Max difference (ONNX FP32 vs ONNX INT8): {diff_int8}")
        except Exception as e:
            print(f"  [ERROR] INT8 Inference failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name to test")
    parser.add_argument("--all", action="store_true", help="Test all models")
    args = parser.parse_args()
    
    models = [d for d in os.listdir("checkpoints") if os.path.isdir(os.path.join("checkpoints", d))]
    
    if args.model:
        test_ctc(args.model)
    elif args.all:
        for m in models:
            if "zipa-cr-" in m: # Only test CTC models with this script
                test_ctc(m)

