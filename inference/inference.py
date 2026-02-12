
import os
import sys
import torch
import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tokens, ctc_greedy_decode, transducer_greedy_decode, get_fbank_extractor

def main():
    parser = argparse.ArgumentParser(description="Run inference using ONNX models.")
    parser.add_argument("audio_file", help="Path to input audio file")
    parser.add_argument("--model-path", required=True, help="Path to ONNX model file (CTC) or directory (Transducer)")
    parser.add_argument("--model-type", choices=["ctc", "transducer"], default="ctc", help="Model architecture")
    parser.add_argument("--tokens", default="ipa_simplified/tokens.txt", help="Path to tokens.txt")
    parser.add_argument("--suffix", default=".onnx", help="Search suffix for Transducer files (e.g. .fp16.onnx)")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file {args.audio_file} not found.")
        sys.exit(1)

    audio, sr = sf.read(args.audio_file)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
        
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    extractor = get_fbank_extractor()
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    features = extractor.extract_batch([audio_tensor], sampling_rate=16000)
    feature = features[0].unsqueeze(0) 
    feat_lens = np.array([feature.shape[1]], dtype=np.int64)
    
    vocab = load_tokens(args.tokens)
    if not vocab:
        print("Warning: Vocabulary empty or tokens file not found.")

    if args.model_type == "ctc":
        if not os.path.isfile(args.model_path):
             print(f"Error: For CTC, --model-path must be a file.")
             sys.exit(1)
             
        session = ort.InferenceSession(args.model_path)
        
        inputs = {"x": feature.numpy(), "x_lens": feat_lens}
        outputs = session.run(None, inputs)
        log_probs = outputs[0][0] 
        
        phones = ctc_greedy_decode(log_probs, vocab)
        print(" ".join(phones))
        
    elif args.model_type == "transducer":
        base_path = args.model_path
        enc_path, dec_path, join_path = None, None, None
        
        if os.path.isdir(base_path):
             search_term = args.suffix
             for f in os.listdir(base_path):
                if f.startswith("encoder-") and f.endswith(search_term):
                    if search_term == ".onnx" and (".fp16.onnx" in f or ".int8.onnx" in f):
                        continue
                    
                    suffix = f.replace("encoder-", "")
                    enc_path = os.path.join(base_path, f)
                    dec_path = os.path.join(base_path, f.replace("encoder-", "decoder-"))
                    join_path = os.path.join(base_path, f.replace("encoder-", "joiner-"))
                    break
        else:
             print("Error: For Transducer, --model-path should be a directory containing the component ONNX files.")
             sys.exit(1)
             
        if enc_path and os.path.exists(enc_path) and os.path.exists(dec_path):
             sess_enc = ort.InferenceSession(enc_path)
             sess_dec = ort.InferenceSession(dec_path)
             sess_join = ort.InferenceSession(join_path)
             
             enc_out = sess_enc.run(None, {"x": feature.numpy(), "x_lens": feat_lens})[0][0]
             decoded_phones = transducer_greedy_decode(enc_out, sess_dec, sess_join, vocab)
             print(" ".join(decoded_phones))
        else:
             print(f"Could not find valid Transducer model files with suffix '{args.suffix}' in {base_path}")
             sys.exit(1)

if __name__ == "__main__":
    main()
