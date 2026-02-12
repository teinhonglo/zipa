
import os
import sys
import torch
import numpy as np
import onnxruntime as ort
from lhotse.features.kaldi.extractors import Fbank, FbankConfig
import soundfile as sf
import librosa
import argparse

def load_tokens(token_file):
    tokens = {}
    if not os.path.exists(token_file):
        print(f"Warning: Token file {token_file} not found.")
        return {}
        
    with open(token_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                token = parts[0]
                idx = int(parts[1]) if len(parts) > 1 else len(tokens)
                tokens[idx] = token
    return tokens

def ctc_greedy_decode(probs, vocab):
    # probs: (T, Vocab)
    preds = np.argmax(probs, axis=-1)
    decoded = []
    prev_idx = -1
    blank_id = 0 
    
    for idx in preds:
        if idx != blank_id and idx != prev_idx:
            decoded.append(vocab.get(idx, ""))
        prev_idx = idx
    return decoded

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single audio file using ONNX model")
    parser.add_argument("audio_file", help="Path to input audio file (wav, flac, etc.)")
    parser.add_argument("--model-path", required=True, help="Path to ONNX model file (CTC) or directory (Transducer)")
    parser.add_argument("--model-type", choices=["ctc", "transducer"], default="ctc", help="Type of model")
    parser.add_argument("--tokens", default="ipa_simplified/tokens.txt", help="Path to tokens.txt")
    parser.add_argument("--search-suffix", default=".onnx", help="Suffix to search for Transducer models (e.g. .fp16.onnx)")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file {args.audio_file} does not exist.")
        sys.exit(1)

    # Load audio
    print(f"Loading {args.audio_file}...")
    audio, sr = sf.read(args.audio_file)
    if len(audio.shape) > 1:
        audio = audio[:, 0] # Mono
        
    target_sr = 16000
    if sr != target_sr:
        print(f"Resampling from {sr} to {target_sr}...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
    # Feature Extraction
    print("Extracting features (Fbank 80)...")
    config = FbankConfig(num_filters=80, dither=0.0, snip_edges=False)
    extractor = Fbank(config)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    features = extractor.extract_batch([audio_tensor], sampling_rate=16000)
    feature = features[0].unsqueeze(0) # (1, T, 80)
    feat_lens = np.array([feature.shape[1]], dtype=np.int64)
    
    # Load Tokens
    vocab = load_tokens(args.tokens)
    
    # Inference
    if args.model_type == "ctc":
        print(f"Running CTC inference with model: {args.model_path}")
        session = ort.InferenceSession(args.model_path)
        
        inputs = {
            "x": feature.numpy(),
            "x_lens": feat_lens
        }
        outputs = session.run(None, inputs)
        log_probs = outputs[0][0] # (T, Vocab)
        
        phones = ctc_greedy_decode(log_probs, vocab)
        print("\nPredicted Phones:", " ".join(phones))
        
    elif args.model_type == "transducer":
        print(f"Running Transducer inference with model dir: {args.model_path}")
        # Find encoder, decoder, joiner
        pass 
        # (Transducer inference logic needs to be fully implemented if specific example requested)
        # For brevity in example, focusing on structure. User asked for example script.
        # I will implement basic loading check or leave placeholder if complex.
        # But 'test_pipeline.py' has the logic. I can copy it.
        
        enc_path = None
        base_path = args.model_path
        if os.path.isdir(base_path):
             # We look for encoder-*.{suffix}
             # If suffix is default .onnx, we want to avoid .fp16.onnx and .int8.onnx if possible,
             # UNLESS the user explicitly asked for .fp16.onnx.
             
             search_term = args.search_suffix
             
             for f in os.listdir(base_path):
                if f.startswith("encoder-") and f.endswith(search_term):
                    # Strict check: if looking for .onnx, ensure it's not .fp16.onnx
                    if search_term == ".onnx" and (".fp16.onnx" in f or ".int8.onnx" in f):
                        continue
                    
                    suffix = f.replace("encoder-", "")
                    enc_path = os.path.join(base_path, f)
                    
                    # Deduce other files
                    # f is encoder-epoch-999-avg-1.fp16.onnx
                    # suffix is epoch-999-avg-1.fp16.onnx
                    # But wait, decoder is decoder-epoch-999-avg-1.fp16.onnx
                    # So we just replace encoder- with decoder-
                    
                    dec_path = os.path.join(base_path, f.replace("encoder-", "decoder-"))
                    join_path = os.path.join(base_path, f.replace("encoder-", "joiner-"))
                    break
        else:
             print("Error: For transducer, --model-path should be the directory containing .onnx files")
             sys.exit(1)
             
        if enc_path:
             sess_enc = ort.InferenceSession(enc_path)
             sess_dec = ort.InferenceSession(dec_path)
             sess_join = ort.InferenceSession(join_path)
             
             # Encode
             enc_out = sess_enc.run(None, {"x": feature.numpy(), "x_lens": feat_lens})[0][0]
             
             # Decode (Simplified)
             # Reuse logic or import? Better self-contained.
             decoder_input = np.zeros((1, 2), dtype=np.int64) 
             dec_out = sess_dec.run(None, {"y": decoder_input})[0]
             
             decoded = []
             max_sym = 3
             blank_id = 0
             T = enc_out.shape[0]
             
             for t in range(T):
                enc_frame = enc_out[t:t+1, :]
                for _ in range(max_sym):
                    joiner_out = sess_join.run(None, {"encoder_out": enc_frame, "decoder_out": dec_out})[0]
                    pred = np.argmax(joiner_out, axis=-1).item()
                    if pred == blank_id:
                        break
                    decoded.append(vocab.get(pred, ""))
                    decoder_input[0, 0] = decoder_input[0, 1]
                    decoder_input[0, 1] = pred
                    dec_out = sess_dec.run(None, {"y": decoder_input})[0]
             
             print("\nPredicted Phones:", " ".join(decoded))
        else:
             print("Could not find encoder/decoder/joiner ONNX files in directory.")

if __name__ == "__main__":
    main()
