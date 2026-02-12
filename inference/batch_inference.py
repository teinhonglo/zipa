
import os
import sys
import glob
import torch
import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa
import argparse
from typing import List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tokens, ctc_greedy_decode, transducer_greedy_decode, get_fbank_extractor

def load_audio_batch(files: List[str], target_sr=16000):
    audio_list = []
    
    for f in files:
        audio, sr = sf.read(f)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        audio_list.append(torch.from_numpy(audio).float())
        
    return audio_list

def main():
    parser = argparse.ArgumentParser(description="Run batch inference using ONNX models.")
    parser.add_argument("input_path", help="Path to input audio file or directory")
    parser.add_argument("--model-path", required=True, help="Path to ONNX model file (CTC) or directory (Transducer)")
    parser.add_argument("--model-type", choices=["ctc", "transducer"], default="ctc", help="Model architecture")
    parser.add_argument("--tokens", default="ipa_simplified/tokens.txt", help="Path to tokens.txt")
    parser.add_argument("--suffix", default=".onnx", help="Search suffix for Transducer files")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()

    # Collect files
    files = []
    if os.path.isfile(args.input_path):
        files.append(args.input_path)
    elif os.path.isdir(args.input_path):
        # Recursive search for wav/flac
        for ext in ["**/*.wav", "**/*.flac", "**/*.mp3"]:
             files.extend(glob.glob(os.path.join(args.input_path, ext), recursive=True))
    
    if not files:
        print(f"No audio files found in {args.input_path}")
        sys.exit(0)
        
    files.sort()
    print(f"Found {len(files)} files.")
    
    vocab = load_tokens(args.tokens)
    extractor = get_fbank_extractor()

    # Initialize sessions once
    session_ctc = None
    sess_enc, sess_dec, sess_join = None, None, None

    if args.model_type == "ctc":
        session_ctc = ort.InferenceSession(args.model_path)
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
                    
        if enc_path and os.path.exists(enc_path):
             sess_enc = ort.InferenceSession(enc_path)
             sess_dec = ort.InferenceSession(dec_path)
             sess_join = ort.InferenceSession(join_path)
        else:
             print("Transducer model not found.")
             sys.exit(1)

    # Process in batches
    total_files = len(files)
    batch_size = args.batch_size
    
    for i in range(0, total_files, batch_size):
        batch_files = files[i : i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{ (total_files + batch_size - 1) // batch_size } ({len(batch_files)} files)...")

        audio_list = load_audio_batch(batch_files)
        
        # Extract features (lhotse handles padding)
        features_list = extractor.extract_batch(audio_list, sampling_rate=16000)
        
        feat_lens = None
        features_padded = None

        if isinstance(features_list, list):
            # Pad them
            import torch.nn.utils.rnn as rnn_utils
            feat_lens = np.array([f.shape[0] for f in features_list], dtype=np.int64)
            features_padded = rnn_utils.pad_sequence(features_list, batch_first=True) # (B, T, 80)
        else:
            # Already a tensor
            features_padded = features_list
            feat_lens = np.array([features_padded.shape[1]] * len(audio_list), dtype=np.int64)
            
        # The model has a subsampling factor of 4 (Transducer) or 2 (CTC).
        
        if args.model_type == "ctc":
            inputs = {
                "x": features_padded.numpy(), 
                "x_lens": feat_lens
            }
            outputs = session_ctc.run(None, inputs)
            log_probs = outputs[0] # (B, T_sub, V)
            
            # Subsample lengths (CTC approx 2x)
            decoded_lens = feat_lens // 2  
            
            # To be safe, clamp to actual output size.
            seq_len = log_probs.shape[1]
            decoded_lens = np.clip(decoded_lens, 0, seq_len)
            
            results = ctc_greedy_decode(log_probs, vocab, lengths=decoded_lens)
            
            for f, r in zip(batch_files, results):
                print(f"File: {f}")
                print(f"Pred: {' '.join(r)}")
                print("-" * 20)
                
        elif args.model_type == "transducer":
             enc_out = sess_enc.run(None, {
                 "x": features_padded.numpy(), 
                 "x_lens": feat_lens
             })[0] # (B, T, D)
             
             # Subsample lengths for Transducer (4x confirmed)
             decoded_lens = feat_lens // 4
             seq_len = enc_out.shape[1]
             decoded_lens = np.clip(decoded_lens, 0, seq_len)
             
             decoded_phones = transducer_greedy_decode(enc_out, sess_dec, sess_join, vocab, lengths=decoded_lens)
             
             for f, r in zip(batch_files, decoded_phones):
                print(f"File: {f}")
                print(f"Pred: {' '.join(r)}")
                print("-" * 20)

if __name__ == "__main__":
    main()
