
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

    # Load audio
    # For simplicity in this example, we process all in one batch.
    # In production, you'd want to chunk this.
    
    audio_list = load_audio_batch(files)
    
    extractor = get_fbank_extractor()
    # extract_batch handles padding
    features = extractor.extract_batch(audio_list, sampling_rate=16000) 
    # features: (Batch, Frames, 80) padded
    
    # We need actual lengths to create x_lens
    # lhotse pads with 0. 
    # We can compute lengths from audio lengths.
    # Frame shift is 10ms (0.01s).
    
    feat_lens = []
    for a in audio_list:
        # Number of frames roughly audio_len / sample_rate / frame_shift
        # Accurate way: fbank implementation details.
        # But extractor returns a tensor. 
        # Wait, lhotse extract_batch returns a single tensor if inputs are even length? 
        # No, it pads.
        # We can just iterate and confirm non-zero? No, silent audio is 0.
        # Better: (len(audio) / sr) / 0.01 ?
        # Lhotse Fbank is standard Kaldi.
        # num_frames = (num_samples + frame_shift/2) // frame_shift? 
        # Easier: let's recalculate based on duration.
        duration = len(a) / 16000
        # Kaldi: num_frames = 1 + (num_samples - frame_length) / frame_shift
        # but with snip_edges=False, it's roughly duration / shift.
        # Let's trust just passing the padded length for now?
        # NO, x_lens is crucial for CTC/Transducer to know when to stop.
        # Let's count frames for each.
    
    # Extract features (lhotse handles padding)
    features_list = extractor.extract_batch(audio_list, sampling_rate=16000)
    
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
    # We must adjust feat_lens passed to decoding.
    
    vocab = load_tokens(args.tokens)

    if args.model_type == "ctc":
        session = ort.InferenceSession(args.model_path)
        inputs = {
            "x": features_padded.numpy(), 
            "x_lens": feat_lens
        }
        outputs = session.run(None, inputs)
        log_probs = outputs[0] # (B, T_sub, V)
        
        log_probs = outputs[0] # (B, T_sub, V)
        
        # Subsample lengths (CTC approx 2x)
        decoded_lens = feat_lens // 2  
        
        # To be safe, clamp to actual output size.
        seq_len = log_probs.shape[1]
        decoded_lens = np.clip(decoded_lens, 0, seq_len)
        
        results = ctc_greedy_decode(log_probs, vocab, lengths=decoded_lens)
        
        for f, r in zip(files, results):
            print(f"File: {f}")
            print(f"Pred: {' '.join(r)}")
            print("-" * 20)
            
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
             
             enc_out = sess_enc.run(None, {
                 "x": features_padded.numpy(), 
                 "x_lens": feat_lens
             })[0] # (B, T, D)
             
             # Subsample lengths for Transducer (4x confirmed)
             decoded_lens = feat_lens // 4
             seq_len = enc_out.shape[1]
             decoded_lens = np.clip(decoded_lens, 0, seq_len)
             
             decoded_phones = transducer_greedy_decode(enc_out, sess_dec, sess_join, vocab, lengths=decoded_lens)
             
             for f, r in zip(files, decoded_phones):
                print(f"File: {f}")
                print(f"Pred: {' '.join(r)}")
                print("-" * 20)
        else:
             print("Transducer model not found.")

if __name__ == "__main__":
    main()
