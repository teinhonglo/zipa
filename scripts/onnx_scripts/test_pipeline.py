
import os
import sys
import torch
import numpy as np
import onnxruntime as ort
from datasets import load_dataset, Audio
from lhotse.features.kaldi.extractors import Fbank, FbankConfig
import soundfile as sf
import librosa
import argparse

def load_tokens(token_file):
    tokens = {}
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
    blank_id = 0 # Assuming 0 is blank
    
    for idx in preds:
        if idx != blank_id and idx != prev_idx:
            decoded.append(vocab.get(idx, ""))
        prev_idx = idx
    return decoded

def transducer_greedy_decode(encoder_out, decoder_model, joiner_model, vocab):
    # encoder_out: (T, enc_dim)
    # decoder_model: ONNX session
    # joiner_model: ONNX session
    
    decoded = []
    decoder_input = np.zeros((1, 2), dtype=np.int64) # Context size 2, blank=0
    
    # We need to loop over encoder frames
    # Simple greedy search for transducer
    
    T = encoder_out.shape[0]
    t = 0
    max_sym_per_frame = 3 # Prevent infinite loops
    
    # This is a simplified greedy search logic since we can't easily batch it in pure python loop efficiently without guidance
    # We'll process frame by frame.
    
    # But wait, ONNX decoder is stateful? No, it takes context.
    # We update context (decoder_input) as we predict non-blank symbols.
    
    # Standard greedy:
    # 1. Initialize decoder_input with blanks [0, 0]
    # 2. Compute decoder_out = decoder(decoder_input)
    # 3. Loop t from 0 to T:
    #    Loop sym_count from 0 to max_sym_per_frame:
    #       encoder_frame = encoder_out[t]
    #       logits = joiner(encoder_frame, decoder_out)
    #       pred = argmax(logits)
    #       if pred == blank:
    #           break (next frame)
    #       else:
    #           decoded.append(vocab[pred])
    #           update decoder_input (shift and append pred)
    #           decoder_out = decoder(decoder_input)
    
    blank_id = 0
    
    # Pre-compute decoder output for initial context
    dec_out = decoder_model.run(None, {"y": decoder_input})[0] # (1, dec_dim)
    
    for t in range(T):
        enc_frame = encoder_out[t:t+1, :] # (1, enc_dim)
        
        for _ in range(max_sym_per_frame):
            # Joiner
            # Joiner input names might vary. Let's assume standard 'encoder_out', 'decoder_out'
            # enc_frame: (1, enc_dim)
            # dec_out: (1, dec_dim)
            joiner_out = joiner_model.run(None, {
                "encoder_out": enc_frame,
                "decoder_out": dec_out
            })[0] # (1, vocab)
            
            pred = np.argmax(joiner_out, axis=-1).item()
            
            if pred == blank_id:
                break
            else:
                decoded.append(vocab.get(pred, ""))
                # Update decoder context
                # Shift left
                decoder_input[0, 0] = decoder_input[0, 1]
                decoder_input[0, 1] = pred
                # Re-run decoder
                dec_out = decoder_model.run(None, {"y": decoder_input})[0]
                
    return decoded

def main():
    parser = argparse.ArgumentParser(description="Test ONNX pipeline with dummy data")
    parser.add_argument("--ctc-model", default="zipa-cr-small-300k", help="CTC model name")
    parser.add_argument("--transducer-model", default="zipa-t-small-300k", help="Transducer model name")
    args = parser.parse_args()

    print("Loading dataset hf-internal-testing/librispeech_asr_dummy...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    sample = ds[0]
    
    # Audio decoding manually
    audio_path = sample["audio"]["path"]
    if not os.path.isabs(audio_path):
        # Use bytes if available or try to find path?
        # datasets usually caches it. 'path' is mostly useful if local or streamed.
        # But 'bytes' should be there.
        if "bytes" in sample["audio"] and sample["audio"]["bytes"]:
            import io
            audio_bytes = sample["audio"]["bytes"]
            audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        else:
            # Fallback for some dataset versions
            # But librispeech dummy usually has bytes when decode=False
            print("Error: Could not find audio bytes.")
            sys.exit(1)
    else:
        audio_array, sample_rate = sf.read(audio_path)
    
    if len(audio_array.shape) > 1:
        audio_array = audio_array[:, 0] # Mono
    text = sample["text"] # Normalized text, might not match phones exactly but good for reference
    
    print(f"Sample Text: {text}")
    print(f"Audio shape: {audio_array.shape}, SR: {sample_rate}")
    
    # Resample to 16000 if needed
    target_sr = 16000
    if sample_rate != target_sr:
        print(f"Resampling to {target_sr}...")
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=target_sr)
    
    # Feature Extraction (Fbank 80)
    print("Extracting features (Fbank 80)...")
    # FbankConfig takes parameters directly
    config = FbankConfig(num_filters=80, dither=0.0, snip_edges=False)
    extractor = Fbank(config)
    # extract expects tensor or array
    # extract_batch expects list of tensors
    # Let's use torch tensor
    audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0) # (1, T)
    
    # We can use the logic from inference script:
    # features = self.fbank.extract_batch(audio, sampling_rate=16000)
    # But we need to instantiate it similarly.
    # Lhotse Fbank.extract_batch takes list of cuts or list of tensors?
    # extract_batch(audio: List[Tensor], ...)
    
    features = extractor.extract_batch([audio_tensor], sampling_rate=16000)
    # features is list of tensors
    feature = features[0] # (T_frames, 80)
    feature = feature.unsqueeze(0) # (1, T_frames, 80)
    feat_lens = np.array([feature.shape[1]], dtype=np.int64)
    
    print(f"Feature shape: {feature.shape}")
    
    # Load Tokens
    token_file = "ipa_simplified/tokens.txt"
    if not os.path.exists(token_file):
        # Fallback to absolute path or relative
        token_file = "../ipa_simplified/tokens.txt"
        
    if os.path.exists(token_file):
        vocab = load_tokens(token_file)
        print(f"Loaded {len(vocab)} tokens.")
    else:
        print(f"Warning: Token file not found at {token_file}. Decoding will use IDs.")
        vocab = {}

    # 1. CTC Inference
    print(f"\n--- CTC Inference ({args.ctc_model}) ---")
    ctc_path = f"checkpoints/{args.ctc_model}/exp/model.onnx"
    if not os.path.exists(ctc_path): ctc_path = f"../checkpoints/{args.ctc_model}/exp/model.onnx"
    
    if os.path.exists(ctc_path):
        session = ort.InferenceSession(ctc_path)
        inputs = {
            "x": feature.numpy(),
            "x_lens": feat_lens
        }
        outputs = session.run(None, inputs)
        log_probs = outputs[0] # (1, T, Vocab)
        
        decoded_phones = ctc_greedy_decode(log_probs[0], vocab)
        print("Decoded (CTC):", " ".join(decoded_phones))
    else:
        print(f"CTC model not found at {ctc_path}")

    # 2. Transducer Inference
    print(f"\n--- Transducer Inference ({args.transducer_model}) ---")
    base_path = f"checkpoints/{args.transducer_model}/exp"
    if not os.path.exists(base_path): base_path = f"../checkpoints/{args.transducer_model}/exp"
    
    # We need to find the correct filenames (epoch-999-avg-1)
    # Let's search
    enc_path = None
    for f in os.listdir(base_path):
        if f.startswith("encoder-") and f.endswith(".onnx") and "fp16" not in f and "int8" not in f:
            suffix = f.replace("encoder-", "")
            enc_path = os.path.join(base_path, f)
            dec_path = os.path.join(base_path, f"decoder-{suffix}")
            join_path = os.path.join(base_path, f"joiner-{suffix}")
            break
            
    if enc_path and os.path.exists(enc_path):
        print(f"Loading Transducer parts from {base_path}...")
        sess_enc = ort.InferenceSession(enc_path)
        sess_dec = ort.InferenceSession(dec_path)
        sess_join = ort.InferenceSession(join_path)
        
        # Encoder
        enc_out = sess_enc.run(None, {"x": feature.numpy(), "x_lens": feat_lens})[0] # (1, T, enc_dim)
        encoder_output = enc_out[0] # (T, enc_dim)
        
        decoded_phones_t = transducer_greedy_decode(encoder_output, sess_dec, sess_join, vocab)
        print("Decoded (Transducer):", " ".join(decoded_phones_t))

    else:
        print(f"Transducer model files not found in {base_path}")

if __name__ == "__main__":
    main()
