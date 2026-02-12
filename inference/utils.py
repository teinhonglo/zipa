
import os
import torch
import numpy as np
from lhotse.features.kaldi.extractors import Fbank, FbankConfig

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

def ctc_greedy_decode(probs, vocab, lengths=None):
    # probs: (Batch, Time, Vocab) or (Time, Vocab)
    if len(probs.shape) == 2:
        probs = probs[np.newaxis, :, :] # (1, T, V)
    
    if lengths is None:
        lengths = np.array([probs.shape[1]] * probs.shape[0])
        
    batch_size = probs.shape[0]
    preds = np.argmax(probs, axis=-1) # (B, T)
    
    results = []
    blank_id = 0
    
    for b in range(batch_size):
        decoded = []
        prev_idx = -1
        valid_len = lengths[b]
        
        for t in range(valid_len):
            idx = preds[b, t]
            if idx != blank_id and idx != prev_idx:
                decoded.append(vocab.get(idx, ""))
            prev_idx = idx
        results.append(decoded)
        
    # If input was 2D (single sample), return list of tokens (legacy behavior)
    # But wait, original returned list of tokens for single sample.
    # To maintain backward compatibility while supporting batch:
    # If batch_size was 1 and input was 2D, return results[0].
    # But easier to always return list of lists for batch, and list for single?
    # Let's keep it simple: always return list of lists (batch mode) 
    # UNLESS we want to break legacy 'inference.py'.
    # inference.py calls it with (T, V). -> batch size 1.
    # It expects list of tokens. 
    # So if input was 2D, return results[0].
    
    return results if len(results) > 1 else results[0] # Ambiguous if batch=1 but passed as 3D
    # Let's strictly check input dim.
    
# Re-implement cleaner version that respects input dim
def ctc_greedy_decode(probs, vocab, lengths=None):
    # probs: (Batch, Time, Vocab) or (Time, Vocab)
    is_batch = len(probs.shape) == 3
    if not is_batch:
        probs = probs[np.newaxis, :, :]
        
    if lengths is None:
        lengths = np.array([probs.shape[1]] * probs.shape[0])
        
    batch_size = probs.shape[0]
    preds = np.argmax(probs, axis=-1)
    
    results = []
    blank_id = 0
    
    for b in range(batch_size):
        decoded = []
        prev_idx = -1
        valid_len = lengths[b]
        for t in range(valid_len):
            idx = preds[b, t]
            if idx != blank_id and idx != prev_idx:
                decoded.append(vocab.get(idx, ""))
            prev_idx = idx
        results.append(decoded)
        
    if not is_batch:
        return results[0]
    return results

def transducer_greedy_decode(encoder_out, decoder_model, joiner_model, vocab, lengths=None):
    # encoder_out: (Batch, Time, D) or (Time, D)
    is_batch = len(encoder_out.shape) == 3
    if not is_batch:
        encoder_out = encoder_out[np.newaxis, :, :]
        
    if lengths is None:
        lengths = np.array([encoder_out.shape[1]] * encoder_out.shape[0])
        
    batch_size = encoder_out.shape[0]
    results = []
    
    # We process each sample independently for simplicity.
    # Vectorizing transducer greedy search is complex.
    
    for b in range(batch_size):
        enc_seq = encoder_out[b, :lengths[b], :] # (T_valid, D)
        
        decoded = []
        decoder_input = np.zeros((1, 2), dtype=np.int64) 
        dec_out = decoder_model.run(None, {"y": decoder_input})[0]
        
        T = enc_seq.shape[0]
        blank_id = 0
        max_sym_per_frame = 3
        
        for t in range(T):
            enc_frame = enc_seq[t:t+1, :]
            
            for _ in range(max_sym_per_frame):
                joiner_out = joiner_model.run(None, {
                    "encoder_out": enc_frame,
                    "decoder_out": dec_out
                })[0]
                
                pred = np.argmax(joiner_out, axis=-1).item()
                
                if pred == blank_id:
                    break
                else:
                    decoded.append(vocab.get(pred, ""))
                    decoder_input[0, 0] = decoder_input[0, 1]
                    decoder_input[0, 1] = pred
                    dec_out = decoder_model.run(None, {"y": decoder_input})[0]
        
        results.append(decoded)

    if not is_batch:
        return results[0]
    return results

def get_fbank_extractor():
    config = FbankConfig(num_filters=80, dither=0.0, snip_edges=False)
    return Fbank(config)
