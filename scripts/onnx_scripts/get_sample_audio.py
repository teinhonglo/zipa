
import io
import soundfile as sf
from datasets import load_dataset, Audio
import os

def main():
    print("Loading dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    
    sample = ds[0]
    
    # Audio decoding manually
    audio_path = sample["audio"]["path"]
    audio_bytes = sample["audio"]["bytes"]
    
    if audio_bytes:
        print("Found audio bytes. Saving to sample.wav...")
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        sf.write('sample.wav', audio_array, sample_rate)
        print(f"Saved sample.wav (SR: {sample_rate})")
    elif audio_path and os.path.exists(audio_path):
         print(f"Found local path: {audio_path}")
         audio_array, sample_rate = sf.read(audio_path)
         sf.write('sample.wav', audio_array, sample_rate)
         print(f"Saved sample.wav (SR: {sample_rate})")
    else:
        print("Could not retrieve audio.")

if __name__ == "__main__":
    main()
