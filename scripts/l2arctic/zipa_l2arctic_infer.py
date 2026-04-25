#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from zipa_ctc_inference import initialize_model


def read_kaldi_text(path: Path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt, value = line.split(maxsplit=1)
            data[utt] = value
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-scp", required=True)
    parser.add_argument("--ref-phn", required=True, help="Reference phone text file")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--bpe-model", default="ipa_simplified/unigram_127.model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--remove-sil", action="store_true")
    args = parser.parse_args()

    wavs = read_kaldi_text(Path(args.wav_scp))
    refs = read_kaldi_text(Path(args.ref_phn))

    model = initialize_model(args.model_path, args.bpe_model)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out:
        for utt in tqdm(sorted(wavs.keys())):
            if utt not in refs:
                continue

            wave, sr = torchaudio.load(wavs[utt])
            if sr != 16000:
                wave = torchaudio.functional.resample(wave, sr, 16000)
            hyp = model.inference([wave[0].contiguous().to(torch.float32)])[0]

            ref = refs[utt].split()
            if args.remove_sil:
                ref = [p for p in ref if p != "sil"]
                hyp = [p for p in hyp if p != "sil"]

            out.write(f"{utt}:\tref={ref}\n")
            out.write(f"{utt}:\thyp={hyp}\n")


if __name__ == "__main__":
    main()
