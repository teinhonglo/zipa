#!/usr/bin/env python3
import argparse
import glob
import os
import re
import string
from pathlib import Path

import textgrid

TRAIN_SPK = ["EBVS", "ERMS", "HQTV", "PNV", "ASI", "RRBI", "BWC", "LXC", "HJK", "HKK", "ABA", "SKA"]
DEV_SPK = ["MBMPS", "THV", "SVBI", "NCC", "YDCK", "YBAA"]
TEST_SPK = ["NJS", "TLV", "TNI", "TXHC", "YKWK", "ZHAA"]


def normalize_phone(phn: str) -> str:
    phn = phn.rstrip(string.digits + "*_").strip()
    if phn in {"sp", "SIL", " ", "spn", ""}:
        return "sil"
    mapping = {
        "ERR": "err",
        "err": "err",
        "ER)": "er",
        "AX": "ah",
        "ax": "ah",
        "AH)": "ah",
        "V``": "v",
        "W`": "w",
    }
    return mapping.get(phn, phn.lower())


def del_repeat_sil(phn_lst, dur_lst, start_lst):
    if not phn_lst:
        return [], [], []
    out_phn, out_dur, out_start = [phn_lst[0]], [str(dur_lst[0])], [str(start_lst[0])]
    for i in range(1, len(phn_lst)):
        if phn_lst[i] == phn_lst[i - 1] == "sil":
            continue
        out_phn.append(phn_lst[i])
        out_dur.append(str(dur_lst[i]))
        out_start.append(str(start_lst[i]))
    return out_phn, out_dur, out_start


def process_split(l2_path: str, split: str, out_dir: Path):
    spk_set = {"train": TRAIN_SPK, "dev": DEV_SPK, "test": TEST_SPK}[split]
    anno_glob = os.path.join(l2_path, "*", "annotation", "*.TextGrid")
    tg_paths = sorted(glob.glob(anno_glob))

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "wrd_text": open(out_dir / "wrd_text", "w", encoding="utf-8"),
        "wav.scp": open(out_dir / "wav.scp", "w", encoding="utf-8"),
        "phn_text": open(out_dir / "phn_text", "w", encoding="utf-8"),
        "transcript_phn_text": open(out_dir / "transcript_phn_text", "w", encoding="utf-8"),
        "phn_dur": open(out_dir / "phn_dur", "w", encoding="utf-8"),
        "transcript_phn_dur": open(out_dir / "transcript_phn_dur", "w", encoding="utf-8"),
        "phn_start": open(out_dir / "phn_start", "w", encoding="utf-8"),
        "transcript_phn_start": open(out_dir / "transcript_phn_start", "w", encoding="utf-8"),
    }

    for tg_path in tg_paths:
        spk_id = tg_path.split("/")[-3]
        if spk_id not in spk_set:
            continue

        utt_base = Path(tg_path).stem
        utt_id = f"{spk_id}_{utt_base}"
        wav_path = re.sub(r"/annotation/", "/wav/", tg_path).replace(".TextGrid", ".wav")
        txt_path = re.sub(r"/annotation/", "/transcript/", tg_path).replace(".TextGrid", ".txt")

        tg = textgrid.TextGrid.fromFile(tg_path)
        phone_tier = tg[1]

        transcript_phns, perceived_phns = [], []
        phn_dur, phn_start = [], []
        for interval in phone_tier:
            dur = int((float(interval.maxTime) - float(interval.minTime)) * 100)
            start = int(float(interval.minTime) * 100)
            phn_dur.append(dur)
            phn_start.append(start)

            mark = interval.mark
            if not mark:
                transcript_phns.append("sil")
                perceived_phns.append("sil")
                continue

            parts = mark.split(",")
            transcript = normalize_phone(parts[0])
            perceived = normalize_phone(parts[1] if len(parts) > 1 else parts[0])
            transcript_phns.append(transcript)
            perceived_phns.append(perceived)

        perceived_phns, perceived_dur, perceived_start = del_repeat_sil(perceived_phns, phn_dur, phn_start)
        transcript_phns, transcript_dur, transcript_start = del_repeat_sil(transcript_phns, phn_dur, phn_start)

        with open(txt_path, "r", encoding="utf-8") as tf:
            text = tf.read().strip().lower()

        files["wrd_text"].write(f"{utt_id} {text}\n")
        files["wav.scp"].write(f"{utt_id} {wav_path}\n")
        files["phn_text"].write(f"{utt_id} {' '.join(perceived_phns)}\n")
        files["transcript_phn_text"].write(f"{utt_id} {' '.join(transcript_phns)}\n")
        files["phn_dur"].write(f"{utt_id} {' '.join(perceived_dur)}\n")
        files["transcript_phn_dur"].write(f"{utt_id} {' '.join(transcript_dur)}\n")
        files["phn_start"].write(f"{utt_id} {' '.join(perceived_start)}\n")
        files["transcript_phn_start"].write(f"{utt_id} {' '.join(transcript_start)}\n")

    for f in files.values():
        f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2arctic-dir", required=True)
    parser.add_argument("--output-dir", default="data-kaldi/l2arctic")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    for split in ["train", "dev", "test"]:
        process_split(args.l2arctic_dir, split, out_root / split)


if __name__ == "__main__":
    main()
