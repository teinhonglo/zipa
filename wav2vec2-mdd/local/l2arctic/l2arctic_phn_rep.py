#!/bin/python
import argparse
import os
import json
import torchaudio

parser = argparse.ArgumentParser()
parser.add_argument("--wavscp", type=str, default="data-kaldi/l2arctic/test/wav.scp")
parser.add_argument("--phn_start", type=str, default="data-kaldi/l2arctic/test/phn_start")
parser.add_argument("--phn_dur", type=str, default="data-kaldi/l2arctic/test/phn_dur")
parser.add_argument("--phn_text", type=str, default="data-kaldi/l2arctic/test/phn_text")
parser.add_argument("--phn_text_rep", type=str, default="data-kaldi/l2arctic/test/phn_text_rep")
parser.add_argument("--special_token", type=str, default="sil")
args = parser.parse_args()

def file2dict(fname):
    uttid_list = []
    data_dict = {}
    with open(fname, "r") as fn:
        for line in fn.readlines():
            info = line.split()
            id, text = info[0], " ".join(info[1:])
            data_dict[id] = text
            uttid_list.append(id)
    
    return data_dict, uttid_list

# Paramters
wavscp = args.wavscp
phn_start = args.phn_start
phn_dur = args.phn_dur
phn_text = args.phn_text
phn_text_rep = args.phn_text_rep
special_token = args.special_token

wav_dict, uttid_list = file2dict(wavscp)
start_dict, uttid_list = file2dict(phn_start)
dur_dict, _ = file2dict(phn_dur)
phn_text_dict, _ = file2dict(phn_text)

def get_frames(path):
    speech_array, sampling_rate = torchaudio.load(path)
    num_seconds = speech_array.shape[1] / sampling_rate
    num_frames = int(num_seconds * 100)
    return num_frames

wav_frames = {}

for uttid, wav_path in wav_dict.items():
    wav_frames[uttid] = get_frames(wav_path)

phn_text_rep_dict = {}
# rep phones
for uttid, frames in wav_frames.items():
    phn_list = [ special_token for _ in range(frames) ]
    
    start_list = start_dict[uttid].split()
    dur_list = dur_dict[uttid].split()
    phn_text_list = phn_text_dict[uttid].split()

    for s, d, pt in zip(start_list, dur_list, phn_text_list):
        s = int(s)
        d = int(d)
        
        # 避免不同個 phone 被視作同個 phone
        if s > 0 and phn_list[s - 1] == pt:
            phn_list[s - 1] = special_token
        
        phn_list[s: s + d] = [pt] * d
    
    phn_text_rep_dict[uttid] = phn_list

with open(phn_text_rep, "w") as w_fn:
    for uttid in uttid_list:
        phn_list = phn_text_rep_dict[uttid]
        phn_text_rep = " ".join(phn_list)
        w_fn.write(f"{uttid} {phn_text_rep}\n")

