#!/bin/python
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="asr")
parser.add_argument("--data", type=str)
parser.add_argument("--json", type=str)
parser.add_argument("--extra_fns", type=str, help="audio_ref:wavref.scp")
args = parser.parse_args()

wav_scp = os.path.join(args.data, "wav.scp")
text = os.path.join(args.data, "phn_text")
text_wrd = os.path.join(args.data, "wrd_text")

# Init.
data_dict = {
    "id":[],
    "audio":[],
    "text":[],
    "text_wrd":[]
}

extra_fns = {}
if args.extra_fns is not None and args.extra_fns != "none":
    for einfo in args.extra_fns.split(","):
        key, fname = einfo.split(":")
        data_dict[key] = []
        extra_fns[key] = os.path.join(args.data, fname)

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

# Load file to Dict
wav_dict, uttid_list = file2dict(wav_scp)
text_dict, _ = file2dict(text)
text_wrd_dict, _ = file2dict(text_wrd)

extra_dict = {}
for key, fname in extra_fns.items():
    my_dict, _ = file2dict(fname)
    extra_dict[key] = my_dict

for uttid in uttid_list:
    data_dict["id"].append(uttid)
    data_dict["audio"].append(wav_dict[uttid])
    data_dict["text"].append(text_dict[uttid])
    data_dict["text_wrd"].append(text_wrd_dict[uttid])

# NOTE: if mdd task, add prompt, detection label
if args.task == 'mdd':
    
    prompt = os.path.join(args.data, "transcript_phn_text")
    detection_targets = os.path.join(args.data, "detection_targets")
    detection_targets_ppl = os.path.join(args.data, "detection_targets_ppl")
    
    prompt_dict, _ = file2dict(prompt)
    detection_dict, _ = file2dict(detection_targets)
    detection_dict_ppl, _ = file2dict(detection_targets_ppl)
    
    data_dict["prompt"] = []
    data_dict["detection_targets"] = []
    data_dict["detection_targets_ppl"] = []
    
    for ui, uttid in enumerate(uttid_list):
        data_dict["prompt"].append(prompt_dict[uttid])
        data_dict["detection_targets"].append([ int(d) for d in detection_dict[uttid].split()])
        data_dict["detection_targets_ppl"].append([ int(d) for d in detection_dict_ppl[uttid].split()])
        assert len(data_dict["text"][ui].split()) == len(data_dict["detection_targets_ppl"][ui])

        for key, edict in extra_dict.items():
            data_dict[key].append(edict[uttid])

with open(args.json, 'w') as jsonfile:
    json.dump(data_dict, jsonfile)
