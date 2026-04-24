# -*- coding: utf-8 -*-
import sys
import argparse
import kaldi_io
from tqdm import tqdm
from collections import defaultdict
from utils import load_phone_symbol_table

parser = argparse.ArgumentParser()
parser.add_argument("--text-phone")
parser.add_argument("--text-group-phone-bpe")
parser.add_argument("--text-supphone")
args = parser.parse_args()

#text_pure = "data/train_phone/text"
#text_bpe_supphone = "data/train_supphone/text_bpe"
#text_bpe_supphone_align = "data/train_supphone/text_bpe_align"

pure = {}
with open(args.text_phone, "r") as rf:
    for line in rf.readlines():
        uttid, pure_phones = line.split()[0], line.split()[1:]
        pure[uttid] = pure_phones

bpe_supphone = defaultdict(list)
with open(args.text_group_phone_bpe, "r") as rf:
    for line in rf.readlines():
        uttid, bpe_supphones = line.split()[0], line.split()[1:]
        bpe_supphone[uttid] = bpe_supphones

with open(args.text_supphone, "w") as wf:
    for uttid, pure_phones in pure.items():
        bpe_supphone_align = []

        idx = 0
        for pure in pure_phones:
            if pure != "sil":
                # if pure phone not in supphone go next
                while bpe_supphone[uttid][idx].find(pure) == -1:
                    print(uttid, bpe_supphone[uttid], idx, pure)
                    idx += 1
                bpe_supphone_align.append(bpe_supphone[uttid][idx])
            else:
                # if pure phone == "sil"
                bpe_supphone_align.append(pure)


        wf.write(uttid + " " + " ".join(bpe_supphone_align) + "\n")
