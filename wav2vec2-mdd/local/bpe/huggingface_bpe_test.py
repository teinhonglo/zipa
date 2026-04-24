#!/bin/python
import argparse
from tokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--bpe-model", default="bpe_model/tokenizer-ls76M-supphone-100/model.json")
parser.add_argument("--kaldi-text")
parser.add_argument("--kaldi-text-bpe")
args = parser.parse_args()

bpe_model = args.bpe_model
kaldi_text = args.kaldi_text
kaldi_text_bpe = args.kaldi_text_bpe

#bpe_model = "bpe_model/tokenizer-ls76M-supphone-100/model.json"
#kaldi_text = "../../gop_speechocean762/s5/data/train_supphone/text"
#kaldi_text = "../../gop_speechocean762/s5/data/test_supphone/text"
#kaldi_text_bpe = kaldi_text + "_bpe"

# Input Example:
# 000010011 WIY KAOL IHT BEHR

tokenizer = Tokenizer.from_file(bpe_model)

with open(kaldi_text_bpe, "w") as wf:
    with open(kaldi_text, "r") as rf:
        for line in rf.readlines():
            id, text = line.split()[0], line.strip().split()[1:]
            text = " ".join(text)
            output = tokenizer.encode(text).tokens
            output = " ".join(output)
            wf.write(id + " " + output + "\n")
