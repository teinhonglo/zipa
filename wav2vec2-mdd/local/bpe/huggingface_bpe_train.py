#!/bin/python
import os
import json
import argparse
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, CharDelimiterSplit, Sequence

parser = argparse.ArgumentParser()
parser.add_argument("--train-files", default="data/local/lm/librispeech-lm-corpus-76M-supphone.txt")
parser.add_argument("--exp-dir", default="bpe_model/tokenizer-ls76M-supphone-100")
parser.add_argument("--vocab-size", default=100)
args = parser.parse_args()

train_files = ["".join(args.train_files)]
exp_dir = args.exp_dir
vocab_size = int(args.vocab_size)

bpe_model = exp_dir + "/model.json"
bpe_vocab = exp_dir + "/vocab.json"

#test = "NOW WAHN WAHZ HHERT IHN DHAH IHKSPLOWZHN"
#test = "AELIHS WIHL AHRAYV IHN JHAENYUWERIY SIHKSTH"

# define model
tokenizer = Tokenizer(BPE(unk_token="[unk]"))

#"TH", "NG", "ZH", "AE", "OY", "AO"
#"AW"
trainer = BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=["[unk]", "th", "ng", "zh", "ae", "oy", "ao"],
)
tokenizer.pre_tokenizer = Whitespace()

# train
tokenizer.train(train_files, trainer)
# save model
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
tokenizer.save(bpe_model)
# save dictionary
with open(bpe_vocab, "w") as wf:
    json.dump(tokenizer.get_vocab(), wf, indent=4)

# test
#tokenizer = Tokenizer.from_file(bpe_model)

#output = tokenizer.encode(test)
#print(output.tokens)
#print(output.ids)
