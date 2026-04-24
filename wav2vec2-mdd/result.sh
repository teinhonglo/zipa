#!/bin/bash

GREEN='\033[0;32m' # green
NC='\033[0m' # no color

# timit
dataset=timit
echo -e "${GREEN}--- TIMIT ---${NC}"

echo -e "${GREEN}dev set ${NC}"
for x in exp/$dataset/train_*; do
   wer=$(head -1 $x/decode_dev/asr_result.txt)
   echo $wer $x
done | sort -nk1

echo -e "${GREEN}test set ${NC}"
for x in exp/$dataset/train_*; do
   wer=$(head -1 $x/decode_test/asr_result.txt)
   echo $wer $x
done | sort -nk1

# l2arctic
dataset=l2arctic
echo -e "${GREEN}--- L2ARCTIC ---${NC}"

echo -e "${GREEN}test set ${NC}"
for x in exp/$dataset/train_*; do
   wer=$(grep "w/o silence WER" $x/decode_test/asr_result.txt)
   f1=$(grep "F1" $x/decode_test/mdd_result.txt)
   echo $f1, $wer $x
done | sort -rnk1
