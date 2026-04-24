#!/bin/bash
set -euo pipefail

# data prep config
timit_dir='/share/corpus/TIMIT'
phoneme_map='60-39'
map_file=local/timit/phones.60-48-39.l2arctic.map
data_kaldi='data-kaldi/timit'
data_json='data-json/timit'
train=train
dev=dev
tests="dev test"

# training config
nj=4
gpuid=1
seed=66 #824
train_conf=conf/train_timit_baseline_wav2vec2_large_lv60.json  # dev: 7.3 / test: 8.4
suffix=

# stage
stage=1
stop_stage=1000

. ./local/parse_options.sh
. ./path.sh

GREEN='\033[0;32m' # green
NC='\033[0m' # no color

conf_tag=$(basename -s .json $train_conf)
exp_root=exp/timit/${conf_tag}${suffix}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then

    echo -e "${GREEN}Stage 0: data preparation ...${NC}"

    local/timit/data_prep.sh \
        --stage 0 \
        --timit-dir $timit_dir --phoneme-map $phoneme_map --map-file $map_file \
        --data-kaldi $data_kaldi --data-json $data_json || exit 1

fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then

    echo -e "${GREEN}Stage 1: start training ...${NC}"

    [ ! -d $exp_root ] && mkdir -p $exp_root

    CUDA_VISIBLE_DEVICES="$gpuid" \
        python -u train.py \
            --train-conf $train_conf \
            --units $data_json/units \
            --seed $seed \
            --train-json $data_json/$train.json \
            --valid-json $data_json/$dev.json \
            --exp-dir $exp_root \
            --nj $nj | tee -a $exp_root/train.log || exit 1

fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then

    echo -e "${GREEN}Stage 2: start testing ...${NC}"

    for test in $tests; do

        CUDA_VISIBLE_DEVICES="$gpuid" \
            python test.py --remove-sil \
                --model-path $exp_root \
                --decoded_type greedy \
                --test-json $data_json/$test.json \
                --result-dir $exp_root/decode_$test \
                --nj $nj || exit 1

        echo
        echo -e "${GREEN}Write asr result to $exp_root/decode_$test/asr_result.txt${NC}"
        echo
        cat $exp_root/decode_$test/asr_result.txt
        echo

    done
    exit 0

fi
