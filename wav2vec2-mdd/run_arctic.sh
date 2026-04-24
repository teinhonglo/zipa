#!/bin/bash
set -euo pipefail

# data prep config
arctic_dir=corpus/arctic
spks="slt bdl rms clb jmk awb"
data_kaldi='data-kaldi/arctic'
data_json='data-json/arctic'
train=train
dev=dev
tests="dev test"

# training config
nj=4
gpuid=2
# baseline model
train_conf=conf/train_arctic_baseline_wav2vec2_large_lv60.json
suffix=

# stage
stage=0

. ./local/parse_options.sh
. ./path.sh

GREEN='\033[0;32m' # green
NC='\033[0m' # no color

conf_tag=$(basename -s .json $train_conf)
exp_root=exp/arctic/${conf_tag}${suffix}

if [ $stage -le 0 ]; then

    echo -e "${GREEN}Stage 0: data preparation ...${NC}"

    local/arctic/data_prep.sh --stage 0 \
        --arctic-dir $arctic_dir --spks "$spks" \
        --data-kaldi $data_kaldi --data-json $data_json || exit 1

    exit 0

fi

if [ $stage -le 1 ]; then

    echo -e "${GREEN}Stage 1: start training ...${NC}"

    [ ! -d $exp_root ] && mkdir -p $exp_root

    CUDA_VISIBLE_DEVICES="$gpuid" \
        python -u train.py \
            --train-conf $train_conf \
            --units $data_json/units \
            --train-json $data_json/$train.json \
            --valid-json $data_json/$dev.json \
            --exp-dir $exp_root \
            --nj $nj | tee -a $exp_root/train.log || exit 1

fi

if [ $stage -le 2 ]; then

    echo -e "${GREEN}Stage 2: start testing ...${NC}"

    for test in $tests; do

        CUDA_VISIBLE_DEVICES="$gpuid" \
            python test.py --remove-sil \
                --model-path $exp_root \
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
