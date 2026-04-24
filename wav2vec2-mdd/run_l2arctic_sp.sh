#!/bin/bash
set -euo pipefail

# data prep config
l2arctic_dir="/share/corpus/l2arctic_release_v4.0"
timit_dir='/share/corpus/TIMIT'
phoneme_map='60-39'
data_kaldi='data-kaldi/l2arctic'
data_json='data-json/l2arctic'
train=train_l2_sp
dev=dev
tests="test"

# training config
nj=4
gpuid=1
seed=33
train_conf=conf/train_l2arctic_baseline_wav2vec2_large_lv60_timitft_prompt.json
suffix=

# stage
stage=1


. ./local/parse_options.sh
. ./path.sh

GREEN='\033[0;32m' # green
NC='\033[0m' # no color

conf_tag=$(basename -s .json $train_conf)
exp_root=exp/l2arctic_sp/${conf_tag}${suffix}

if [ $stage -le 0 ]; then

    echo -e "${GREEN}Stage 0: data preparation ...${NC}"

    local/l2arctic/data_prep.sh \
        --l2arctic-dir $l2arctic_dir --timit-dir $timit_dir --phoneme-map $phoneme_map \
        --data-kaldi $data_kaldi --data-json $data_json || exit 1
fi

if [ $stage -le 1 ]; then

    echo -e "${GREEN}Stage 1: start training ...${NC}"

    [ ! -d $exp_root ] && mkdir -p $exp_root

    CUDA_VISIBLE_DEVICES="$gpuid" \
        python train.py \
            --train-conf $train_conf \
            --units $data_json/units \
            --seed $seed \
            --train-json $data_json/$train.json \
            --valid-json $data_json/$dev.json \
            --exp-dir $exp_root \
            --nj $nj | tee -a $exp_root/train.log || exit 1

fi

if [ $stage -le 2 ]; then

    echo -e "${GREEN}Stage 2: start testing ...${NC}"

    for test in $tests; do
        for decoded_type in greedy beam; do

            CUDA_VISIBLE_DEVICES="$gpuid" \
                python test.py --remove-sil \
                    --model-path $exp_root \
                    --decoded_type $decoded_type \
                    --test-json $data_json/${test}.json \
                    --result-dir $exp_root/decode_${decoded_type}_${test} \
                    --nj $nj || exit 1

            echo
            echo -e "${GREEN}Write asr result to $exp_root/decode_${decoded_type}_$test/asr_result.txt${NC}"
            echo
            cat $exp_root/decode_${decoded_type}_$test/asr_result.txt
            echo
        done

    done

fi

if [ $stage -le 3 ]; then

    echo -e "${GREEN}Stage 3: eval mdd metrics ...${NC}"

    for test in $tests; do
        for decoded_type in greedy beam; do

            eval_mdd/mdd_result.sh \
                eval_mdd/human_seq \
                eval_mdd/ref \
                $exp_root/decode_${decoded_type}_$test/hyp_nosil \
                $exp_root/decode_${decoded_type}_$test

            echo
            echo -e "${GREEN}Write mdd result to $exp_root/decode_${decoded_type}_$test/mdd_result.txt${NC}"
            echo
            cat $exp_root/decode_${decoded_type}_$test/mdd_result.txt
            echo
        done
    done

fi

exit 0;

if [ $stage -le 4 ]; then

    echo -e "${GREEN}Stage 4: start aligning ...${NC}"

    for test in $tests; do

        CUDA_VISIBLE_DEVICES="$gpuid" \
            python align.py --remove-sil \
                --model-path $exp_root \
                --test-json $data_json/${test}.json \
                --result-dir $exp_root/align_${test} \
                --nj $nj || exit 1


    done

fi
