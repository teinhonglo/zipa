#!/bin/bash

# Author: Fu-An Chao
# 	  this code is modified from Kaiqi Fu, JinJong Lin

stage=0

librispeech_dir='/share/corpus/Librispeech'
phoneme_map='60-39'
map_file=local/librispeech/phones.60-48-39.l2arctic.map
data_kaldi='data-kaldi/librispeech'
data_json='data-json/librispeech'

. ./path.sh
. ./local/parse_options.sh

GREEN='\033[0;32m' # green
NC='\033[0m' # no color

if [ $stage -le 0 ]; then
    echo -e "${GREEN}kaldi data preparation ...${NC}"

    if [ ! -d $data_kaldi ]; then
        echo "Please create $data_kaldi with kaldi librispeech (s5-mct) first"
        exit 0;
    fi
fi

if [ $stage -le 1 ]; then
    echo -e "${GREEN}create detection targets ...${NC}"
    for data in train_960_cleaned_val30s train_960 dev_clean dev_other test_clean test_other; do
        # Usage: local/l2arctic/create_detection_targets.sh <phn_text> <transcript_phn_text> <detection-targets>
        local/l2arctic/create_detection_targets.sh \
            ${data_kaldi}/$data/phn_text \
            ${data_kaldi}/$data/transcript_phn_text \
            ${data_kaldi}/$data/detection_targets
    done
fi

if [ $stage -le 2 ]; then
    echo -e "${GREEN}json data preparation ...${NC}"

    [ ! -d $data_json ] && mkdir -p $data_json
    python local/utils/get_model_units.py \
        ${data_kaldi}/train_960/phn_text $data_json/units || exit 1

    for data in train_960_cleaned_val30s train_960 dev_clean dev_other test_clean test_other; do
        python local/utils/prep_json.py --task "mdd" \
            --data ${data_kaldi}/$data --json $data_json/$data.json || exit 1
    done
    echo DONE
fi
