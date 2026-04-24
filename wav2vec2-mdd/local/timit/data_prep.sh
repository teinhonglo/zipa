#!/bin/bash

# Author: Fu-An Chao
# 	  this code is modified from Kaiqi Fu, JinJong Lin

stage=0

timit_dir='/share/corpus/TIMIT'
phoneme_map='60-39'
map_file=local/timit/phones.60-48-39.l2arctic.map
data_kaldi='data-kaldi/timit'
data_json='data-json/timit'

. ./path.sh
. ./local/parse_options.sh

GREEN='\033[0;32m' # green
NC='\033[0m' # no color

if [ $stage -le 0 ]; then
    echo -e "${GREEN}kaldi data preparation ...${NC}"

    [ ! -d $data_kaldi ] && mkdir -p $data_kaldi
    local/timit/timit_data_prep.sh \
        --map-file $map_file \
        $timit_dir $phoneme_map ${data_kaldi} || exit 1

fi

if [ $stage -le 1 ]; then
    echo -e "${GREEN}create detection targets ...${NC}"
    for data in train dev test; do
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
        ${data_kaldi}/train/phn_text $data_json/units || exit 1

    for data in train dev test; do
        python local/utils/prep_json.py --task "mdd" \
            --data ${data_kaldi}/$data --json $data_json/$data.json || exit 1
    done
    echo DONE
fi
