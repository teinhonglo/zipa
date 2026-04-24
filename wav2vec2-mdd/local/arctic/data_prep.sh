#!/bin/bash

# Copyright 2023 RCPET@NTNU (Fu-An Chao)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

stage=-1

arctic_dir=corpus/arctic
spks="slt bdl rms clb jmk awb"
data_kaldi='data-kaldi/arctic'
data_json='data-json/arctic'

. ./path.sh
. ./local/parse_options.sh

GREEN='\033[0;32m' # green
NC='\033[0m' # no color

if [ ${stage} -le -1 ]; then
    for spk in $spks; do
        echo -e "${GREEN}download cmu arctic corpus for speaker ${spk} ...${NC}"
	local/arctic/data_download.sh ${arctic_dir} ${spk}
	echo
    done
fi

if [ ${stage} -le 0 ]; then
    [ ! -d $data_kaldi ] && mkdir -p $data_kaldi
    for spk in $spks; do
        echo -e "${GREEN}kaldi data preparation for speaker $spk ...${NC}"
	local/arctic/arctic_prep.sh ${arctic_dir}/cmu_us_${spk}_arctic ${spk} $data_kaldi/${spk}
	echo
    done
fi

if [ ${stage} -le 1 ]; then
    echo -e "${GREEN}creating dataset for training ...${NC}"
    echo -e "${GREEN}---------------------------------${NC}"
    echo -e "${GREEN}train: slt, bdl, rms, clb${NC}"
    echo -e "${GREEN}dev: jmk${NC}"
    echo -e "${GREEN}test: awb${NC}"
    echo -e "${GREEN}---------------------------------${NC}"

    # train: slt, bdl, rms, clb
    local/utils/combine_data.sh $data_kaldi/train $data_kaldi/slt $data_kaldi/bdl $data_kaldi/rms $data_kaldi/clb

    # dev: jmk
    cp -r $data_kaldi/jmk $data_kaldi/dev

    # test: awb
    cp -r $data_kaldi/awb $data_kaldi/test
    exit 0
fi

if [ $stage -le 2 ]; then
    echo -e "${GREEN}json data preparation ...${NC}"

    [ ! -d $data_json ] && mkdir -p $data_json
    python local/utils/get_model_units.py \
        ${data_kaldi}/train/phn_text $data_json/units || exit 1

    for data in train dev test; do
        python local/utils/prep_json.py --task "asr" \
            --data ${data_kaldi}/$data --json $data_json/$data.json || exit 1
    done

    echo DONE
fi
