#!/bin/bash

# Author: Fu-An Chao
# 	  this code is modified from Kaiqi Fu, JinJong Lin

stage=0
stop_stage=10000

l2arctic_dir="/share/corpus/l2arctic_release_v4.0"
timit_dir='/share/corpus/TIMIT'
phoneme_map='60-39'
data_kaldi='data-kaldi/l2arctic'
data_json='data-json/l2arctic'
#tts_data_dir='/share/nas167/teinhonglo/espnets/espnet-hakka/egs2/l2arctic/mdd1/data-mdd'
tts_data_dir=
remove_sil=false

. ./path.sh
. ./local/parse_options.sh

GREEN='\033[0;32m' # green
NC='\033[0m' # no color

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo -e "${GREEN}kaldi data preparation ...${NC}"

    [ ! -d $data_kaldi ] && mkdir -p $data_kaldi
    local/l2arctic/timit_data_prep.sh $timit_dir $phoneme_map ${data_kaldi}/train_timit || exit 1
    python local/l2arctic/l2arctic_prep.py --l2_path=$l2arctic_dir --save_path=${data_kaldi}/train_l2 || exit 1
    python local/l2arctic/l2arctic_prep.py --l2_path=$l2arctic_dir --save_path=${data_kaldi}/dev || exit 1
    python local/l2arctic/l2arctic_prep.py --l2_path=$l2arctic_dir --save_path=${data_kaldi}/test || exit 1

    # unsup
    #python local/l2arctic/l2arctic_prep.py --unsup --l2_path=$l2arctic_dir --save_path=${data_kaldi}/train_l2_unsup || exit 1
    #python local/l2arctic/l2arctic_prep.py --unsup --l2_path=$l2arctic_dir --save_path=${data_kaldi}/dev_unsup || exit 1
    #python local/l2arctic/l2arctic_prep.py --unsup --l2_path=$l2arctic_dir --save_path=${data_kaldi}/test_unsup || exit 1

    local/l2arctic/timit_l2_merge.sh \
        ${data_kaldi}/train_timit ${data_kaldi}/train_l2 ${data_kaldi}/train || exit 1
    #rm -rf ${data_kaldi}/train_l2 ${data_kaldi}/train_timit
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo -e "${GREEN}create detection targets ...${NC}"
    for data in train train_l2 dev test; do #train_l2_unsup dev_unsup test_unsup; do
        if [ "$remove_sil" == "true" ]; then
            sed -i "s/ sil//g" ${data_kaldi}/$data/phn_text
            sed -i "s/ sil//g" ${data_kaldi}/$data/transcript_phn_text
        fi
        
        # Usage: local/l2arctic/create_detection_targets.sh <phn_text> <transcript_phn_text> <detection-targets>
        local/l2arctic/create_detection_targets.sh \
            ${data_kaldi}/$data/phn_text \
            ${data_kaldi}/$data/transcript_phn_text \
            ${data_kaldi}/$data/detection_targets
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo -e "${GREEN}repeate phn targets (transcript_phn_text,phn_text)...${NC}"
    for data in train_l2 dev test; do  #train_l2_unsup dev_unsup test_unsup; do
        
        # Usage: local/l2arctic/create_detection_targets.sh <phn_text> <transcript_phn_text> <detection-targets>
        python local/l2arctic/l2arctic_phn_rep.py \
            --wavscp ${data_kaldi}/$data/wav.scp \
            --phn_start ${data_kaldi}/$data/transcript_phn_start \
            --phn_dur ${data_kaldi}/$data/transcript_phn_dur \
            --phn_text ${data_kaldi}/$data/transcript_phn_text \
            --phn_text_rep ${data_kaldi}/$data/transcript_phn_text_rep
        
        python local/l2arctic/l2arctic_phn_rep.py \
            --wavscp ${data_kaldi}/$data/wav.scp \
            --phn_start ${data_kaldi}/$data/phn_start \
            --phn_dur ${data_kaldi}/$data/phn_dur \
            --phn_text ${data_kaldi}/$data/phn_text \
            --phn_text_rep ${data_kaldi}/$data/phn_text_rep
        
        python local/l2arctic/l2arctic_phn_rep.py \
            --wavscp ${data_kaldi}/$data/wav.scp \
            --phn_start ${data_kaldi}/$data/transcript_phn_start \
            --phn_dur ${data_kaldi}/$data/transcript_phn_dur \
            --phn_text ${data_kaldi}/$data/detection_targets \
            --phn_text_rep ${data_kaldi}/$data/detection_targets_rep \
            --special_token "1"
        
        python local/l2arctic/l2arctic_phn_rep.py \
            --wavscp ${data_kaldi}/$data/wav.scp \
            --phn_start ${data_kaldi}/$data/phn_start \
            --phn_dur ${data_kaldi}/$data/phn_dur \
            --phn_text ${data_kaldi}/$data/detection_targets_ppl \
            --phn_text_rep ${data_kaldi}/$data/detection_targets_ppl_rep \
            --special_token "1"
        
        data_dir=${data_kaldi}/$data; 
        cp $data_dir/wrd_text $data_dir/text; 
        awk '{print $1}' $data_dir/text | awk -F"_" '{print $1"_"$2"_"$3" "$1}' > $data_dir/utt2spk; 
        utils/fix_data_dir.sh --utt_extra_files "detection_targets detection_targets_ppl detection_targets_ppl_rep detection_targets_rep phn_dur phn_start phn_text phn_text_rep transcript_phn_dur transcript_phn_start transcript_phn_text transcript_phn_text_rep" $data_dir
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo -e "${GREEN}json data preparation ...${NC}"

    [ ! -d $data_json ] && mkdir -p $data_json
    python local/utils/get_model_units.py \
        ${data_kaldi}/train/phn_text $data_json/units || exit 1
    
    # create wav_ref.scp for data-kaldi/*/train
    
    if [ ! -z $tts_data_dir ] && [ -d $tts_data_dir ] ; then
        # tts-mdd
        for data in train train_l2 dev test; do #train_l2_unsup dev_unsup test_unsup; do
            if [ "$data" == "train" ]; then
                grep TIMIT ${data_kaldi}/train/wav.scp > ${data_kaldi}/train/wav_timit.scp
                cat $tts_data_dir/train_l2/wav_ref.scp | sed "s: data: ${tts_data_dir}:g" | sed "s/data-mdd/data/g" > ${data_kaldi}/train/wav_l2.scp
                cat ${data_kaldi}/train/wav_timit.scp ${data_kaldi}/train/wav_l2.scp > ${data_kaldi}/train/wav_ref.scp
                rm -rf ${data_kaldi}/train/{wav_timit.scp,wav_l2.scp}
                extra_fns="audio_ref:wav_ref.scp"
            else
                cat $tts_data_dir/$data/wav_ref.scp | sed "s: data: ${tts_data_dir}:g" | sed "s/data-mdd/data/g" > ${data_kaldi}/$data/wav_ref.scp
                extra_fns="audio_ref:wav_ref.scp,prompt_dur:transcript_phn_dur,dur:phn_dur"
            fi
            python local/utils/prep_json.py --task "mdd" --extra_fns "$extra_fns" \
                --data ${data_kaldi}/$data --json $data_json/$data.json || exit 1
        done
    else
        # mdd
        for data in train train_l2 dev test; do #train_l2_unsup dev_unsup test_unsup; do
            if [ "$data" == "train" ]; then
                extra_fns="none"
            else
                extra_fns="prompt_dur:transcript_phn_dur,dur:phn_dur,transcript_phn_text_rep:transcript_phn_text_rep,phn_text_rep:phn_text_rep,detection_targets_rep:detection_targets_rep,detection_targets_ppl_rep:detection_targets_ppl_rep"
            fi
           
            if [ ! -f $data_json/$data.json ]; then
                python local/utils/prep_json.py --task "mdd" --extra_fns "$extra_fns" \
                    --data ${data_kaldi}/$data --json $data_json/$data.json || exit 1
            fi
        done
    fi

fi

echo DONE
