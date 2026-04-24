#!/bin/bash
set -euo pipefail

# data prep config
data_kaldi='data-kaldi/timit'
data_json='data-json/timit'
espnet_root=/share/nas167/teinhonglo/espnets/espnet-hakka/egs2/l2arctic/asr1
perturb_sets=train

# training config
nj=4
gpuid=2
suffix=

# stage
stage=0

. ./local/parse_options.sh
. ./path.sh

GREEN='\033[0;32m' # green
NC='\033[0m' # no color

if [ $stage -le 0 ]; then
    for data in $perturb_sets; do
        if [ -d $data_kaldi/${data}_tmp ]; then
            rm -rf $data_kaldi/${data}_tmp
        fi
        
        if [ -d $data_kaldi/${data}_sp ]; then
            rm -rf $data_kaldi/${data}_sp
        fi
        
        mkdir -p $data_kaldi/${data}_tmp
        rsync -avP $data_kaldi/${data}/ $data_kaldi/${data}_tmp/
        awk -F" " '{print $1" "$1}' $data_kaldi/${data}_tmp/wav.scp > $data_kaldi/${data}_tmp/utt2spk
        mv $data_kaldi/${data}_tmp/wav_sph.scp $data_kaldi/${data}_tmp/wav.scp
        utils/fix_data_dir.sh $data_kaldi/${data}_tmp
        utils/data/perturb_data_dir_speed_3way.sh $data_kaldi/${data}_tmp  $data_kaldi/${data}_sp
        cur_root=`pwd`
        cd $espnet_root
        local/format_data_dir.sh --data_root $cur_root/$data_kaldi --test_sets ${data}_sp 
        cd -
        # rename
        rm -rf $data_kaldi/${data}_sp
        # NOTE: Dirty Code
        mv $data_kaldi/${data}_sp_16k $data_kaldi/${data}_sp
        sed -i "s/_16k//g" $data_kaldi/${data}_sp/wav.scp
        
        for f in detection_targets phn_text wrd_text transcript_phn_text; do
            #sp0.9- and sp1.1-
            sed -e 's/^/sp1.1-/' $cur_root/$data_kaldi/${data}_tmp/$f > $data_kaldi/${data}_sp/${f}.sp1.1
            sed -e 's/^/sp0.9-/' $cur_root/$data_kaldi/${data}_tmp/$f > $data_kaldi/${data}_sp/${f}.sp0.9
            cat $cur_root/$data_kaldi/${data}_tmp/$f $data_kaldi/${data}_sp/${f}.sp1.1 $data_kaldi/${data}_sp/${f}.sp0.9 > $data_kaldi/${data}_sp/${f}
            rm -rf $data_kaldi/${data}_sp/{${f}.sp1.1,${f}.sp0.9}
        done
        utils/fix_data_dir.sh --utt_extra_files "detection_targets phn_text wrd_text transcript_phn_text" $data_kaldi/${data}_sp
        rm -rf $cur_root/$data_kaldi/${data}_tmp
    done
fi

if [ $stage -le 1 ]; then
    for data in $perturb_sets; do
        data=${data}_sp
        python local/utils/prep_json.py --task "mdd" \
            --data ${data_kaldi}/$data --json $data_json/$data.json || exit 1
    done
fi
