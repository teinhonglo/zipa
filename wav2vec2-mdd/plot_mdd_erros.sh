#!/bin/bash
set -euo pipefail

# human-seq : human perceived results
# ref : canonical phone sequences

# stage
stage=0
stop_stage=1000
human_seq=eval_mdd/human_seq
ref=eval_mdd/ref
topN=10


. ./local/parse_options.sh
. ./path.sh

GREEN='\033[0;32m' # green
NC='\033[0m' # no color

capt_root=capt

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then

    echo -e "${GREEN}Stage 0: aligning ...${NC}"
    mkdir -p $capt_root/
    sed "s/ sil//g" $ref > $capt_root/ref
    sed "s/ sil//g" $human_seq > $capt_root/human_seq
    align-text ark:$capt_root/ref  ark:$capt_root/human_seq ark,t:- | eval_mdd/utils/wer_per_utt_details.pl > $capt_root/ref_human_detail

fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then

    echo -e "${GREEN}Stage 1: start calculating ...${NC}"
    python local/data/calc_all_mdd_errors.py --anno $capt_root/ref_human_detail --capt_dir $capt_root/
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then

    echo -e "${GREEN}Stage 2: start ploting ...${NC}"
    python local/data/plot_from_csv_results.py --capt_dir $capt_root --topN $topN
fi
