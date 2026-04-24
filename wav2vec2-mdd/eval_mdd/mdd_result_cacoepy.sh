#!/bin/bash
# modified from https:/github.com/cageyoko/CTC-Attention-Mispronunciation/blob/master/egs/attention_aug/result/mdd_result.sh

if [ $# -ne 4 ]; then
    echo "Usage: $0     <human-seq>     <ref>       <hyp>       <decode_dir>"
    echo "  human-seq : human perceived results"
    echo "        ref : canonical phone sequences"
    echo "  decode_dir: decode result"
    exit 0
fi

human_seq=$1
ref=$2
hyp=$3
decode_dir=$4

. ./path_py10.sh

# step 0, filter & sort, ensure human_seq, ref contain sames utterances with hyp
eval_mdd/utils/filter_scp.pl -f 1 $hyp $human_seq | sort -nk1 > human_seq
eval_mdd/utils/filter_scp.pl -f 1 $hyp $ref | sort -nk1 > ref
sort -nk1 $hyp > hyp

# step 1
# note : sequence only have 39 phoneme, no sil
python local/align_with_cacoepy.py --ref_fn ref --hyp_fn human_seq --output_fn ref_human_detail
python local/align_with_cacoepy.py --ref_fn human_seq --hyp_fn hyp --output_fn human_our_detail
python local/align_with_cacoepy.py --ref_fn ref --hyp_fn hyp --output_fn ref_our_detail
python eval_mdd/utils/ins_del_sub_cor_analysis.py > $decode_dir/mdd_result_cacoepy.txt

# step 2
compute-wer --text --mode=present ark:human_seq ark:hyp >> $decode_dir/mdd_result_cacoepy.txt|| exit 1;
# step3 
mkdir -p $decode_dir/cacoepy
python eval_mdd/utils/compute_capt_accuracy.py --anno ref_human_detail --pred ref_our_detail --capt_dir $decode_dir/cacoepy

ls $decode_dir/cacoepy
mv ref_human_detail human_our_detail ref_our_detail $decode_dir/cacoepy
rm human_seq ref hyp
