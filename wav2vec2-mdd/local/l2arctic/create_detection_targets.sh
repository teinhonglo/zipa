#!/bin/bash
# Author: Fuann (National Taiwan Normal University)
# This script create detection targets that align to <transcript_phn_text>

if [ $# -ne 3 ]; then
    echo "Usage: $0 <phn_text> <transcript_phn_text> <detection-targets>"
    echo "  phn_text : human perceived phones "
    echo "  transcript_phn_text : canonical phones "
    exit 0
fi

phn_text=$1
transcript_phn_text=$2
detection_targets=$3

. ./path.sh

align-text ark:$transcript_phn_text ark:$phn_text ark,t:- | eval_mdd/utils/wer_per_utt_details.pl | \
    grep op | sed s:"op"::g | tr -s [:space:] | \
    sed s:" C":" 1":g | sed s:" S":" 0":g | sed s:" D":" 0":g | sed s:" I":"":g > $detection_targets

align-text ark:$transcript_phn_text ark:$phn_text ark,t:- | eval_mdd/utils/wer_per_utt_details.pl | \
    grep op | sed s:"op"::g | tr -s [:space:] | \
    sed s:" C":" 1":g | sed s:" S":" 0":g | sed s:" I":" 0":g | sed s:" D":"":g > ${detection_targets}_ppl
