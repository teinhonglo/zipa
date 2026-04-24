#!/bin/python
import argparse
from cacoepy.aligner import AlignARPAbet2
from cacoepy.core.utils import pretty_sequences

def create_kaldi_aligned_format(uttid, aligned_ref, aligned_hyp):
    ref_info = [f"{uttid}", "ref"]
    hyp_info = [f"{uttid}", "hyp"]
    op_info = [f"{uttid}", "op"]
    csid_info = [f"{uttid}", "#csid"]
    csid_dict = { o: 0 for o in ["C", "S", "I", "D"]}
 
    for i, (ref, hyp) in enumerate(zip(aligned_ref, aligned_hyp)):
        if ref == "-":
            ref = "<eps>"
            op = "I"
        elif hyp == "-":
            hyp = "<eps>"
            op = "D"
        elif ref != hyp:
            op = "S"
        else:
            op = "C"
        
        if ref == "sil":
            ref = "err"
        
        if hyp == "sil":
            hyp = "err"
        
        ref_info.append(ref)
        hyp_info.append(hyp)
        op_info.append(op)
        csid_dict[op] += 1

    csid_summary = [str(csid_dict[o]) for o in ["C", "S", "I", "D"]]
    csid_info += csid_summary
    kaldi_format = "\n".join([" ".join(ref_info), " ".join(hyp_info), " ".join(op_info), " ".join(csid_info)])
    
    return kaldi_format


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_fn', type=str, default="eval_mdd/ref")
    parser.add_argument('--hyp_fn', type=str, default="eval_mdd/human_seq")
    parser.add_argument('--output_fn', type=str, default="ref_human_detail")
    parser.add_argument('--gap_penalty', type=float, default=-4)
    args = parser.parse_args()

    utt_ref_dict = {}
    utt_hyp_dict = {}
    uttid_list = []
    aligner = AlignARPAbet2(gap_penalty=args.gap_penalty)
    
    with open(args.ref_fn, "r") as fn:
        for line in fn.readlines():
            info = line.split()
            utt_ref_dict[info[0]] = " ".join(info[1:]).replace("err", "sil")
            uttid_list.append(info[0])
    
    with open(args.hyp_fn, "r") as fn:
        for line in fn.readlines():
            info = line.split()
            utt_hyp_dict[info[0]] = " ".join(info[1:]).replace("err", "sil")

    output_file = open(args.output_fn, "w")
    
    for uttid in uttid_list:
        ref_phonemes = utt_ref_dict[uttid].split(" ")
        hyp_phonemes = utt_hyp_dict[uttid].split(" ")
        aligned_hyp, aligned_ref, score = aligner(hyp_phonemes, ref_phonemes)
        kaldi_format = create_kaldi_aligned_format(uttid, aligned_ref, aligned_hyp)
        output_file.write(f"{kaldi_format}\n")
    
    output_file.close()
