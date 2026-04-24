#!/bin/python
import os
import re
import json
import numpy as np
import argparse
import torch
from transformers import Wav2Vec2Processor
from datasets import load_from_disk

# local import
from utils import make_dataset, load_from_json
from metrics_np import compute_metrics
from models.wav2vec2_model import AutoMDDModel, AutoProtoMDDModel, CfrModel

# decoder
from decoder import *

from align import get_word_timestamps


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_conf_path = os.path.join(args.model_path, "train_conf.json")
    config_path = os.path.join(args.model_path, "config.pth")
    best_model_path = os.path.join(args.model_path, "best")

    # load train_args, model_args
    train_args, model_args = load_from_json(train_conf_path)

    # load config and model
    config = torch.load(config_path)
    processor = Wav2Vec2Processor.from_pretrained(args.model_path)
    vocab_dict = processor.tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
    model_args["num_labels"] = len(sort_vocab)

    if "bpe_path" in model_args:
        processor_sup = Wav2Vec2Processor.from_pretrained(args.model_path + "/bpe_model")
        sup_vocab_dict = processor_sup.tokenizer.get_vocab()
        model_args["num_supphones"] = len(sup_vocab_dict) - 3 # remove special tokens
    
    print("[INFO] Test a {} model. Load pretrained model from {} ...".format(model_args["model_type"], best_model_path))
    if model_args["model_type"] == "prototype":
        model = AutoProtoMDDModel(model_args, config=config, processor=processor).to(device)
    elif model_args["model_type"] == "baseline":
        model = AutoMDDModel(model_args, config=config, processor=processor).to(device)
    elif model_args["model_type"] == "conformer":
        model = CfrModel(model_args, config=config, processor=processor).to(device)
    
    if os.path.isfile(os.path.join(best_model_path, "pytorch_model.bin")):
        model.load_state_dict(torch.load(best_model_path+"/pytorch_model.bin", map_location=device), strict=False)
    elif os.path.isfile(os.path.join(best_model_path, "model.safetensors")):
        model.load_state_dict(torch.load(best_model_path+"/model.safetensors", map_location=device), strict=False)
    model.eval()

    vocab_dict = processor.tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())

    # Lower case ALL letters
    vocab = []
    for _, token in sort_vocab:
        if token in ['<s>', '</s>']: continue
        vocab.append(token)

    # replace the word delimiter with a white space since the white space is used by the decoders
    #vocab[vocab.index(processor.tokenizer.word_delimiter_token)] = ' '
    print("Decode Type", args.decoded_type)
    if args.decoded_type == "beam":
        lm_path = None

        # alpha, beta, and beam_wdith SHOULD be tuned on the dev-set to get the best settings
        # Feel free to check other inputs of the BeamCTCDecoder
        alpha=0
        beta=0

        beam_width = 20
        beam_decoder = BeamCTCDecoder(vocab, lm_path=lm_path,
                                      alpha=alpha, beta=beta,
                                      cutoff_top_n=40, cutoff_prob=1.0,
                                      beam_width=beam_width, num_processes=16,
                                      blank_index=vocab.index(processor.tokenizer.pad_token))
        decoder = beam_decoder
    else:
        greedy_decoder = GreedyDecoder(vocab, blank_index=vocab.index(processor.tokenizer.pad_token))
        decoder = greedy_decoder

    # loading test set
    def preprocess_function(batch):
        audio = batch["audio"]
        # extract features return input_values
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
            # NOTE: mdd task, add prompt
            if model_args["task_type"] in ["mdd", "mdd-tts"]:
                # NOTE: none pad
                batch["prompts"] = processor(batch["prompt"]).input_ids
                batch["prompts_text"] = batch["prompt"]
                batch["detection_targets"] = batch["detection_targets"]
                batch["detection_targets_ppl"] = batch["detection_targets_ppl"]
                
                assert len(batch["detection_targets"]) == len(batch["prompts"])
                assert len(batch["detection_targets_ppl"]) == len(batch["labels"])

        return batch

    # test set
    model_name = "-".join(model_args["model_path"].split("/")[-2:])
    test_basename = os.path.basename(args.test_json).split('.')[0]
    test_dataset_path = os.path.dirname(args.test_json) + "/{}/{}_dataset".format(model_name,test_basename)
    
    if not os.path.exists(test_dataset_path + "/dataset.arrow"):
        print("[INFO] Loading data from {} ...".format(args.test_json))
        te_dataset = make_dataset(args.test_json, model_args)
        te_dataset = te_dataset.map(preprocess_function, num_proc=args.nj, remove_columns=['audio'])
        te_dataset.save_to_disk(test_dataset_path)
    else:
        print("[INFO] {} exists, using it".format(test_dataset_path + "/dataset.arrow"))
        te_dataset = load_from_disk(test_dataset_path)

    def get_align_info(audio, processor, tokenizer, transcript, probs):   
        #  return [{"token" : w, "start" : p[0], "end" : p[1], "conf" : p[2]} for w,p in zip(words, segments)]
        
        seg_info = get_word_timestamps(
                        samplerate=16000,
                        audio=audio,
                        processor=processor,
                        tokenizer=tokenizer,
                        transcript=transcript,
                        probs=probs
                    )
        

        conf_list = " ".join([ "{:.2f}".format(time_info["conf"]) for time_info in seg_info ])
        
        return conf_list

    # forward
    def predict(batch):
        with torch.no_grad():
            # NOTE: mdd task, add prompt
            input_values = torch.tensor(batch["input_values"], device=device).unsqueeze(0)
            labels = torch.tensor(batch["labels"], device=device).unsqueeze(0)
            if model_args["task_type"] == "mdd":
                prompts = torch.tensor(batch["prompts"], device=device).unsqueeze(0)
                output = model(input_values, prompts=prompts, labels=None, return_dict=True)
            else:
                output = model(input_values, return_dict=True)
            logits = output.logits
            logits_detection = output.logits_detection_ppl # NOTE: detect
        
        #pred_ids = torch.argmax(logits, dim=-1)

        #batch["hyp"] = processor.batch_decode(pred_ids)[0]
        # greedy_decoded_output, greedy_decoded_offsets = greedy_decoder.decode(logits)
        decoded_output, decoded_offsets = decoder.decode(logits)
        batch["hyp"] = decoded_output[0][0]
        batch["ref"] = processor.decode(batch["labels"], group_tokens=False)
        
        logits = logits.cpu()
        probs = torch.nn.functional.softmax(logits,dim=-1)[0]
        
        if len(decoded_output[0]) > 1:
            batch["nbest"] = decoded_output[0]
            batch["scores"] = decoder.get_scores()
        # beam_decoded_output, beam_decoded_offsets = beam_decoder(logits)
        batch["hyp_conf"] = get_align_info(input_values[0], processor, processor.tokenizer, batch["hyp"], probs)
        batch["ref_conf"] = get_align_info(input_values[0], processor, processor.tokenizer, batch["ref"], probs)

        # NOTE: post-processing -> remove sil
        if args.remove_sil:
            # remove sil
            batch["hyp_nosil"] = re.sub(r'sil', "", batch["hyp"])
            batch["ref_nosil"] = re.sub(r'sil', "", batch["ref"])
            # strip space
            batch["hyp_nosil"] = " ".join(batch["hyp_nosil"].split())
            batch["ref_nosil"] = " ".join(batch["ref_nosil"].split())
            # conf
            batch["hyp_nosil_conf"] = get_align_info(input_values[0], processor, processor.tokenizer, batch["hyp_nosil"], probs)
            batch["ref_nosil_conf"] = get_align_info(input_values[0], processor, processor.tokenizer, batch["hyp_nosil"], probs)
        
        return batch

    # output [pred] [label] is list
    results = te_dataset.map(predict)

    # NOTE: output all result
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # recognition / reference results w/ sil
    with open(args.result_dir + '/hyp', "w") as wf:
        for i in range(len(results)):
            wf.write(results["id"][i] + " " + results["hyp"][i] + "\n")
    with open(args.result_dir + '/ref', "w") as wf:
        for i in range(len(results)):
            wf.write(results["id"][i] + " " + results["ref"][i] + "\n")

    with open(args.result_dir + '/hyp_conf', "w") as wf:
        for i in range(len(results)):
            wf.write(results["id"][i] + " " + results["hyp_conf"][i] + "\n")
    with open(args.result_dir + '/ref_conf', "w") as wf:
        for i in range(len(results)):
            wf.write(results["id"][i] + " " + results["ref_conf"][i] + "\n")
    # compute metrics results
    total_metrics = {}
    compute_metrics(total_metrics, results["hyp"], results["ref"])

    # recognition / reference results w/o sil
    if args.remove_sil:
        with open(args.result_dir + '/hyp_nosil', "w") as wf:
            for i in range(len(results)):
                wf.write(results["id"][i] + " " + results["hyp_nosil"][i] + "\n")
        with open(args.result_dir + '/ref_nosil', "w") as wf:
            for i in range(len(results)):
                wf.write(results["id"][i] + " " + results["ref_nosil"][i] + "\n")
        # compute metrics results
        total_metrics_nosil = {}
        compute_metrics(total_metrics_nosil, results["hyp_nosil"], results["ref_nosil"])
        
        with open(args.result_dir + '/hyp_nosil_conf', "w") as wf:
            for i in range(len(results)):
                wf.write(results["id"][i] + " " + results["hyp_nosil_conf"][i] + "\n")
        with open(args.result_dir + '/ref_nosil_conf', "w") as wf:
            for i in range(len(results)):
                wf.write(results["id"][i] + " " + results["ref_nosil_conf"][i] + "\n")

    with open(args.result_dir + "/asr_result.txt", "w") as wf:
        wf.write("WER: {:.3f}\n".format(total_metrics["wer"]))
        wf.write("CER: {:.3f}\n".format(total_metrics["cer"]))
        if args.remove_sil:
            wf.write("w/o silence WER: {:.3f}\n".format(total_metrics_nosil["wer"]))
            wf.write("w/o silence CER: {:.3f}\n".format(total_metrics_nosil["cer"]))
    
    if args.decoded_type == "beam":
        nbest_dict = {}
        
        for i in range(len(results)):
            nbest_dict[results["id"][i]] = {"nbest": results["nbest"][i], "scores": results["scores"][i]}
        
        with open(args.result_dir + "/nbest.json", "w") as wf:
            json.dump(nbest_dict, wf, indent=2)
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--remove-sil', action="store_true")
    parser.add_argument('--test-json', type=str)
    parser.add_argument('--train-conf', type=str)
    parser.add_argument('--model-path', type=str, default="facebook/wav2vec2-base")
    parser.add_argument('--result-dir', type=str)
    parser.add_argument('--nj', type=int, default=4)
    parser.add_argument('--decoded_type', type=str, default="beam", choices=["beam", "greedy"])
    args = parser.parse_args()

    main(args)
