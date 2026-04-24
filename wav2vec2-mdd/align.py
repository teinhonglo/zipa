#!/bin/python
import os
import re
import json
import argparse
import numpy as np

import torch
import ctc_segmentation
from datasets import load_from_disk
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer


from models.wav2vec2_model import AutoMDDModel, AutoProtoMDDModel
from utils import make_dataset, load_from_json

def get_word_timestamps(
    samplerate : int,
    audio: np.ndarray,
    processor : Wav2Vec2Processor,
    tokenizer : Wav2Vec2CTCTokenizer,
    transcript : str,
    probs: np.ndarray
):
    assert audio.ndim == 1
         
    # Split the transcription into words
    words = transcript.split()
    
    # Align
    vocab = tokenizer.get_vocab()
    inv_vocab = {v:k for k,v in vocab.items()}
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio.shape[0] / probs.size()[0] / samplerate
    
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_tokenized_text(config, words)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
    segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, words)
    return [{"token" : w, "start" : p[0], "end" : p[1], "conf" : p[2]} for w,p in zip(words, segments)]

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
    if model_args["model_type"] == "prototype":
        model = AutoProtoMDDModel(model_args, config=config).to(device)
    else:
        model = AutoMDDModel(model_args, config=config).to(device)
    model.load_state_dict(torch.load(best_model_path+"/pytorch_model.bin", map_location=device))
    model.eval()

    # loading test set
    def preprocess_function(batch):
        audio = batch["audio"]
        # extract features return input_values
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
            # NOTE: mdd task, add prompt
            if model_args["task_type"] == "mdd":
                # NOTE: none pad
                batch["prompts"] = processor(batch["prompt"]).input_ids

                # NOTE: pad to max
                """
                prompt_features = [{"input_ids": batch["prompts"]}]
                prompts_batch = processor.pad(
                    prompt_features,
                    padding="max_length",
                    max_length=79,  # max len in tr and cv
                    pad_to_multiple_of=None,
                    return_tensors="pt",
                )
                batch["prompts"] = prompts_batch["input_ids"][0]
                """

        return batch

    # test set
    test_basename = os.path.basename(args.test_json).split('.')[0]
    test_dataset_path = os.path.dirname(args.test_json) + "/{}_dataset".format(test_basename)
    if not os.path.exists(test_dataset_path + "/dataset.arrow"):
        print("[INFO] Loading data from {} ...".format(args.test_json))
        te_dataset = make_dataset(args.test_json, model_args)
        te_dataset = te_dataset.map(preprocess_function, num_proc=args.nj)
        te_dataset.save_to_disk(test_dataset_path)
    else:
        print("[INFO] {} exists, using it".format(test_dataset_path + "/dataset.arrow"))
        te_dataset = load_from_disk(test_dataset_path)

    # forward
    def predict(batch):
        with torch.no_grad():
            # NOTE: mdd task, add prompt
            input_values = torch.tensor(batch["input_values"], device=device).unsqueeze(0)
            if model_args["task_type"] == "mdd":
                prompts = torch.tensor(batch["prompts"], device=device).unsqueeze(0)
                output = model(input_values, prompts=prompts, return_dict=True)
            else:
                output = model(input_values, return_dict=True)
            logits = output.logits

        pred_ids = torch.argmax(logits, dim=-1)

        batch["hyp"] = processor.batch_decode(pred_ids)[0]
        batch["ref"] = processor.decode(batch["labels"], group_tokens=False)

        # NOTE: post-processing -> remove sil
        if args.remove_sil:
            # remove sil
            batch["hyp_nosil"] = re.sub(r'sil', "", batch["hyp"])
            batch["ref_nosil"] = re.sub(r'sil', "", batch["ref"])
            # strip space
            batch["hyp_nosil"] = " ".join(batch["hyp_nosil"].split())
            batch["ref_nosil"] = " ".join(batch["ref_nosil"].split())

        return batch

    def align(batch):
        with torch.no_grad():
            # NOTE: mdd task, add prompt
            input_values = torch.tensor(batch["input_values"], device=device).unsqueeze(0)
            if model_args["task_type"] == "mdd":
                prompts = torch.tensor(batch["prompts"], device=device).unsqueeze(0)
                output = model(input_values, prompts=prompts, return_dict=True)
            else:
                output = model(input_values, return_dict=True)
            logits = output.logits
        
        logits = logits.cpu()
        probs = torch.nn.functional.softmax(logits,dim=-1)[0]

        batch["hyp_ts"] = get_word_timestamps(
                                    samplerate=16000,
                                    audio=batch["audio"]["array"],
                                    processor=processor,
                                    tokenizer=processor.tokenizer,
                                    transcript=batch["hyp"],
                                    probs=probs)
        
        batch["ref_ts"] = get_word_timestamps(
                                    samplerate=16000,
                                    audio=batch["audio"]["array"],
                                    processor=processor,
                                    tokenizer=processor.tokenizer,
                                    transcript=batch["ref"],
                                    probs=probs)
        
        batch["hyp_nosil_ts"] = get_word_timestamps(
                                    samplerate=16000,
                                    audio=batch["audio"]["array"],
                                    processor=processor,
                                    tokenizer=processor.tokenizer,
                                    transcript=batch["hyp_nosil"],
                                    probs=probs)
        
        batch["ref_nosil_ts"] = get_word_timestamps(
                                    samplerate=16000,
                                    audio=batch["audio"]["array"],
                                    processor=processor,
                                    tokenizer=processor.tokenizer,
                                    transcript=batch["ref_nosil"],
                                    probs=probs)
        
        return batch

    # output [pred] [label] is list
    result_dataset_path = os.path.dirname(args.test_json) + "/{}_align".format(test_basename)
    if not os.path.exists(result_dataset_path + "/dataset.arrow") or True:
        print("[INFO] Loading data from {} ...".format(args.test_json))
        te_dataset = te_dataset.map(predict)
        results = te_dataset.map(align)
        results.save_to_disk(result_dataset_path)
    else:
        print("[INFO] {} exists, using it".format(result_dataset_path + "/dataset.arrow"))
        results = load_from_disk(result_dataset_path)
    

    # NOTE: output all result
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    

    def write_result_to_file(result_dir, results, tag):
        wf = open(result_dir + "/" + tag, "w")
        
        for i in range(len(results)):
            for time_info in results[tag][i]:
                uttid = results["id"][i]
                filled = "A" 
                start = time_info["start"]
                end = time_info["end"]
                conf = time_info["conf"]
                token = time_info["token"]
                
                wf.write("{uttid}\t{filled}\t{start:.2f}\t{end:.2f}\t{conf:.2f}\t{token}\n".format(
                                uttid=uttid, filled=filled,
                                start=start, end=end,
                                conf=conf, token=token)
                        )
        wf.close()
                    
    write_result_to_file(args.result_dir, results, "hyp_ts")
    write_result_to_file(args.result_dir, results, "ref_ts")
    write_result_to_file(args.result_dir, results, "hyp_nosil_ts")
    write_result_to_file(args.result_dir, results, "ref_nosil_ts")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--remove-sil', action="store_true")
    parser.add_argument('--test-json', type=str)
    parser.add_argument('--train-conf', type=str)
    parser.add_argument('--model-path', type=str, default="facebook/wav2vec2-base")
    parser.add_argument('--result-dir', type=str)
    parser.add_argument('--nj', type=int, default=4)
    args = parser.parse_args()

    main(args)
