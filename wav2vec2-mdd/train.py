#!/bin/python
import os
import json
import glob
import argparse
import random
import torch
import torchaudio
from torch.optim import Adam, AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2Processor, Wav2Vec2PhonemeCTCTokenizer
from transformers import TrainingArguments, Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_from_disk
import numpy as np

# local import
from utils import make_vocab, make_dataset, DataCollatorCTCWithPadding, cal_class_weight, load_from_json, save_to_json
from metrics_np import compute_metrics
from models.wav2vec2_model import AutoMDDModel, AutoProtoMDDModel, CfrModel
import time

vow_set = ["aa", 'ae', 'ah', 'ao', 'aw', 'ay', 'eh', 'er', 'ey', 'ih', 'iy', 'ow', 'oy', 'uh', 'uw']
#torch.autograd.set_detect_anomaly(True)

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load train_args, model_args
    training_args = load_from_json(args.train_conf)
    train_args, model_args = training_args[0], training_args[1]
    # save train_args, model_args to exp_dir
    train_conf_path = os.path.join(args.exp_dir, 'train_conf.json')
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    save_to_json(training_args, train_conf_path)
    # show the model_args
    print("[NOTE] Model args ...")
    print(json.dumps(model_args, indent=4))

    # create vocab.json
    if "pretrained_path" in model_args:
        print("[INFO] Using vocab.json from {} ...".format(model_args["pretrained_path"]))
        vocab_path = model_args["pretrained_path"] + '/vocab.json'
        vocab_dict = load_from_json(vocab_path)
        save_to_json(vocab_dict, args.exp_dir + '/vocab.json')
        model_args["num_labels"] = len(vocab_dict)
    elif not os.path.exists(args.exp_dir + '/vocab.json'):
        vocab_dict, model_args["num_labels"] = make_vocab(args.units, args.exp_dir, unk_token="err")
    else:
        vocab_dict = load_from_json(args.exp_dir + '/vocab.json')
        model_args["num_labels"] = len(vocab_dict)

    # find vow/con labels
    #vow_labels = [vocab_dict[key] for key in vocab_dict.keys() if key in vow_set]
    #con_labels = [vocab_dict[key] for key in vocab_dict.keys() if key not in vow_set]
    #model_args["vow_labels"] = vow_labels
    #model_args["con_labels"] = con_labels

    # load wav2vec2 model
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args["model_path"]
    )

    # NOTE: padd to max sequence length when task is mdd, word_delimiter_token="|",
    tokenizer = Wav2Vec2PhonemeCTCTokenizer(
        args.exp_dir + "/vocab.json",
        unk_token="err", pad_token="[PAD]",
        do_phonemize=False
    )

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # save to exp_dir
    tokenizer.save_pretrained(args.exp_dir)
    feature_extractor.save_pretrained(args.exp_dir)
    processor.save_pretrained(args.exp_dir)
    print("[INFO] Save tokenizer/extractor/processor to {} ...".format(args.exp_dir))

    # NOTE: data preprocess
    def preprocess_function(batch):
        audio = batch["audio"]
        # extract features return input_values
        batch["input_values"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

        # tokenize labels
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
            batch["labels_text"] = batch["text"]
            
            # NOTE: mdd task
            if model_args["task_type"] in ["mdd", "mdd-tts"]:
                batch["prompts"] = processor(batch["prompt"]).input_ids
                batch["prompts_text"] = batch["prompt"]
                batch["detection_targets"] = batch["detection_targets"]
                batch["detection_targets_ppl"] = batch["detection_targets_ppl"]
                batch["prompt_text_wrd"] = batch["text_wrd"]
                
                assert len(batch["detection_targets"]) == len(batch["prompts"])
                assert len(batch["detection_targets_ppl"]) == len(batch["labels"])
            
        return batch
    
    model_name = "-".join(model_args["model_path"].split("/")[-2:])
    # train set
    train_basename = os.path.basename(args.train_json).split('.')[0]
    train_basename += "_sp" if "speed_perturb" in model_args else ""
    train_dataset_path = os.path.dirname(args.train_json) + "/{}/{}_dataset".format(model_name,train_basename)

    if not os.path.exists(train_dataset_path + "/dataset.arrow"):
        print("[INFO] Loading data from {} ...".format(args.train_json))
        tr_dataset = make_dataset(args.train_json, model_args, do_augment=True)
        tr_dataset = tr_dataset.map(preprocess_function, num_proc=args.nj, remove_columns=['audio'])
        tr_dataset.save_to_disk(train_dataset_path)
    else:
        print("[INFO] {} exists, using it".format(train_dataset_path + "/dataset.arrow"))
        tr_dataset = load_from_disk(train_dataset_path, keep_in_memory=True)

    # valid set
    valid_basename = os.path.basename(args.valid_json).split('.')[0]
    valid_dataset_path = os.path.dirname(args.valid_json) + "/{}/{}_dataset".format(model_name,valid_basename)
    
    if not os.path.exists(valid_dataset_path + "/dataset.arrow"):
        print("[INFO] Loading data from {} ...".format(args.valid_json))
        cv_dataset = make_dataset(args.valid_json, model_args)
        cv_dataset = cv_dataset.map(preprocess_function, num_proc=args.nj, remove_columns=['audio'])
        cv_dataset.save_to_disk(valid_dataset_path)
    else:
        print("[INFO] {} exists, using it".format(valid_dataset_path + "/dataset.arrow"))
        cv_dataset = load_from_disk(valid_dataset_path, keep_in_memory=True)

    # NOTE: data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, task_type=model_args["task_type"]
    )

    # NOTE: class_weight cal from trainset
    if "class_weight_alpha" in model_args and model_args["class_weight_alpha"] != 0:
        print("[INFO] Use class weight alpha {} ...".format(model_args["class_weight_alpha"]))
        class_weight = cal_class_weight(tr_dataset['detection_targets'], 2, \
            alpha=model_args["class_weight_alpha"]).to(device)
    else:
        print("[INFO] No class weight is provide ...")
        class_weight = None

    # NOTE: define model
    if "pretrained_path" in model_args:
        print("[INFO] Train a {} model. Load pretrained model from {} ...".format(model_args["model_type"], model_args["pretrained_path"]))
        best_model_path = model_args["pretrained_path"] + "/best"
        if model_args["model_type"] == "prototype":
            model = AutoProtoMDDModel(model_args, class_weight=class_weight, processor=processor).to(device)
        elif model_args["model_type"] == "baseline":
            model = AutoMDDModel(model_args, class_weight=class_weight, processor=processor).to(device)
        elif model_args["model_type"] == "conformer":
            model = CfrModel(model_args, class_weight=class_weight, processor=processor).to(device)
        else:
            raise ValueError("")
        
        total_params = sum(p.numel() for p in model.parameters())
        loaded_params = total_params
        missing_keys, _ = model.load_state_dict(torch.load(best_model_path+"/pytorch_model.bin", map_location=device), strict=False)
        # Calculate the number of missing parameters
        for key in missing_keys:
            loaded_params -= model.state_dict()[key].numel()
        
        # Calculate the ratio of successfully loaded parameters
        success_ratio = loaded_params / total_params
        print("Success Load ratio", success_ratio)        
        
        if model_args["model_type"] == "prototype":
            model.init_prototypes(from_pretrain=True)
    else:
        print("[INFO] Train a {} model from {} ...".format(model_args["model_type"], model_args["model_path"]))
        if model_args["model_type"] == "prototype":
            model = AutoProtoMDDModel(model_args, class_weight=class_weight, pretrained=True, processor=processor).to(device)
            model.init_prototypes(from_pretrain=False)
        elif model_args["model_type"] == "baseline":
            model = AutoMDDModel(model_args, class_weight=class_weight, pretrained=True, processor=processor).to(device)
        elif model_args["model_type"] == "conformer":
            model = CfrModel(model_args, class_weight=class_weight, processor=processor).to(device)
        else:
            raise ValueError("")
 
    # print # of parameters
    trainables = [p for p in model.parameters() if p.requires_grad]
    print('[INFO] Total parameter number is : {:.3f} M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('[INFO] Total trainable parameter number is : {:.3f} M'.format(sum(p.numel() for p in trainables) / 1e6))
    # save model_config
    torch.save(model.config, args.exp_dir + '/config.pth')
    model.config.to_json_file(args.exp_dir + '/config.json')

    # NOTE: define metric
    def calculate_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        # metrics
        total_metrics = {}
        compute_metrics(total_metrics, pred_str, label_str)
        return total_metrics

    # NOTE: define training args
    training_args = TrainingArguments(
        output_dir=args.exp_dir,
        group_by_length=True,
        fp16=True,
        load_best_model_at_end=True,
        **train_args
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=calculate_metrics,
        train_dataset=tr_dataset,
        eval_dataset=cv_dataset,
        tokenizer=processor.feature_extractor,
    )
    
    def check_gpu_memory(required_memory_gb):
        """
        Check if there is enough GPU memory available.
        :param required_memory_gb: Memory required in GB
        :return: True if there is enough memory, False otherwise
        """
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        available_memory = gpu_memory - (torch.cuda.memory_reserved(0) / (1024 ** 3))
    
        if available_memory >= required_memory_gb:
            return True
        else:
            return False

    required_memory_gb = 20  # Change this to the amount of memory you need

    while not check_gpu_memory(required_memory_gb):
        print("Not enough GPU memory available. Checking again in 5 seconds...")
        time.sleep(5)

    # Your code to execute if there is enough GPU memory
    print("Sufficient GPU memory available. Executing next line...")

    if glob.glob(os.path.join(args.exp_dir, 'checkpoint*')):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # save the best model
    best_path = os.path.join(args.exp_dir, 'best')
    trainer.save_model(best_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', type=str, help="kaldi-format data", default="/share/nas167/fuann/asr/gop_speechocean762/s5/data/train")
    parser.add_argument('--valid-json', type=str, help="kaldi-format data", default="/share/nas167/fuann/asr/gop_speechocean762/s5/data/test")
    parser.add_argument('--train-conf', type=str)
    parser.add_argument('--seed', type=int, default=824)
    parser.add_argument('--units', type=str)
    parser.add_argument('--exp-dir', type=str, default="exp-finetune/facebook/wav2vec2-large-xlsr-53")
    parser.add_argument('--nj', type=int, default=4)
    args = parser.parse_args()

    # set seed
    print("[INFO] Set manual seed {}".format(args.seed))
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.backends.cudnn.enabled = False
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
    #torch.use_deterministic_algorithms(True)

    main(args)

