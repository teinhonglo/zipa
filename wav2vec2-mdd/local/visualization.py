import argparse
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()

parser.add_argument("--result_root",
                    default="exp/l2arctic/train_l2arctic_baseline_wav2vec2_large_lv60_timitft_prompt",
                    type=str)

parser.add_argument("--datasets",
                    default="decode_beam_train_l2,decode_beam_dev,decode_beam_test",
                    type=str)

args = parser.parse_args()

result_root = args.result_root
datasets = args.datasets.split(",")
json_dict = {}

def merge_dict(first_dict, second_dict):
    third_dict = {**first_dict, **second_dict}
    return third_dict

for dataset in datasets: 
    with open(os.path.join(result_root, dataset, "phone_embed.json")) as fn: 
        json_dict = merge_dict(json_dict, json.load(fn))

print(list(json_dict["phone_embed"].keys()))
print(list(json_dict["phone_enc_embed"].keys()))

def draw(phn_dict, title="T-SNE of Phone Representations (L2-ARCTIC)", fig_fn="l2arctic-phones.png"):
    # 整理數據：分離特徵和標籤
    features = []
    labels = []
    for label, vectors in phn_dict.items():
        for v in vectors:
            features.append(v)
            labels.append(label)
    
    # 轉換成numpy數組以供T-SNE使用
    features = np.array(features)
    labels = np.array(labels)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    # 執行T-SNE
    tsne = TSNE(n_components=2, random_state=0)
    features_tsne = tsne.fit_transform(features)
    
    # 數據視覺化
    plt.figure(figsize=(12, 10))
    for i, label in enumerate(np.unique(labels)):
        idx = labels == label
        plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], color=colors[i], label=label)
    
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    plt.savefig(fig_fn)

draw(json_dict["phone_embed"], title="T-SNE of Phone Representations (L2-ARCTIC) - phn_embed", fig_fn="l2arctic-phone_embeds.png")
draw(json_dict["phone_enc_embed"], title="T-SNE of Phone Representations (L2-ARCTIC) - phn_enc_embed", fig_fn="l2arctic-phone_enc_embeds.png")
