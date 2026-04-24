import argparse
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import pprint
import json
from scipy import stats
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import math

pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
   
# configuration.
parser.add_argument('--capt_dir', type=str, default="exp/models/tdnn/decode/capt")
parser.add_argument('--topN', type=int, default="10")

args = parser.parse_args()

capt_dir = args.capt_dir
topN = args.topN
csv_fn = os.path.join(capt_dir, "per_l1_anno_only.csv")

df = pd.read_csv(csv_fn)

top_erros = defaultdict(dict)
l1_stats = defaultdict(dict)
total_counts = defaultdict(int)

l1_colors = {
    'Spanish': '#FF5733',
    'Vietnamese': '#FFD700',
    'Hindi': '#FF8C00',
    'Mandarin': '#DC143C',
    'Korean': '#4682B4',
    'Arabic': '#32CD32'
}

for i in range(len(df["l1"])):
    l1 = df["l1"][i]
    err_type = df["err_type"][i]
    total = df["total"][i]

    err_type, err_details = err_type.split("_")
    total_counts[err_type] += int(total)

    if err_details not in top_erros[err_type]:
        top_erros[err_type][err_details] = int(total)
    else:
        top_erros[err_type][err_details] += int(total)
    
    if err_type not in l1_stats[l1]:
        l1_stats[l1][err_type] = {err_details: int(total)}
    else:
        l1_stats[l1][err_type][err_details] = int(total)

# Convert to percentages
for err_type in top_erros:
    total = total_counts[err_type]
    for err_details in top_erros[err_type]:
        top_erros[err_type][err_details] = (top_erros[err_type][err_details] / total) * 100

    for l1 in l1_stats:
        if err_type in l1_stats[l1]:
            for err_details in l1_stats[l1][err_type]:
                l1_stats[l1][err_type][err_details] = (l1_stats[l1][err_type][err_details] / total) * 100

# Generate unique random colors for each L1
def generate_unique_colors(num_colors):
    colors = set()
    while len(colors) < num_colors:
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        colors.add(color)
    return list(colors)

def plot_histogram(l1_stats, topn_errors, exp_root, err_type="S", topN=10, unique_colors=None, num_rows=1):
    l1_list = list(l1_stats.keys())

    errors_by_l1 = {}
    for l1 in l1_list:
        errors_by_l1[l1] = {}
        for de, n in topn_errors:
            if de in l1_stats[l1][err_type]:
                errors_by_l1[l1][de] = l1_stats[l1][err_type][de]
            else:
                errors_by_l1[l1][de] = 0

    error_types = [ de for de, n in topn_errors]

    # Assign a unique random color to each L1
    if unique_colors is None:
        unique_colors = generate_unique_colors(len(errors_by_l1))
        colors = {l1: unique_colors[i] for i, l1 in enumerate(errors_by_l1.keys())}
    else:
        colors = unique_colors

    # Plotting combined histogram with 90 degree rotation and different unique colors for each L1
    num_cols = math.ceil((len(errors_by_l1) + 1) / num_rows)
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(72, 20), sharey=True)
    if num_rows > 1 or num_cols > 1:
        ax = ax.flatten()

    plt.rcParams.update({'font.size': 42})

    # Plotting TOPN errors histogram
    topn_labels, topn_values = zip(*topn_errors)
    ax[0].barh(topn_labels, topn_values, color='skyblue', height=0.4)
    ax[0].set_title(f'Top {topN} Errors')
    #ax[0].set_xlabel('Count')

    # Plotting errors by L1 histogram
    for idx, (l1, errors) in enumerate(errors_by_l1.items()):
        labels, values = zip(*errors.items())
        ax[idx + 1].barh(labels, values, color=colors[l1], height=0.4)
        ax[idx + 1].set_title(f'{l1}', fontsize=48)
        ax[idx + 1].set_xlim(0, max(topn_values))

    # Hide any unused subplots
    for j in range(len(errors_by_l1) + 1, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout(pad=1.0, w_pad=0.2, h_pad=0.2)
    plt.savefig(os.path.join(exp_root, err_type + ".png"))


# D, S, I
for err_type in list(top_erros.keys()):
    example_dict = top_erros[err_type]
    sorted_keys_desc = sorted(example_dict, key=lambda x: example_dict[x], reverse=True)

    topn_errors = [(k, example_dict[k]) for k in sorted_keys_desc[:topN]]
    topn_errors = list(reversed(topn_errors))
    print(topn_errors)
    plot_histogram(l1_stats=l1_stats, topn_errors=topn_errors, 
                    exp_root=capt_dir, err_type=err_type, topN=topN, 
                    unique_colors=l1_colors, num_rows=2)
