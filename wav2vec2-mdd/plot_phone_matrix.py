import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_phoneme_data(file):
    with open(file, "r") as file:
       data = json.load(file)
       return data

def create_phoneme_vectors(attributes, phoneme_attribute_map):
    num_attributes = len(attributes)
    phoneme_vectors = {
        x: np.zeros(num_attributes, dtype=int)
        for x in phoneme_attribute_map
    }
    
    for phone in phoneme_vectors:
        for i, att in enumerate(attributes):
            if att in phoneme_attribute_map[phone]:
                phoneme_vectors[phone][i] = 1
    
    return phoneme_vectors

def phoneme_similarity_score(vec_a: np.array, vec_b: np.array):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    normaliser = 1 / (norm_a * norm_b)
    score = normaliser * np.dot(vec_a, vec_b)
    #score = round((score - 0.5) * 20)
    score = round((score - 0.0) * 10)
     
    return score

def create_phoneme_similarity_matrix(phoneme_vectors):
    similarity_matrix = {}
    for pha, veca in phoneme_vectors.items():
        similarity_matrix[pha] = {}
        for phb, vecb in phoneme_vectors.items():
            similarity_matrix[pha][phb] = phoneme_similarity_score(veca, vecb)
    
    return similarity_matrix

def dict_to_matrix(similarity_dict):
    phonemes = list(similarity_dict.keys())
    size = len(phonemes)
    matrix = np.zeros((size, size), dtype=int)

    for i, pha in enumerate(phonemes):
        for j, phb in enumerate(phonemes):
            matrix[i][j] = similarity_dict[pha][phb]
    
    return phonemes, matrix

def plot_heatmap(matrix, labels, output_file):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="coolwarm", fmt="d")
    plt.title("Phoneme Similarity Matrix")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

data = load_phoneme_data("prep_data/attribute_mapping.json")

attribute_list = (
                data["consonant_attributes"] + data["vowel_attributes"]
            )

phoneme_attribute_map = data["phoneme_attribute_map"]

vectors = create_phoneme_vectors(attribute_list, phoneme_attribute_map)
matrix_dict = create_phoneme_similarity_matrix(vectors)
phoneme, matrix_np = dict_to_matrix(matrix_dict)
plot_heatmap(matrix_np, phoneme, "phone_simlarity.png")
