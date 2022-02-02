import ast
import csv
import linecache
import pickle
import pandas as pd

def generate_lazy_dataset_file(pickle_data_file, output_file):

    # import pre-processed pickle file
    with open(pickle_data_file, 'rb') as f:
        dataset = pickle.load(f)
        input_ids = [input for input in dataset.input_ids]
        attention_mask = [mask for mask in dataset.attention_mask]
        token_type_ids = [token_type for token_type in dataset.token_type_ids]
        targets = dataset.labels

    dataset_list = [row for row in zip(input_ids, attention_mask, token_type_ids, targets)]

    with open(output_file, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(dataset_list)

# generate_lazy_dataset_file("../../data/mimic2/train_data_full_level_1_seq-512.pkl", "../../data/mimic2/train_data_full_level_1_lazy.csv")
# generate_lazy_dataset_file("../../data/mimic2/dev_data_full_level_1_seq-512.pkl", "../../data/mimic2/dev_data_full_level_1_lazy.csv")
# generate_lazy_dataset_file("../../data/mimic2/test_data_full_level_1_seq-512.pkl", "../../data/mimic2/test_data_full_level_1_lazy.csv")

# generate_lazy_dataset_file("../../data/mimic3/50/train_data_50_level_1_rand_reordered_seq-512.pkl", "../../data/mimic3/50/train_data_50_level_1_rand_reordered_lazy.csv")
# generate_lazy_dataset_file("../../data/mimic3/50/dev_data_50_level_1_rand_reordered_seq-512.pkl", "../../data/mimic3/50/dev_data_50_level_1_rand_reordered_lazy.csv")
# generate_lazy_dataset_file("../../data/mimic3/50/test_data_50_level_1_rand_reordered_seq-512.pkl", "../../data/mimic3/50/test_data_50_level_1_rand_reordered_lazy.csv")

generate_lazy_dataset_file("../../data/mimic3/full/train_data_full_level_1_reordered_seq-512.pkl", "../../data/mimic3/full/train_data_full_level_1_reordered_lazy.csv")
generate_lazy_dataset_file("../../data/mimic3/full/dev_data_full_level_1_reordered_seq-512.pkl", "../../data/mimic3/full/dev_data_full_level_1_reordered_lazy.csv")
generate_lazy_dataset_file("../../data/mimic3/full/test_data_full_level_1_reordered_seq-512.pkl", "../../data/mimic3/full/test_data_full_level_1_reordered_lazy.csv")

