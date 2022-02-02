import csv
import linecache
import pickle
import random
import subprocess

import numpy as np
import redis
import torch
import logging
import ast

from datasets import Dataset
from tqdm import tqdm

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, auc
from torch.utils.data import DataLoader
from transformers import AutoModel, DataCollatorWithPadding, XLNetTokenizer, XLNetTokenizerFast, AutoTokenizer, \
    XLNetModel, is_torch_tpu_available

logger = logging.getLogger("lwat")


class MimicIIIDataset(Dataset):
    def __init__(self, data):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.token_type_ids = data["token_type_ids"]
        self.labels = data["targets"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[item], dtype=torch.float),
            "token_type_ids": torch.tensor(self.token_type_ids[item], dtype=torch.long),
            "targets": torch.tensor(self.labels[item], dtype=torch.float)
        }

class LazyMimicIIIDataset(Dataset):
    def __init__(self, filename, task, dataset_type):
        print("lazy load from {}".format(filename))
        self.filename = filename
        self.redis = redis.Redis(unix_socket_path="/tmp/redis.sock")
        self.pipe = self.redis.pipeline()
        self.num_examples = 0
        self.task = task
        self.dataset_type = dataset_type
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f.readlines()):
                self.num_examples += 1
                example = eval(line)
                key = task + '_' + dataset_type + '_' + str(line_num)
                input_ids = eval(example[0])
                attention_mask = eval(example[1])
                token_type_ids = eval(example[2])
                labels = eval(example[3])
                example_tuple = (input_ids, attention_mask, token_type_ids, labels)

                self.pipe.set(key, pickle.dumps(example_tuple))
                if line_num % 100 == 0:
                    self.pipe.execute()
            self.pipe.execute()
        if is_torch_tpu_available():
            import torch_xla.core.xla_model as xm
            xm.rendezvous(tag="featuresGenerated")

    def __len__(self):
        return self.num_examples

    def __getitem__(self, item):
        key = self.task + '_' + self.dataset_type + '_' + str(item)
        example = pickle.loads(self.redis.get(key))

        return {
            "input_ids": torch.tensor(example[0], dtype=torch.long),
            "attention_mask": torch.tensor(example[1], dtype=torch.float),
            "token_type_ids": torch.tensor(example[2], dtype=torch.long),
            "targets": torch.tensor(example[3], dtype=torch.float)
        }


class ICDCodeDataset(Dataset):
    def __init__(self, data):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.token_type_ids = data["token_type_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[item], dtype=torch.float),
            "token_type_ids": torch.tensor(self.token_type_ids[item], dtype=torch.long)
        }


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def tokenize_inputs(text_list, tokenizer, max_seq_len=512):
    """
    Tokenizes the input text input into ids. Appends the appropriate special
    characters to the end of the text to denote end of sentence. Truncate or pad
    the appropriate sequence length.
    """
    # tokenize the text, then truncate sequence to the desired length minus 2 for
    # the 2 special characters
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:max_seq_len - 2], text_list))
    # convert tokenized text into numeric ids for the appropriate LM
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # get token type for token_ids_0
    token_type_ids = [tokenizer.create_token_type_ids_from_sequences(x) for x in input_ids]
    # append special token to end of sentence: <sep> <cls>
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    # attention mask
    attention_mask = [[1] * len(x) for x in input_ids]

    # padding to max_length
    def padding_to_max(sequence, value):
        padding_len = max_seq_len - len(sequence)
        padding = [value] * padding_len
        return sequence + padding

    input_ids = [padding_to_max(x, tokenizer.pad_token_id) for x in input_ids]
    attention_mask = [padding_to_max(x, 0) for x in attention_mask]
    token_type_ids = [padding_to_max(x, tokenizer.pad_token_type_id) for x in token_type_ids]

    return input_ids, attention_mask, token_type_ids


def tokenize_dataset(tokenizer, text, labels, max_seq_len):
    if (isinstance(tokenizer, XLNetTokenizer) or isinstance(tokenizer, XLNetTokenizerFast)):
        data = list(map(lambda t: tokenize_inputs(t, tokenizer, max_seq_len=max_seq_len), text))
        input_ids, attention_mask, token_type_ids = zip(*data)
    else:
        tokenizer.model_max_length = max_seq_len
        input_dict = tokenizer(text, padding=True, truncation=True)
        input_ids = input_dict["input_ids"]
        attention_mask = input_dict["attention_mask"]
        token_type_ids = input_dict["token_type_ids"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "targets": labels
    }


def initial_code_title_vectors(label_dict, transformer_model_name, tokenizer_name, code_max_seq_length, code_batch_size,
                               d_model, device):
    logger.info("Generate code title representations from base transformer model")
    model = AutoModel.from_pretrained(transformer_model_name)
    if isinstance(model, XLNetModel):
        model.config.use_mems_eval = False
    #
    # model.config.use_mems_eval = False
    # model.config.reuse_len = 0
    code_titles = label_dict["long_title"].fillna("").tolist()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="right")
    data = tokenizer(code_titles, padding=True, truncation=True)
    code_dataset = ICDCodeDataset(data)

    model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer, padding="max_length",
                                            max_length=code_max_seq_length)
    code_param = {"batch_size": code_batch_size, "collate_fn": data_collator}
    code_dataloader = DataLoader(code_dataset, **code_param)

    code_dataloader_progress_bar = tqdm(code_dataloader, unit="batches",
                                        desc="Code title representations")
    code_dataloader_progress_bar.clear()

    # output shape: (num_labels, hidden_size)
    initial_code_vectors = torch.zeros(len(code_dataset), d_model)

    for i, data in enumerate(code_dataloader_progress_bar):
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.float)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)

        # output shape: (batch_size, sequence_length, hidden_size)
        output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Mean pooling. output shape: (batch_size, hidden_size)
        mean_last_hidden_state = torch.mean(output[0], 1)
        # Max pooling. output shape: (batch_size, hidden_size)
        # max_last_hidden_state = torch.max((output[0] * attention_mask.unsqueeze(-1)), 1)[0]

        initial_code_vectors[i * input_ids.shape[0]:(i + 1) * input_ids.shape[0], :] = mean_last_hidden_state

    code_dataloader_progress_bar.refresh(True)
    code_dataloader_progress_bar.clear(True)
    code_dataloader_progress_bar.close()
    logger.info("Code representations ready for use. Shape {}".format(initial_code_vectors.shape))
    return initial_code_vectors


def normalise_labels(labels, n_label):
    norm_labels = []
    for label in labels:
        one_hot_vector_label = [0] * n_label
        one_hot_vector_label[label] = 1
        norm_labels.append(one_hot_vector_label)
    return np.asarray(norm_labels)


def segment_tokenize_inputs(text, tokenizer, max_seq_len, num_chunks):
    # input is full text of one document
    tokenized_texts = []
    tokens = tokenizer.tokenize(text)
    start_idx = 0
    seq_len = max_seq_len - 2
    for i in range(num_chunks):
        if start_idx > len(tokens):
            tokenized_texts.append([])
            continue
        tokenized_texts.append(tokens[start_idx:(start_idx + seq_len)])
        start_idx += seq_len

    # convert tokenized text into numeric ids for the appropriate LM
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # get token type for token_ids_0
    token_type_ids = [tokenizer.create_token_type_ids_from_sequences(x) for x in input_ids]
    # append special token to end of sentence: <sep> <cls>
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    # attention mask
    attention_mask = [[1] * len(x) for x in input_ids]

    # padding to max_length
    def padding_to_max(sequence, value):
        padding_len = max_seq_len - len(sequence)
        padding = [value] * padding_len
        return sequence + padding

    input_ids = [padding_to_max(x, tokenizer.pad_token_id) for x in input_ids]
    attention_mask = [padding_to_max(x, 0) for x in attention_mask]
    token_type_ids = [padding_to_max(x, tokenizer.pad_token_type_id) for x in token_type_ids]

    return input_ids, attention_mask, token_type_ids


def segment_tokenize_dataset(tokenizer, text, labels, max_seq_len, num_chunks):
    data = list(
        map(lambda t: segment_tokenize_inputs(t, tokenizer, max_seq_len, num_chunks), text))
    input_ids, attention_mask, token_type_ids = zip(*data)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "targets": labels
    }


# The following functions are modified from the relevant codes of https://github.com/aehrc/LAAT
def roc_auc(true_labels, pred_probs, average="macro"):
    if pred_probs.shape[0] <= 1:
        return

    fpr = {}
    tpr = {}
    if average == "macro":
        # get AUC for each label individually
        relevant_labels = []
        auc_labels = {}
        for i in range(true_labels.shape[1]):
            # only if there are true positives for this label
            if true_labels[:, i].sum() > 0:
                fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], pred_probs[:, i])
                if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                    auc_score = auc(fpr[i], tpr[i])
                    if not np.isnan(auc_score):
                        auc_labels["auc_%d" % i] = auc_score
                        relevant_labels.append(i)

        # macro-AUC: just average the auc scores
        aucs = []
        for i in relevant_labels:
            aucs.append(auc_labels['auc_%d' % i])
        score = np.mean(aucs)
    else:
        # micro-AUC: just look at each individual prediction
        flat_pred = pred_probs.ravel()
        fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), flat_pred)
        score = auc(fpr["micro"], tpr["micro"])

    return score


def union_size(x, y, axis):
    return np.logical_or(x, y).sum(axis=axis).astype(float)


def intersect_size(x, y, axis):
    return np.logical_and(x, y).sum(axis=axis).astype(float)


def macro_accuracy(true_labels, pred_labels):
    num = intersect_size(true_labels, pred_labels, 0) / (union_size(true_labels, pred_labels, 0) + 1e-10)
    return np.mean(num)


def macro_precision(true_labels, pred_labels):
    num = intersect_size(true_labels, pred_labels, 0) / (pred_labels.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_recall(true_labels, pred_labels):
    num = intersect_size(true_labels, pred_labels, 0) / (true_labels.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_f1(true_labels, pred_labels):
    prec = macro_precision(true_labels, pred_labels)
    rec = macro_recall(true_labels, pred_labels)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


def precision_at_k(true_labels, pred_probs, ks=[1, 5, 8, 10, 15]):
    # num true labels in top k predictions / k
    sorted_pred = np.argsort(pred_probs)[:, ::-1]
    output = []
    for k in ks:
        topk = sorted_pred[:, :k]

        # get precision at k for each example
        vals = []
        for i, tk in enumerate(topk):
            if len(tk) > 0:
                num_true_in_top_k = true_labels[i, tk].sum()
                denom = len(tk)
                vals.append(num_true_in_top_k / float(denom))

        output.append(np.mean(vals))
    return output


def micro_recall(true_labels, pred_labels):
    flat_true = true_labels.ravel()
    flat_pred = pred_labels.ravel()
    return intersect_size(flat_true, flat_pred, 0) / flat_true.sum(axis=0)


def micro_precision(true_labels, pred_labels):
    flat_true = true_labels.ravel()
    flat_pred = pred_labels.ravel()
    if flat_pred.sum(axis=0) == 0:
        return 0.0
    return intersect_size(flat_true, flat_pred, 0) / flat_pred.sum(axis=0)


def micro_f1(true_labels, pred_labels):
    prec = micro_precision(true_labels, pred_labels)
    rec = micro_recall(true_labels, pred_labels)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


def micro_accuracy(true_labels, pred_labels):
    flat_true = true_labels.ravel()
    flat_pred = pred_labels.ravel()
    return intersect_size(flat_true, flat_pred, 0) / union_size(flat_true, flat_pred, 0)


def calculate_scores(true_labels, logits, average="macro", is_multilabel=True, threshold=0.5):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    pred_probs = sigmoid(logits)
    pred_labels = np.rint(pred_probs - threshold + 0.5)

    max_size = min(len(true_labels), len(pred_labels))
    true_labels = true_labels[: max_size]
    pred_labels = pred_labels[: max_size]
    pred_probs = pred_probs[: max_size]
    p_1 = 0
    p_5 = 0
    p_8 = 0
    p_10 = 0
    p_15 = 0
    if pred_probs is not None:
        if not is_multilabel:
            normalised_labels = normalise_labels(true_labels, len(pred_probs[0]))
            auc_score = roc_auc(normalised_labels, pred_probs, average=average)
            accuracy = accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels, average=average)
            recall = recall_score(true_labels, pred_labels, average=average)
            f1 = f1_score(true_labels, pred_labels, average=average)
        else:
            if average == "macro":
                accuracy = macro_accuracy(true_labels, pred_labels)  # categorical accuracy
                precision, recall, f1 = macro_f1(true_labels, pred_labels)
                p_ks = precision_at_k(true_labels, pred_probs, [1, 5, 8, 10, 15])
                p_1 = p_ks[0]
                p_5 = p_ks[1]
                p_8 = p_ks[2]
                p_10 = p_ks[3]
                p_15 = p_ks[4]

            else:
                accuracy = micro_accuracy(true_labels, pred_labels)
                precision, recall, f1 = micro_f1(true_labels, pred_labels)
            auc_score = roc_auc(true_labels, pred_probs, average)

    else:
        auc_score = -1

    output = {"{}_precision".format(average): precision, "{}_recall".format(average): recall,
              "{}_f1".format(average): f1, "{}_accuracy".format(average): accuracy,
              "{}_auc".format(average): auc_score, "{}_P@1".format(average): p_1, "{}_P@5".format(average): p_5,
              "{}_P@8".format(average): p_8, "{}_P@10".format(average): p_10, "{}_P@15".format(average): p_15}

    return output


