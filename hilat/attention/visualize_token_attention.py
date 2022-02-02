import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import pickle5 as pickle
from transformers import AutoTokenizer

# load label dict
label_dict = pd.read_csv('../data/mimic3/50/labels_dictionary_50_level_1.csv')
# code_list = label_dict['icd9_code'].tolist()

# read raw data
with open('../data/attention/test_data_50_level_1_rand_reordered_seq-512.pkl', 'rb') as f:
    test_data = pickle.load(f)

# read prediction results
with open('../data/attention/predict_results_mimic3-50.pkl', 'rb') as f:
    predict_results = pickle.load(f)

def colorize(words, color_array):
    cmap=matplotlib.cm.RdPu
    template = '<span style="display: inline-block; color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color_weight in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color_weight)[:3])
        colored_string += template.format(color, word + '&nbsp') if color_weight>0.035 else '<span style="display: inline-block; color: black;">' + word + '&nbsp</span>'
    return colored_string

def normalize(num_list):
    min_value = min(num_list)
    max_value = max(num_list)

    for i, val in enumerate(num_list):
        num_list[i] = (val - min_value)/(max_value - min_value)*0.7


tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')

def parse_tokens(tokens):
    # input a list of tokens. output index of words
    words_indexes = []
    word_start_idx = None

    for i in range(len(tokens)):
        token = tokens[i]
        if token.startswith("▁"):
            if word_start_idx == None:
                word_start_idx = i
            else:
                word_end_idx = i
                words_indexes.append((word_start_idx, word_end_idx))
                word_start_idx = i
    return  words_indexes

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def display_attention(data, results, code_dict, tokenizer, record_no):
    tokens_attention = results['predictions'][2][record_no]
    chunk_attention = results['predictions'][3][record_no]
    true_labels = results['labels'][record_no]
    pred_labels = np.rint(results['predictions'][1][record_no])

    input_ids = (data.input_ids)[record_no]

    tokens = []
    for i in range(len(input_ids)):
        tokens.append(tokenizer.convert_ids_to_tokens(input_ids[i], True))

    html_text = "<div>"
    for i in range(len(true_labels)):
        if true_labels[i] == 1 and pred_labels[i] == 1:
            # code i
            code = code_dict.iloc[i]
            code_no = code['icd9_code']
            code_description = code['long_title']
            tokens_attention_code = tokens_attention[:, i, :]
            chunk_attetion_code = chunk_attention[i]

            concat_tokens = []
            concat_attention = []

            for i in range(len(tokens)):
                chunk_att = chunk_attetion_code[i]
                concat_tokens.extend(tokens[i])
                attention_numbers = tokens_attention_code[i, :len(tokens[i])] * chunk_att

                concat_attention.extend(attention_numbers)

            words_idx = parse_tokens(concat_tokens)

            words = []
            word_attention = []

            for index in words_idx:
                word = concat_tokens[index[0]: index[1]]
                word = (''.join(word)).replace('▁', '')
                attention = concat_attention[index[0]: index[1]]
                words.append(word)
                word_attention.append(sum(attention))

            normalize(word_attention)

            att_text = colorize(words, word_attention)

            code_text = "<span style='color: blue; font-weight: bold'>Code: {} - {}</span><br/>".format(code_no, code_description)
            html_text = html_text + code_text + att_text + '<br/>'

    html_text = html_text + '</div>'
    file_name = 'att_text_record-{}.html'.format(record_no)
    with open('../data/attention/2022_1/'+file_name, 'w', encoding='utf-8') as f:
        f.write(html_text)
    print('Done')


for i in range(20):
    display_attention(test_data, predict_results, label_dict, tokenizer, i)



