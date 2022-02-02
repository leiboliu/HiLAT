# prepare random chunking data: divide the raw text into 10 chunks with 512 length for each chunk
# This is based on the segmented chunks.
# reorganize chunk order to make pertinent results, hospital course be the last 2 chunks
import pickle

import pandas as pd
from transformers import AutoTokenizer
from absl import flags, app
from hilat.models.utils import MimicIIIDataset

FLAGS = flags.FLAGS
flags.DEFINE_bool("reordered", default=True,
                  help="whether move the three diagnosis sections to the beginning of the documents")
flags.DEFINE_integer("max_seq_length", default=512, help="the maximum sequence length")
flags.DEFINE_string("tokenizer", default="xlnet-base-cased",
                    help="tokenizer names. "
                         "xlnet-base-cased for XLNet, "
                         "emilyalsentzer/Bio_ClinicalBERT for ClinicalBERT,"
                         "")
flags.DEFINE_string("input_file", default="", help="the input train or dev or test file")
flags.DEFINE_string("output_file", default="", help="the output train or dev or test file")


def preprocess_dataset(tokenizer_name, data_file, output_file, max_seq_length, reordered=False):
    tokenzier = AutoTokenizer.from_pretrained(tokenizer_name)
    data = pd.read_csv(data_file)
    text = data.loc[:, data.columns.str.startswith("Chunk")]
    if reordered:
        text = text[
            ["Chunk8", "Chunk1", "Chunk2", "Chunk3", "Chunk4", "Chunk5", "Chunk6", "Chunk7", "Chunk9", "Chunk10"]]

    text = text.fillna("").apply(lambda x: [seg for seg in x], axis=1).tolist()

    # combine 10 chunks into one chunk
    text = [' '.join(doc) for doc in text]
    labels = data.iloc[:, 11:].apply(lambda x: [seg for seg in x], axis=1).tolist()
    dataset = segment_tokenize_dataset(tokenzier, text, labels, max_seq_len=max_seq_length)
    with open(output_file, 'wb') as f:
        pickle.dump(MimicIIIDataset(dataset), f)
    print("Successfully export dataset {}".format(output_file))


def segment_tokenize_inputs(text, tokenizer, num_chunks=10, max_seq_len=512):
    # input is full text of one document
    tokenized_texts = []
    tokens = tokenizer.tokenize(text)
    start_idx = 0
    for i in range(num_chunks):
        if start_idx > len(tokens):
            tokenized_texts.append([])
            continue
        tokenized_texts.append(tokens[start_idx:(start_idx + 510)])
        start_idx += 510

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


def segment_tokenize_dataset(tokenizer, text, labels, max_seq_len=512, num_chunks=10):
    data = list(
        map(lambda t: segment_tokenize_inputs(t, tokenizer, num_chunks=num_chunks, max_seq_len=max_seq_len), text))
    input_ids, attention_mask, token_type_ids = zip(*data)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "targets": labels
    }

def main():
    reordered = FLAGS.reordered
    max_seq = FLAGS.max_seq_length
    tokenizer_name = FLAGS.tokenizer_name
    input_file = FLAGS.input_file
    output_file = FLAGS.output_file

    preprocess_dataset(tokenizer_name, input_file,
                       output_file, max_seq, reordered)

if __name__ == '__main__':
    app.run(main)
