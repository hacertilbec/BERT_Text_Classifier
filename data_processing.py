import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import pickle
import numpy as np
import os

MAX_LENGTH = 64

def process_data(filepath):
    # read labeled data
    data = pd.read_excel(filepath).astype(str)
    sentences = data.Text.values
    labels = data.label.values
    # encoding labels to unique numbers
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    # save label encoder to use it later to transform
    # actual labels
    # np.save('model/classes.npy', le.classes_)
    # loading Turkish BERT tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(
        "dbmdz/bert-base-turkish-128k-uncased"
    )
    # encode each sentence in the data using BERT tokenizer and
    # create attention mask vectors
    input_ids = []
    attention_masks = []
    for sent in sentences:
        bert_inp = bert_tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            pad_to_max_length=True,
            return_attention_mask=True,
        )
        input_ids.append(bert_inp["input_ids"])
        attention_masks.append(bert_inp["attention_mask"])

    input_ids = np.asarray(input_ids)
    attention_masks = np.array(attention_masks)
    labels = np.array(labels_encoded)

    return input_ids, labels, attention_masks


def save_bert_inputs(input_ids, labels, attention_masks, output_type):
    folder_path = os.path.join("inputs", "train" if output_type == "train" else "test")
    pickle_inp_path = os.path.join(folder_path, "bert_inp.pkl")
    pickle_mask_path = os.path.join(folder_path, "bert_mask.pkl")
    pickle_label_path = os.path.join(folder_path, "bert_label.pkl")

    pickle.dump((input_ids), open(pickle_inp_path, "wb"))
    pickle.dump((attention_masks), open(pickle_mask_path, "wb"))
    pickle.dump((labels), open(pickle_label_path, "wb"))

    print(
        "Pickle files saved as ", pickle_inp_path, pickle_mask_path, pickle_label_path
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='BERT text classification / data processing')
    parser.add_argument('--input_filepath', metavar='path', required=True,
                        help='the path to file')
    parser.add_argument('--type', metavar='path', required=True,
                        help='type of the data, either "train" or "test"')

    args = parser.parse_args()

    print("Processing input data...\n")
    input_ids, labels, attention_masks = process_data(args.input_filepath)
    print("Processing done.\n")
    print("Saving processed input files...\n")
    save_bert_inputs(input_ids, labels, attention_masks, output_type=args.type)
