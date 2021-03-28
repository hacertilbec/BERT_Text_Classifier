from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer
import tensorflow as tf
from model_training import load_bert_inputs
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


def decode_input_ids(input_ids, bert_tokenizer):
    return [bert_tokenizer.decode(input_id).replace("[PAD]","").replace("[CLS]","").replace("[SEP]","").strip() for input_id in input_ids]


def load_label_encoder():
    le = LabelEncoder()
    le.classes_ = np.load('model/classes.npy',allow_pickle=True)
    return le


def load_trained_model():
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

    trained_model = TFBertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-128k-uncased",num_labels=6)
    trained_model.compile(loss=loss,optimizer=optimizer, metrics=[metric])
    trained_model.load_weights("model/bert_model.h5")
    return trained_model

def predict():
    # load Turkish BERT tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(
        "dbmdz/bert-base-turkish-128k-uncased"
    )
    # load label encoder
    le = load_label_encoder()
    # load trained model
    trained_model = load_trained_model()

    # load processed test inputs
    input_ids, labels, attention_masks = load_bert_inputs(
        "inputs/test/bert_inp.pkl",
        "inputs/test/bert_mask.pkl",
        "inputs/test/bert_label.pkl",
    )

    # predict classes
    preds = trained_model.predict([input_ids,attention_masks],batch_size=32)
    # get the class with highest probability as predicted class
    pred_labels = preds.logits.argmax(axis=1)

    # save prediction to the output folder
    pred_df = pd.DataFrame([decode_input_ids(input_ids,bert_tokenizer), le.inverse_transform(labels), le.inverse_transform(pred_labels)]).T
    pred_df.columns = ["Text","Actual Label","Predicted Label"]
    pred_df.to_excel("outputs/predictions.xlsx")

    # report performance of the model on the test data
    acc = accuracy_score(labels,pred_labels)
    print('Accuracy score',acc)
    print('Classification Report')
    print(classification_report(labels,pred_labels,target_names=le.classes_.tolist()))

if __name__ == "__main__":
    predict()

