from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
import numpy as np


def load_bert_inputs(pickle_inp_path, pickle_mask_path, pickle_label_path):
    input_ids = pickle.load(open(pickle_inp_path, "rb"))
    attention_masks = pickle.load(open(pickle_mask_path, "rb"))
    labels = pickle.load(open(pickle_label_path, "rb"))

    print(
        f"Input shape {input_ids.shape} Attention mask shape {attention_masks.shape} Input label shape {labels.shape}"
    )
    return input_ids, labels, attention_masks


def train_model(input_ids, labels, attention_masks):
    # number of unique classes
    num_classes = np.unique(labels.flatten()).shape[0]
    # split data as train and validation with ratio of %80, %20 respectively.
    train_inp, val_inp, train_label, val_label, train_mask, val_mask = train_test_split(
        input_ids, labels, attention_masks, test_size=0.2
    )
    print(
        "Train inp shape {} Val input shape {}\nTrain label shape {} Val label shape {}\nTrain attention mask shape {} Val attention mask shape {}".format(
            train_inp.shape,
            val_inp.shape,
            train_label.shape,
            val_label.shape,
            train_mask.shape,
            val_mask.shape,
        )
    )
    # create Sequence Classifier model using pretrained Turkish BERT model
    bert_tokenizer = BertTokenizer.from_pretrained(
        "dbmdz/bert-base-turkish-128k-uncased"
    )
    bert_model = TFBertForSequenceClassification.from_pretrained(
        "dbmdz/bert-base-turkish-128k-uncased", num_labels=num_classes
    )
    # best model's path
    model_save_path = "model/bert_model.h5"
    # ModelCheckpoint callback is used to save best model
    # EarlyStopping callback is used to avoid overfitting
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=5,
            verbose=0,
            mode="min",
            baseline=None,
            restore_best_weights=True,
        ),
    ]
    # model params
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
    # compile model with params and callbacks
    bert_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    # model training
    history = bert_model.fit(
        [train_inp, train_mask],
        train_label,
        batch_size=32,
        epochs=100,
        validation_data=([val_inp, val_mask], val_label),
        callbacks=callbacks,
    )


if __name__ == "__main__":
    print("Loading input files...\n")
    input_ids, labels, attention_masks = load_bert_inputs(
        "inputs/train/bert_inp.pkl",
        "inputs/train/bert_mask.pkl",
        "inputs/train/bert_label.pkl",
    )
    print("Training model...\n")
    train_model(input_ids, labels, attention_masks)