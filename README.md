# BERT Text Classifier
This is an example of Text Classification model by fine-tuning BERT(Bidirectional Encoder Representations from Transformers).

In the project, there are **6 categories** to predict. These categories are:
1. Phone number (ex. +905331234567, 05331234567, 5331234567, ...)
2. Republic of Turkey Identity Number (ex. 27131846478, ...)
3. Hobbies (ex. Balık tutmak, video oyunları oynamak, yürüyüş yapmak, ...)
4. Amount (ex. 15 TL, 20 USD, $20, 250000 TL, ...)
5. Credit Card Number (ex. 1234 5678 9012 3456, 12345678 9012 3456...)
6. Blood type (ex. 0 RH+, A RH-, A RH Negative)


* The project is implemented in **python** language and the classification model build with **keras**.
* Used BERT pretrained model: **"dbmdz/bert-base-turkish-128k-uncased"**

## Folder Structure

    .
    ├── data
    │   ├── data                # contains some data files for creating a sample training data
    │   │   └── ...
    │   ├── train               # contains a sample train data
    │   │   └── ...
    │   ├── test                # contains a sample test data
    │   │   └── ...
    ├── inputs
    │   ├── train               # folder to keep processed train data inputs
    │   │   └── ...
    │   ├── test                # folder to keep processed test data inputs
    │   │   └── ...
    ├── outputs                 # folder to keep prediction outputs
    │       └── ...
    ├── model                   # contains saved trained model and label encoder model
    │       └── ...
    ├── scripts                 # some scripts to generate fake data
    │       └── ...
    ├── data_collection.py
    ├── data processing.py
    ├── model_training.py
    ├── model_evaluation.py
    ├── requirement.txt
    └── README.md

    
## Prerequisites

* python 3.7.2
* You can download the trained model (trained with the sample train data) from [here](https://drive.google.com/file/d/1YzAmO-mfLLthpPF4YKjgI6E4lnFnQL4C/view?usp=sharing). You should put the trained model file under *model* directory.


## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/hacertilbec/BERT_Text_Classifier.git
   ```
2. Install required python packages
   ```sh
   pip install -r requirements.txt
   ```
   
## Usage

*data_collection.py script used to generate a sample training data.*

1. In the project, there are sample train and test data files. If you want to use your own train and test dataset you should check the files under *data/train* and *data/test* directories and prepare your train and test dataset with the same structure. 
2. In order to process your train and test data files, you should run data_processing.py script with the following commands:
   ```sh
   python data_processing.py --input_filepath [train_data_filepath.xlsx] --type train
   python data_processing.py --input_filepath [test_data_filepath.xlsx] --type test
   ```
3. You can train the classification model with running model_training.py script as below:
   ```sh
   python model_training.py
   ```
4. You can test the model performance on the test set as below:
   ```sh
   python model_evaluation.py
   ```
model_evaluation.py script will print out the performance summary table and save an excel file under the **output** directory. Saved excel file will contain texts, actual labels and predicted labels of the test data.

**NOTE:** If you want to use provided trained model in the project and test the model with your test data, you should just process your test data (step 2) then you can evaluate the model (step 4).

