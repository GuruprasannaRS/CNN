# import required packages
import os
import re
import string
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import numpy as np

if __name__ == "__main__":
    # 1. Load your saved model
    model = load_model('model/21051082_NLP_model.h5')

    # 2. Load your testing data
    path = './data/aclImdb'
    names = ['neg', 'pos']

    # Function to load data from files and preprocess it
    def load_data(path, folders):
        texts, labels = [], []
        for idx, label in enumerate(folders):
            folder_path = os.path.join(path, label)
            for fname in os.listdir(folder_path):
                file_path = os.path.join(folder_path, fname)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    clean_html = re.sub('<.*?>', '', text)
                    clean_special_characters = re.sub(r'[^\w\s]', '', clean_html).lower()
                    texts.append(clean_special_characters)
                    labels.append(label)
        return texts, labels

    # Load and preprocess training and testing data
    train_texts, train_labels = load_data(os.path.join(path, 'train'), names)
    test_texts, test_labels = load_data(os.path.join(path, 'test'), names)

    # Created a tokenizer and convert test data to sequences of integers
    tokens = Tokenizer()
    tokens.fit_on_texts(train_texts)
    x_test_seq = tokens.texts_to_sequences(test_texts)

    # Pad sequences to ensure uniform length for input to the model
    max_seq_len = 1000
    x_test_pad = pad_sequences(x_test_seq, maxlen=max_seq_len)

    # Used LabelEncoder to convert string labels to integer labels and one-hot encode the integer labels
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    # Shuffled the testing data
    test_indices = np.random.permutation(len(x_test_pad))
    x_test_shuffled = x_test_pad[test_indices]
    y_test_shuffled = test_labels_encoded[test_indices]

    # 3. Run prediction on the test data and print the test accuracy
    scores = model.evaluate(x_test_shuffled, y_test_shuffled)
    print("Test Loss:", scores[0])
    print("Test Accuracy:", scores[1])
