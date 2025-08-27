# import required packages
import os
import re
import string
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.layers import Embedding, LSTM, Dense, Bidirectional, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D
import numpy as np

if __name__ == "__main__": 
    # 1. load your training data
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


    # Created a tokenizer and converted text data to sequences of integers
    tokens = Tokenizer()
    tokens.fit_on_texts(train_texts)
    x_train_seq = tokens.texts_to_sequences(train_texts)
    x_test_seq = tokens.texts_to_sequences(test_texts)
    all_sequences = x_train_seq + x_test_seq

    # Calculated the maximum sequence length for each sequence
    sequence_lengths = [len(seq) for seq in all_sequences]
    max_sequence_length = max(len(seq) for seq in all_sequences)
    average_max_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    print("Maximum Sequence Length:", max_sequence_length)
    print("Average Maximum Sequence Length:", average_max_sequence_length)

    # Pad sequences to ensure uniform length for input to the model
    max_seq_len = 1000 # we set the maximum sequence length of 1000
    x_train_pad = pad_sequences(x_train_seq, maxlen=max_seq_len)
    x_test_pad = pad_sequences(x_test_seq, maxlen=max_seq_len)

    # Use LabelEncoder to convert string labels to integer labels
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    train_labels_encoded = label_encoder.transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)
    
    # 2. Train your network

    # Splitted the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train_pad, train_labels_encoded, test_size=0.3, random_state=42)

    # Shuffled the training data
    train_indices = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[train_indices]
    y_train_shuffled = y_train[train_indices]

    # Shuffled the validation data
    val_indices = np.random.permutation(len(x_val))
    x_val_shuffled = x_val[val_indices]
    y_val_shuffled = y_val[val_indices]

    # Build the NLP model using Convolutional Neural Network (CNN) architecture
    word_size = len(tokens.word_index) + 1
    model = Sequential()
    model.add(Embedding(word_size, 16))
    model.add(Dropout(0.2))
    model.add(Conv1D(16, 2, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    # Compiled the model with Adam optimizer and binary cross-entropy loss
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model using the shuffled training data and validate on the shuffled validation data
    history = model.fit(x_train_shuffled, y_train_shuffled, batch_size=512, epochs=20, validation_data=(x_val_shuffled, y_val_shuffled))

    # Get the training and validation loss from the history object 
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the loss values
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Get the training and validation accuracy from the history object
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Plot the accuracy values
    plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # 3. Save your model
    model.save("model/21051082_NLP_model.h5")
    print('model saved successfully')
