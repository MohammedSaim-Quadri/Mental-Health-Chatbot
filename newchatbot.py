import json
import string
import logging
import numpy as np
import nltk
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # Change the import statement
from tensorflow.keras import regularizers

nltk.download('punkt')
nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Hyperparameters (adjust as needed)
embedding_dim = 64
max_len = 20  # Maximum sentence length


def preprocess_text(sentence):
    stemmer = nltk.stem.LancasterStemmer()
    tokens = nltk.word_tokenize(sentence)
    tokens = [stemmer.stem(word.lower()) for word in tokens if word.lower() not in nltk.corpus.stopwords.words('english') and word not in string.punctuation]
    return tokens


def create_vocabulary(data):
    words = []
    labels = []
    for intent in data["intents"]:
        labels.append(intent["tag"])
        for pattern in intent["patterns"]:
            tokens = preprocess_text(pattern)
            words.extend(tokens)

    words = sorted(set(words))
    word_index = {word: i for i, word in enumerate(words)}

    return words, labels, word_index


def prepare_data(data, word_index):
    labels = []
    training = []
    output = []

    for intent in data["intents"]:
        labels.append(intent["tag"])
        for pattern in intent["patterns"]:
            tokens = preprocess_text(pattern)
            input_seq = [word_index[word] for word in tokens if word in word_index]
            training.append(input_seq)
            output.append(labels.index(intent["tag"]))

    training = tf.keras.preprocessing.sequence.pad_sequences(training, maxlen=max_len)

    return np.array(training), np.array(output)


def create_model(vocab_size, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.1)),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(loss=SparseCategoricalCrossentropy(), optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])  # Use SparseCategoricalCrossentropy
    return model


if __name__ == "__main__":
    print(tf.__version__)

    with open("intents.json", "r") as file:
        data = json.load(file)

    words, labels, word_index = create_vocabulary(data)
    vocab_size = len(word_index)
    num_classes = len(labels)

    training, output = prepare_data(data, word_index)

    model = create_model(vocab_size, num_classes)

    model.fit(training, output, epochs=100, batch_size=32, validation_split=0.2)

    print("Chatbot is ready!")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            break

        # Preprocess user input
        tokens = preprocess_text(user_input)
        input_seq = [word_index[word] for word in tokens if word in word_index]
        input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=max_len)

        # Predict intent
        result = model.predict(input_seq)
        result_index = np.argmax(result)
        tag = labels[result_index]
        confidence = np.max(result)

        if confidence > 0.7:
            for intent in data['intents']:
                if intent['tag'] == tag:
                    response = np.random.choice(intent['responses'])
                    print(f"Bot: {response}")
        else:
            print("Bot: I am not sure I understand. Can you rephrase or try a different question?")
