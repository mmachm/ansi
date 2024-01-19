from keras.layers import Input, Dense, Softmax, Embedding

from keras.preprocessing.text import one_hot
import tensorflow as tf
from tensorflow import one_hot

vocab_size = 65536 + 1  # 256*256 for colors and characters + 1 for out-of-frame tag
embedding_size = 64

def main():
    input = Input(shape=(6,))
    one_hot_encoded = one_hot(input, depth=65536 + 1)  # one hot stacked as row vectors

    embedded_vectors = Embedding(vocab_size, embedding_size)(one_hot_encoded)

    raw_prediction = Dense(units=1)(embedded_vectors)
    prediction = Softmax(raw_prediction)


if __name__ == "__main__":
    main()