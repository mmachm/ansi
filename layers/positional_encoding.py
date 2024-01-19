import numpy as np
import tensorflow as tf


def positional_encoding(seq_length, key_dim):
    key_dim = key_dim // 2  # because 1 for the sine, 1 for the cosine

    positions = np.arange(seq_length)[:, np.newaxis]     # shape (seq, 1)
    depths = np.arange(key_dim)[np.newaxis, :] / key_dim   #shape (1, key_dim)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, key_dim, sequence_length):
        super().__init__()
        self.d_model = key_dim
        self.pos_encoding = positional_encoding(seq_length=sequence_length, key_dim=key_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        x = inputs
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
