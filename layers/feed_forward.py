import tensorflow as tf


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, key_dim, dff, dropout_rate=0.1):
        """

        :param key_dim: this is the dimensionality of the input and also the output
        :param dff: this is a internal dimension
        :param dropout_rate:
        """
        super().__init__()
        self.seq = tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='relu'),
          tf.keras.layers.Dense(key_dim),
          tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x