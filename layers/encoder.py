import tensorflow as tf

from layers.attention import GlobalSelfAttention
from layers.feed_forward import FeedForward
from layers.patch_embedding import PatchEmbedding
from layers.positional_encoding import PositionalEmbedding


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, key_dim, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate)

        self.ffn = FeedForward(key_dim, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, sequence_length, key_dim, num_heads,
                 dff, dropout_rate=0.1):
        super().__init__()

        self.sequence_length = sequence_length
        self.key_dim = key_dim
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(sequence_length=sequence_length, key_dim=key_dim)

        self.enc_layers = [
            EncoderLayer(key_dim=key_dim,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        # at this point the shape of the input x is (None, 81, 264)
        # the last dimension needs to become key_dim

        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x
