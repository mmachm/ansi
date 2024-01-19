import tensorflow as tf
from keras.layers import Dense

from layers.attention import CausalSelfAttention, CrossAttention
from layers.feed_forward import FeedForward
from layers.positional_encoding import PositionalEmbedding


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 key_dim,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate)

        self.ffn = FeedForward(key_dim, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, key_dim)`.
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, sequence_length, key_dim, num_heads, dff,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.sequence_length = sequence_length
        self.key_dim = key_dim
        self.num_layers = num_layers

        self.embedding = Dense(key_dim, activation="relu")
        self.pos_embedding = PositionalEmbedding(sequence_length=sequence_length, key_dim=key_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(key_dim=key_dim, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        # shape of x: 22, 81, 264
        x = self.embedding(x)
        # shape of x: 22, 81, key_dim
        x = self.pos_embedding(x)  # (batch_size, targ  et_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x
