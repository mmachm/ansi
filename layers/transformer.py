from typing import List

import tensorflow as tf

from layers.ansi_to_pixels import AnsiToPixelsLayer
from layers.decoder import Decoder
from layers.encoder import Encoder
from keras.layers import Conv2D, ZeroPadding2D, Softmax


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, key_dim, num_heads, dff,
                 input_sequence_length, target_sequence_length, target_vocab_length, dropout_rate=0.1):
        """
        num_layers is the height of the encoder/decoder tower (N_x in the original paper)

        key_dim is just key_dim - in other words it is the dimensionality of the attention vectors
        this is important in the positional embedding and also in other layers.

        num_heads is the duplicity of the attention heads in the attention layers

        dff is the internal dimension of the feed-forward layer. The more the better, probably.

        vocab_size means the number of "words" in the vocabulary.

        IMPORTANT What is missing here is the length of the sequence. That is ok, because this will be determined
        when the transformer is applied to some data and the value is then taken from there.

        :param num_layers:
        :param key_dim:
        :param num_heads:
        :param dff:
        :param input_vocab_size:
        :param target_vocab_size:
        :param dropout_rate:
        """
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, key_dim=key_dim,
                               num_heads=num_heads, dff=dff,
                               sequence_length=input_sequence_length,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, key_dim=key_dim,
                               num_heads=num_heads, dff=dff,
                               sequence_length=target_sequence_length,
                               dropout_rate=dropout_rate)
        self.ans_to_pixel = AnsiToPixelsLayer()

        # todo this is likely not correct as target_sequence_length
        self.final_layer = tf.keras.layers.Dense(target_vocab_length)
        self.conv1 = Conv2D(8, (16, 16), strides=(4, 4), activation='relu')
        self.conv2 = Conv2D(16, (3, 3), strides=(2, 2), activation='relu')
        self.conv3 = Conv2D(self.decoder.key_dim, (10, 10), strides=(10, 10), activation='relu')

        self.zero_padding = ZeroPadding2D(padding=(8, 8))
        self.softmax = Softmax()

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        encoder_input, decoder_input = inputs

        pixel0 = self.ans_to_pixel(encoder_input)
        pixel1 = self.zero_padding(pixel0)
        pixel2 = self.conv1(pixel1)  # (320, 640, 3) > (80, 160, 32)
        pixel3 = self.conv2(pixel2)  # (40, 80, 32)
        pixel4 = self.conv3(pixel3)  # (4, 8, d_model)

        # TODO add embedding layer for the target and figure embedding for the flattened tiles
        # here the 32 is the 4*8 from the previous shape. We are in effect making this the length of the sequence
        preprocessed_encoder_input = tf.reshape(pixel4, shape=[-1, self.encoder.sequence_length, self.encoder.key_dim])

        context = self.encoder(preprocessed_encoder_input)  # (batch_size, source_length, d_model)
        outputs = self.decoder(decoder_input, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(outputs)  # (batch_size, target_len, target_vocab_size)
        char_data, color_data = tf.split(
            logits, [256, 8], axis=2
        )
        char_data = self.softmax(char_data)
        color_data = tf.sigmoid(color_data)
        output = tf.concat((char_data, color_data), axis=2)
        # Return the final output and the attention weights.
        return output
