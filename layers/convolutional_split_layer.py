import tensorflow as tf

from keras.layers import Conv2D
from layers.feed_forward import FeedForward
from layers.positional_encoding import PositionalEmbedding


class ConvolutionalSplitLayer(tf.keras.layers.Layer):
    def __init__(self, ):
        super().__init__()

    def call(self, raw_ansi_block):
        pass

