from tensorflow import float32

from translators.translation_utils import CharToVecTranslator
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


class TrainingSelector():
    def __init__(self, examples_per_file=5):
        self.examples_per_file = examples_per_file

    def select_examples(self, ansi_array):
        encoder_inputs, decoder_inputs, labels = [], [], []
        try:
            padded_ansi = np.pad(ansi_array, pad_width=((20, 0), (1, 1), (0, 0)), constant_values=0)
        except Exception as ex:
            print(ansi_array.shape, ansi_array)
            raise
        probability_of_acceptance = 1
        if ansi_array.shape[0] > self.examples_per_file:
            probability_of_acceptance = self.examples_per_file / ansi_array.shape[0]
        for row_index in range(ansi_array.shape[0]):
            if np.random.rand() > probability_of_acceptance:
                continue
            encoder_inputs.append(padded_ansi[row_index:row_index+20, 1:-1, :])
            decoder_inputs.append(padded_ansi[row_index+20, 1:, :])
            labels.append(padded_ansi[row_index+20, :-1, :])

        encoder_inputs = tf.convert_to_tensor(np.array(encoder_inputs), dtype=float32)
        decoder_inputs = tf.convert_to_tensor(np.array(decoder_inputs), dtype=float32)
        labels = tf.convert_to_tensor(np.array(labels), dtype=float32)

        return encoder_inputs, decoder_inputs, labels

    def split_slice_into_squares(self, slice):
        pass
