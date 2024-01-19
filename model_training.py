import tensorflow as tf
from tensorflow import Tensor

from converters.ans_from_int import AnsFromNumConverter, AnsFromVecConverter
from converters.ans_to_int import AnsToNumConverter
from layers.ansi_to_pixels import transform_onehot_to_pixelmap
from layers.transformer import Transformer
from translators.translation_utils import CharToVecTranslator
import numpy as np

error_dict = {}

#if not tf.executing_eagerly():
#    tf.enable_eager_execution()

BUFFER_SIZE = 20000
BATCH_SIZE = 6
epochs = 1
load_existing = True

def filter_batch_size(x, y, z):
    batch_size = tf.shape(x)[0]
    return tf.py_function(lambda size: size > 0, [batch_size], tf.bool)

ans_to_num_converter = AnsToNumConverter(translator=CharToVecTranslator())
data = (
    tf.data.Dataset
        #.list_files('ansi_files/1990/1990_ABYSS1.ANS')
        .list_files('ansi_files/*/*.ANS')
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        # FUCKING IMPORTANT!!! Make sure that the Tout argument matches the output shape!
        # also, unfortunately ((tf.float32, tf.float32), tf.float32) is not possible due to
        # https://github.com/tensorflow/tensorflow/issues/36276
        # hence the desired format must be achieved with the second map.
        .map(lambda file_path: tf.py_function(ans_to_num_converter.convert_tensor_of_filepaths, [file_path],
                                              Tout=(tf.float32, tf.float32, tf.float32)))
        .filter(filter_batch_size)  # important!!! a lambda will not have info about runtime for some reason??
        .map(lambda x, y, z: ((x, y), z))  # TODO understand why the 3-tuple is already expanded coming into the lambda
        .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test = data.take(1)
print()

num_layers = 2
key_dim = 16
dff = 16
num_heads = 3
dropout_rate = 0.1

target_vocab_length = 264

if load_existing:
    transformer = tf.keras.models.load_model('saved_ansi_model', compile=False)
else:
    transformer = Transformer(
        num_layers=num_layers,
        key_dim=key_dim,
        num_heads=num_heads,
        dff=dff,
        input_sequence_length=32,
        target_sequence_length=81,
        dropout_rate=dropout_rate,
        target_vocab_length=target_vocab_length
    )

def pixel_loss(label, pred):
    # shape None, 81, 264
    #print(pred.shape, label.shape)
    label = tf.expand_dims(label[:, :-1, :], axis=1)
    pred = tf.expand_dims(pred[:, :-1, :], axis=1)

    predicted_pixels = transform_onehot_to_pixelmap(pred)
    expected_pixels = transform_onehot_to_pixelmap(label)

    loss = tf.keras.losses.MeanSquaredError()(expected_pixels, predicted_pixels)
    #print(predicted_pixels.shape, expected_pixels.shape)
    return loss


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


#learning_rate = CustomSchedule(key_dim)
learning_rate = 0.001

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
transformer.compile(
    loss=pixel_loss,
    optimizer=optimizer,
    metrics=["accuracy"]
)

transformer.fit(
    data,
    epochs=epochs,
    validation_data=data
)

transformer.summary()
transformer.save("saved_ansi_model")

