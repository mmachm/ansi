import tensorflow as tf
import numpy as np
from PIL import Image

input_image = Image.open("sample_chars.png")
pixels = np.array(input_image)

chars_by_number = []

for char_number in range(256):
    col = char_number % 80
    row = char_number // 80
    chars_by_number.append(pixels[16*row:16*(row+1), 8*col:8*(col+1)])

char_matrix = tf.convert_to_tensor(
    np.array(
        [char[:, :, 0] for char in chars_by_number]
    ),
    tf.float32
)


class AnsiToPixelsLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AnsiToPixelsLayer, self).__init__()

    def call(self, data):
        "Expected data shape (20, 80, 264)"
        pixel_map = transform_onehot_to_pixelmap(data)
        pixel_map = tf.reshape(pixel_map, [-1, 320, 640, 3])
        return pixel_map


def transform_onehot_to_pixelmap(data):
    # shape of data None, ?20, 80, 264
    try:
        char_data, fg_bold, fg_color, bg_bold, bg_color = tf.split(
            data, [256, 1, 3, 1, 3], axis=3
        )
    except Exception as ex:
        raise

    fg_color = handle_colors(fg_color, fg_bold)
    bg_color = handle_colors(bg_color, bg_bold)

    try:
        raw_pixel_map = tf.tensordot(char_data, char_matrix, axes=([-1], [0]))  # pixel map shape 20, 80, 16, 8
    except Exception as ex:
        raise
    raw_pixel_map = tf.transpose(raw_pixel_map, perm=[0, 1, 3, 2, 4])  # pixel map shape 20, 16, 80, 8

    raw_pixel_map = tf.reshape(raw_pixel_map, [-1, raw_pixel_map.shape[1], 16, raw_pixel_map.shape[3], 8, 1])

    pixel_map = tf.add(
        raw_pixel_map * fg_color,
        (1 - raw_pixel_map) * bg_color
    )
    # shape None, None/20, 16, 80, 8, 3
    return pixel_map

def handle_colors(color, bold):
    # bold, color have shapes (None, 20, 80, 1) and (None, 20, 80, 3)
    # bold = tf.math.add(tf.math.scalar_mul(0.5, bold, name=None), 0.5)
    bold = 0.5 * bold + 0.5
    color = bold * color
    try:
        color = tf.reshape(color, [-1, color.shape[1], 1, 80, 1, 3])
    except:
        raise
    return color
