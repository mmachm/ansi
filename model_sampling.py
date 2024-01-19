import numpy as np
import tensorflow as tf

from converters.ans_from_int import AnsFromVecConverter

number_of_rows = 100

def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred
    mask = label != 0
    match = match & mask
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)

@np.vectorize
def generate_bernoulli(p):
    # Generate a random number between 0 and 1
    rand_num = np.random.rand()

    # Check if the random number is less than the probability p
    if rand_num < p:
        return 1  # Event with probability p occurred
    else:
        return 0  # Event with probability 1-p occurred


def observe_character(probabilities):
    char_probabilities = probabilities[:256]
    color_probabilities = probabilities[256:]
    # Generate a random number between 0 and 1
    rand_num = np.random.rand()

    # Initialize the cumulative probability
    cumulative_prob = 0

    # Iterate over the probabilities until the cumulative probability exceeds the random number
    i = 0
    for i, prob in enumerate(char_probabilities):
        cumulative_prob += prob
        if rand_num <= cumulative_prob:
            break

    char_data = np.zeros(shape=(256,))
    char_data[i] = 1
    color_data = generate_bernoulli(color_probabilities)

    return np.concatenate((char_data, color_data), axis=0)

transformer = tf.keras.models.load_model('saved_ansi_model',  custom_objects={'CustomMetric': masked_accuracy}, compile=False)

outputs = []
encoder_input = tf.convert_to_tensor(np.zeros((1, 20, 80, 264)), dtype=tf.float64)
for _ in range(number_of_rows):
    raw_decoder_input = np.zeros((1, 81, 264))
    random_seed = np.random.rand(264)
    raw_decoder_input[:, 0, :] = random_seed
    decoder_input = tf.convert_to_tensor(raw_decoder_input, dtype=tf.float64)
    for col in range(80):
        input_data = (encoder_input, decoder_input)
        new_output = transformer.predict(input_data)  # 1, 81, 264
        observed_character = observe_character(np.squeeze(new_output[:, col, :]))
        decoder_input = np.array(decoder_input) 
        decoder_input[:, col+1, :] = observed_character
        decoder_input = tf.convert_to_tensor(decoder_input)
    encoder_input = tf.convert_to_tensor(np.concatenate((encoder_input.numpy()[:, :-1, :, :], decoder_input[np.newaxis, :, 1:, :]), axis=1))
    outputs.append(decoder_input[:, 1:, :])

total_output = np.vstack(outputs)
file_content = AnsFromVecConverter.convert(total_output)
with open("test1.ANS", "w", encoding="latin1") as file:
    file.write(file_content)


