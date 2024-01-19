import tensorflow as tf
from keras.layers import Flatten, Dense

class PatchEmbedding(tf.keras.layers.Layer):
    """
    Probably obsolete - the convolutions are going to do the job just as well in this case.

    """
    def __init__(self, patch_size, num_patches, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.flatten = Flatten()
        self.dense_proj = Dense(embed_dim)

    def call(self, inputs):
       x = self.flatten(inputs)
       return self.dense_proj(x)


# # Example usage
# image_size = 224
# patch_size = 16
# in_channels = 3  # Number of input channels (e.g., 3 for RGB images)
# embed_dim = 768  # Dimensionality of the patch embeddings
#
# # Create a random image tensor
# random_image = tf.random.normal((1, image_size, image_size, in_channels))
#
# # Create the patch embedding layer
#
# # Apply patch embedding to the random image
# patch_embeddings = patch_embedding_layer(random_image)
# print("Shape of patch embeddings:", patch_embeddings.shape)