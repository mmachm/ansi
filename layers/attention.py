import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        """
        The number of heads is just a way to make it more powerful.
        The same input is fed to each head and at the end the results are combined.

        Here key_dim is the size of the attention head for query and key. In other words this is the
        internal dimension on which they agree. In other words the dimensionality of the query and
        key vectors for each attention head.

        :param num_heads:
        :param key_dim:
        """
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        """
        Query and key are the same dimension so they can be dotted.

        IMPORTANT - this dot between x and context happens as a matrix multiplication (and a transpose): we get
        shape(k, l) * shape(l, m) = shape(k, m). Here k and m are the lengths of the sequences that go in
        and l is the dimension of the internal embedding, this must be kept equal! But m and k can differ.
        Recall that

        Attention(Q, K, V) = softmax((QK^T)/sqrt d_k) * V
        shape(k, l) = shape(k,m) * shape(m,l) = (shape(k, l) * shape(m, l)^T) * shape(m, l)

        So the output of the attention layer has the same shape as the query. Indeed, this is somehow to be expected
        from a "differentiable dictionary".

        :param x:
        :param context:
        :return:
        """
        try:
            attn_output, attn_scores = self.mha(
                query=x,
                key=context,
                value=context,
                return_attention_scores=True
            )
        except:
            raise
        # attn_output will be a linear combination of words from context, right?

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class GlobalSelfAttention(BaseAttention):
    """
    Really this is the same as Cross attention with passing x for context.

    """

    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
