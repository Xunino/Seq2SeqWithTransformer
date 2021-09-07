import tensorflow as tf
from tensorflow.keras.layers import Dense


class MultiHeadLayer(tf.keras.layers.Layer):
    """
        Multi Head Layer
    """

    def __init__(self,
                 d_model,
                 header_size):
        super(MultiHeadLayer, self).__init__()
        # Initialize params
        self.header_size = header_size
        self.d_model = d_model

        assert d_model % self.header_size == 0

        self.depth = self.d_model // self.header_size

        # Initialize weights
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        :param x: [batch_size, seq_len, hidden_units]
        :return: multi_head: [batch_size, header_size, seq_len, query_size]
        """
        # [batch_size, seq_len, hidden_units] -> [batch_size, seq_len, header_size, query_size]
        x = tf.reshape(x, (batch_size, -1, self.header_size, self.depth))

        # [batch_size, header_size, seq_len, query_size] -> [batch_size, header_size, seq_len, query_size]
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, v, k, q, mask):
        """
        :param inputs: [batch_size, seq_len, hidden_units]
        :return:
        """
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Do attention
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weight = self.scaled_dot_product_attention(q, k, v, mask)

        # outs: [batch_size, header_size, seq_len, query_size] -> [batch_size, seq_len, header_size, query_size]
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # outs: [batch_size, seq_len, header_size, query_size] -> [batch_size, seq_len, hidden_size]
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weight

    def scaled_dot_product_attention(self, q, k, v, mask):
        """
        :param q: [..., ..., seq_len, query_size]
        :param k: [..., ..., seq_len, query_size]
        :param v: [..., ..., seq_len, query_size]
        :return:
        """

        # [..., ..., seq_len, query_size_q] (.) [..., ..., query_size_k, seq_len] = [..., ..., seq_len, seq_len]
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Using padding mask
        if mask is not None:
            scaled_attention_logits += (mask * -1e30)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # [..., ..., seq_len_q, seq_len_k]
        alignment_weight = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # [..., ..., seq_len, seq_len] (.) [..., ..., seq_len, query_size_v] = [..., ..., seq_len, query_size_v]
        outs = tf.matmul(alignment_weight, v)
        return outs, alignment_weight


def print_out(q, k, v):
    temp_out, temp_attn = temp_mha.scaled_dot_product_attention(
        q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)


if __name__ == '__main__':
    import numpy as np

    temp_mha = MultiHeadLayer(d_model=512, header_size=8)

    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, y, y, mask=None)
    print(out.shape, attn.shape)

    np.set_printoptions(suppress=True)

    temp_k = tf.constant([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10],
                          [0, 0, 10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 6]], dtype=tf.float32)  # (4, 2)

    # This `query` aligns with the second `key`,
    # so the second `value` is returned.
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)
