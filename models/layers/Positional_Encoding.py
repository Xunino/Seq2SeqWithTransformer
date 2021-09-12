import tensorflow as tf
import numpy as np


class PositionalEncodingLayer(tf.keras.layers.Layer):

    def __init__(self, d_model):
        super(PositionalEncodingLayer, self).__init__()
        self.d_model = d_model

    def get_angles(self, positions, indexes):
        d_model_tensor = tf.cast(self.d_model, dtype=tf.float32)
        angle_rates = np.power(10000, (2 * (indexes // 2)) / d_model_tensor)
        return positions / angle_rates

    def __call__(self, position, *args, **kwargs):
        """
        :param position: seq_len
        :return position_encoding: [seq_len, d_model]
        """
        positions = np.arange(position)[..., np.newaxis]  # [1, seq_len]
        indexes = np.arange(self.d_model)[np.newaxis, ...]  # [d_model, 1]
        angles = self.get_angles(positions, indexes)  # [seq_len, d_model]
        angles[:, 0::2] = np.sin(angles[:, 0::2])  # apply sin to even indices in the tensor; 2i
        angles[:, 1::2] = np.cos(angles[:, 1::2])  # apply cos to odd indices in the tensor; 2i + 1
        return tf.cast(angles[np.newaxis, ...], dtype=tf.float32)  # [seq_len, d_model]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n, d = 2048, 512
    pos_encoding = PositionalEncodingLayer(d)(n)
    print(pos_encoding.shape)
    pos_encoding = pos_encoding[0]

    # Juggle the dimensions for the plot
    pos_encoding = tf.reshape(pos_encoding, (n, d // 2, 2))
    pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
    pos_encoding = tf.reshape(pos_encoding, (d, n))

    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.ylabel('Depth')
    plt.xlabel('Position')
    plt.colorbar()
    plt.show()