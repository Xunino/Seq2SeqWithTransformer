import time

import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from models.layers.Encode_Layer import EncodeLayer
from models.layers.Positional_Encoding import PositionalEncodingLayer


class Encode(tf.keras.layers.Layer):
    def __init__(self,
                 inp_vocab_size,
                 header_size,
                 diff_deep,
                 d_model,
                 n_layers,
                 maximum_position_encoding,
                 drop_rate=0.1):
        super(Encode, self).__init__()

        # Initialize params
        self.inp_vocab_size = inp_vocab_size
        self.d_model = d_model
        self.maximum_position_encoding = maximum_position_encoding

        # Initialize layers
        self.encode_layers = [EncodeLayer(header_size=header_size,
                                          diff_deep=diff_deep,
                                          d_model=d_model) for _ in range(n_layers)]
        self.embedding = Embedding(inp_vocab_size, d_model)
        self.PE = PositionalEncodingLayer(d_model)
        self.dropout = Dropout(drop_rate)

    def __call__(self, inputs, mask, is_train):
        """
        :param inputs: [batch_size, seq_len]
        :param args: is_train=True/False
        :return:
        """
        seq_len = tf.shape(inputs)[1]

        # Do Embedding
        x = self.embedding(inputs)  # [batch_size, seq_len, embedding_size]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Do Positional Encoding
        PE = self.PE(self.maximum_position_encoding, self.d_model)
        x += PE[:, :seq_len, :]

        # Do Dropout
        x = self.dropout(x)

        # Do Encode layers
        for layer in self.encode_layers:
            x = layer(x, mask, is_train)

        return x


if __name__ == '__main__':
    inp_vocab_size = 8500
    header_size = 8
    diff_deep = 2048
    d_model = 512
    n_layers = 2
    maximum_position_encoding = 10000
    sample_encoder = Encode(inp_vocab_size,
                            header_size,
                            diff_deep,
                            d_model,
                            n_layers,
                            maximum_position_encoding)
    # temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
    temp_input = tf.constant(
        [[1, 464, 45, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    sample_encoder_output = sample_encoder(temp_input, is_train=False, mask=None)

    print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
