import tensorflow as tf

from models.layers.Encode import Encode
from models.layers.Positional_Encoding import PositionalEncodingLayer
from models.layers.Decode_layer import DecodeLayer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout


class Decode(tf.keras.layers.Layer):
    def __init__(self,
                 tar_vocab_size,
                 header_size,
                 diff_deep,
                 d_model,
                 n_layers,
                 maximum_position_encoding,
                 drop_rate=0.1):
        super(Decode, self).__init__()

        # Initialize params
        self.d_model = d_model
        self.diff_deep = diff_deep
        self.maximum_position_encoding = maximum_position_encoding

        # Initialize layer transformer
        self.decode_layers = [DecodeLayer(d_model=d_model,
                                          diff_deep=diff_deep,
                                          header_size=header_size,
                                          drop_rate=drop_rate) for _ in range(n_layers)]
        self.PE = PositionalEncodingLayer(d_model)
        self.embedding = Embedding(tar_vocab_size, d_model)
        self.drop_out = Dropout(drop_rate)

    def __call__(self, inps_decode, encode_outs, look_ahead_mask, padding_mask, is_train):
        seq_len = tf.shape(inps_decode)[1]

        # Do embedding
        x = self.embedding(inps_decode)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Do Positional Encoding
        PE = self.PE(self.maximum_position_encoding, self.d_model)
        x += PE[:, :seq_len, :]

        # Do Dropout
        x = self.drop_out(x, is_train)

        # Do Decode Layers
        attention_weights = {}
        for i, layer in enumerate(self.decode_layers):
            x, block1, block2 = layer(x, encode_outs, look_ahead_mask, padding_mask, is_train)
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2
        return x, attention_weights


if __name__ == '__main__':
    inp_vocab_size = 8500
    tar_vocab_size = 5000
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
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
    sample_encoder_output = sample_encoder(temp_input, is_train=False, mask=None)
    print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

    # ----------------------- Decode ---------------------
    sample_decoder = Decode(tar_vocab_size,
                            header_size,
                            diff_deep,
                            d_model,
                            n_layers,
                            maximum_position_encoding, )
    temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)
    print(sample_encoder_output.shape)
    output, attn = sample_decoder(temp_input,
                                  sample_encoder_output,
                                  False,
                                  None,
                                  None)

    print(output.shape, attn['decoder_layer2_block2'].shape)
