import tensorflow as tf
from models.layers.Encode import Encode
from models.layers.Decode import Decode
from tensorflow.keras.layers import Dense

from models.layers.Masking import create_masks


class Transformer(tf.keras.models.Model):
    def __init__(self,
                 inp_vocab_size,
                 tar_vocab_size,
                 n_layers,
                 header_size,
                 diff_deep,
                 d_model,
                 pe_input,
                 pe_target,
                 drop_rate=0.1):
        super(Transformer, self).__init__()

        # Initialize Encode - Decode
        self.encode = Encode(inp_vocab_size,
                             header_size=header_size,
                             diff_deep=diff_deep,
                             d_model=d_model,
                             n_layers=n_layers,
                             maximum_position_encoding=pe_input,
                             drop_rate=drop_rate)
        self.decode = Decode(tar_vocab_size,
                             header_size=header_size,
                             diff_deep=diff_deep,
                             d_model=d_model,
                             n_layers=n_layers,
                             maximum_position_encoding=pe_target,
                             drop_rate=drop_rate)
        self.last_layer = Dense(tar_vocab_size)

    def __call__(self, inps_encode, inps_decode, is_train):
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inps_encode, inps_decode)

        # (batch_size, inp_seq_len, d_model)
        encode_outs = self.encode(inps_encode, enc_padding_mask, is_train)

        # (batch_size, tar_seq_len, d_model)
        decode, _ = self.decode(inps_decode, encode_outs, look_ahead_mask, dec_padding_mask, is_train)

        return self.last_layer(decode)  # (batch_size, tar_seq_len, target_vocab_size)


if __name__ == '__main__':
    inp_vocab_size = 8500
    tar_vocab_size = 8000
    n_layers = 2
    header_size = 8
    diff_deep = 2048
    d_model = 512
    pe_input = 10000
    pe_target = 6000
    sample_transformer = Transformer(inp_vocab_size,
                                     tar_vocab_size,
                                     n_layers,
                                     header_size,
                                     diff_deep,
                                     d_model,
                                     pe_input,
                                     pe_target)

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

    fn_out = sample_transformer(temp_input, temp_target, False)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
