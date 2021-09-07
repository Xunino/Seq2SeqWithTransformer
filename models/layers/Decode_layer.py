import tensorflow as tf

from models.layers.Encode_Layer import EncodeLayer
from models.layers.Multi_Head_Layer import MultiHeadLayer
from models.layers.Position_Wise_Feed_Forward_Network import FFN
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization


class DecodeLayer(tf.keras.layers.Layer):
    def __init__(self,
                 d_model,
                 diff_deep,
                 header_size,
                 drop_rate=0.1,
                 eps=0.1):
        super(DecodeLayer, self).__init__()

        # Initialize Attention layers
        self.mha_with_mask = MultiHeadLayer(d_model=d_model,
                                            header_size=header_size)
        self.mha = MultiHeadLayer(d_model=d_model,
                                  header_size=header_size)
        self.ffn = FFN(diff_deep=diff_deep,
                       d_model=d_model)

        # Layer normalization
        self.layer_norm_1 = LayerNormalization(epsilon=eps)
        self.layer_norm_2 = LayerNormalization(epsilon=eps)
        self.layer_norm_3 = LayerNormalization(epsilon=eps)

        # Dropout
        self.drop_out_1 = Dropout(drop_rate)
        self.drop_out_2 = Dropout(drop_rate)
        self.drop_out_3 = Dropout(drop_rate)

    def __call__(self, decode_inps, encode_outs, look_ahead_mask, padding_mask, is_train):
        q = k = encode_outs
        x = decode_inps

        # Do Multi Head Attention with Mask
        attn1, self_attention_weights = self.mha_with_mask(x, x, x, look_ahead_mask)
        # Do Norm
        norm_outs_1 = self.layer_norm_1(x + self.drop_out_1(attn1, is_train))

        # Do Multi Head Attention
        context_vector_2, global_attention_weights = self.mha(q, k, norm_outs_1, padding_mask)

        # Do Norm
        norm_outs_2 = self.layer_norm_2(norm_outs_1 + self.drop_out_2(context_vector_2, is_train))

        # Do Feed Forward Network
        ffn_outs = self.ffn(norm_outs_2)

        # Do Norm
        norm_outs_3 = self.layer_norm_3(norm_outs_2 + self.drop_out_3(ffn_outs, training=is_train))

        return norm_outs_3, self_attention_weights, global_attention_weights


if __name__ == '__main__':
    sample_encoder_layer = EncodeLayer(header_size=8, diff_deep=2048, d_model=512)

    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((64, 43, 512)), None, False)

    print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
    sample_decoder_layer = DecodeLayer(header_size=8, diff_deep=2048, d_model=512)

    sample_decoder_layer_output, _, _ = sample_decoder_layer(
        tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
        None, None, False)
    print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)
