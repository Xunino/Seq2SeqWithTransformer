import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from models.layers.Multi_Head_Layer import MultiHeadLayer
from models.layers.Position_Wise_Feed_Forward_Network import FFN


class EncodeLayer(tf.keras.layers.Layer):
    def __init__(self,
                 header_size,
                 diff_deep,
                 d_model,
                 drop_rate=0.1,
                 eps=1e-6):
        super(EncodeLayer, self).__init__()

        # Initialize Attention layers
        self.mha = MultiHeadLayer(d_model=d_model, header_size=header_size)
        self.ffn = FFN(diff_deep=diff_deep, d_model=d_model)

        # Layer Normalization
        self.layer_norm_1 = LayerNormalization(epsilon=eps)
        self.layer_norm_2 = LayerNormalization(epsilon=eps)

        # Dropout
        self.drop_1 = Dropout(drop_rate)
        self.drop_2 = Dropout(drop_rate)

    def __call__(self, inputs, mask, is_train):
        q = k = v = inputs

        # Do Multi Head Attention
        context_vector, attention_weights = self.mha(q, k, v, mask)

        # Do norm
        drop_1 = self.drop_1(inputs, is_train)
        norm_1_outs = self.layer_norm_1(drop_1 + context_vector)

        # Do feed forward network
        ffn_outs = self.ffn(norm_1_outs)

        # Do norm
        drop_2 = self.drop_2(ffn_outs, is_train)
        norm_2_outs = self.layer_norm_2(drop_2 + norm_1_outs)
        return norm_2_outs


if __name__ == '__main__':
    sample_encoder_layer = EncodeLayer(header_size=8, diff_deep=2048, d_model=512)

    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((64, 43, 512)), False, None)

    print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
