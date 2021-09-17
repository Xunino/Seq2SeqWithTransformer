from tensorflow.keras.layers import Dense
import tensorflow as tf


class FFN(tf.keras.layers.Layer):
    """
        Position Wise Feed Forward Network function:
            FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self,
                 diff_deep=1024,
                 d_model=512,
                 activation="gelu"):
        super(FFN, self).__init__()
        self.dense_1 = Dense(diff_deep, activation=activation)
        self.dense_2 = Dense(d_model)

    def __call__(self, inputs, *args, **kwargs):
        return self.dense_2(self.dense_1(inputs))
