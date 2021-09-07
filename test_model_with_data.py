import tensorflow as tf
from models.model import Transformer
from loader import DatasetLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences

if __name__ == '__main__':
    raw_vi, raw_en, inp_builder, tar_builder = DatasetLoader("dataset/seq2seq/train.en.txt",
                                                             "dataset/seq2seq/train.vi.txt",
                                                             min_length=10,
                                                             max_length=14).build_dataset()

    padded_sequences_vi = pad_sequences(raw_vi, maxlen=14, padding="post", truncating="post")
    padded_sequences_en = pad_sequences(raw_en, maxlen=14, padding="post", truncating="post")

    input_lang = tf.data.Dataset.from_tensor_slices(padded_sequences_vi).batch(2)
    target_lang = tf.data.Dataset.from_tensor_slices(padded_sequences_en).batch(2)

    d_model = 256
    header_size = 8
    n_layers = 2
    inp_vocab_size = len(inp_builder.word_index) + 1
    tar_vocab_size = len(tar_builder.word_index) + 1
    diff_deep = 512
    activation = 'relu'

    transfomer = Transformer(
        inp_vocab_size, tar_vocab_size, header_size, diff_deep, d_model, n_layers, activation
    )

    for x, y in zip(input_lang.take(1), target_lang.take(1)):
        outs, _ = transfomer(x, y, is_train=True)
        print(outs.shape)
