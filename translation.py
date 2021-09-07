import os

import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences

from train import TrainTransformer
from argparse import ArgumentParser


class Translate(TrainTransformer):
    def __init__(self, inp_lang_path, tar_lang_path, min_seq_len, max_seq_len, epochs, header_size,
                 diff_deep, d_model, warmup, batch_size, n_layers, retrain, bleu,
                 debug):
        super(Translate, self).__init__(inp_lang_path, tar_lang_path, min_seq_len, max_seq_len, epochs, header_size,
                                        diff_deep, d_model, warmup, batch_size, n_layers, retrain, bleu, debug)

        if os.listdir(self.saved_checkpoint) != []:
            self.ckpt_manager.restore_or_initialize()
            print("[INFO] Restore models")

    def __call__(self, text):
        text = self.loader.remove_punctuation(text)
        vector = self.inp_builder.texts_to_sequences(text.split())
        vector = tf.reshape(vector, shape=(1, -1))
        tensor = pad_sequences(vector,
                               maxlen=self.max_seq_len,
                               padding="post",
                               truncating="post")
        encode_input = tf.convert_to_tensor(tensor, dtype=tf.int64)

        start = [self.tar_builder.word_index["<sos>"]]
        end = [self.tar_builder.word_index["<eos>"]]
        decode_input = tf.convert_to_tensor(start, dtype=tf.int64)
        decode_input = tf.expand_dims(decode_input, 0)

        for _ in range(self.max_seq_len):
            predicted = self.transformer(encode_input, decode_input, is_train=False)
            predicted = predicted[:, -1:, :]
            predicted_id = tf.argmax(predicted, axis=-1)
            decode_input = tf.concat([decode_input, predicted_id], axis=-1)

            if predicted_id == end:
                break
        return " ".join(self.tar_builder.sequences_to_texts(np.array(decode_input)))


if __name__ == '__main__':
    # TrainTransformer("dataset/seq2seq/train.en.txt", "dataset/seq2seq/train.vi.txt").fit()
    parser = ArgumentParser()

    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--inp-lang", required=True, type=str)
    parser.add_argument("--tar-lang", required=True, type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--n_layers", default=1, type=int)
    parser.add_argument("--d-model", default=256, type=int)
    parser.add_argument("--header-size", default=8, type=int)
    parser.add_argument("--diff-deep", default=512, type=int)
    parser.add_argument("--min-sentence", default=4, type=int)
    parser.add_argument("--max-sentence", default=10, type=int)
    parser.add_argument("--warmup-steps", default=4000, type=int)
    parser.add_argument("--retrain", default=False, type=bool)
    parser.add_argument("--bleu", default=False, type=bool)
    parser.add_argument("--debug", default=False, type=bool)

    args = parser.parse_args()

    # FIXME
    # Project Description

    print('---------------------Welcome to Hợp tác xã Kiên trì-------------------')
    print('Github: https://github.com/Xunino')
    print('Email: ndlinh.ai@gmail.com')
    print('------------------------------------------------------------------------')
    print(f'Training Sequences To Sequences model with hyper-params:')
    print('------------------------------------')
    for k, v in vars(args).items():
        print(f"|  +) {k} = {v}")
    print('====================================')

    # FIXME
    # Do Training
    translate = Translate(inp_lang_path=args.inp_lang,
                          tar_lang_path=args.tar_lang,
                          batch_size=args.batch_size,
                          epochs=args.epochs,
                          d_model=args.d_model,
                          header_size=args.header_size,
                          diff_deep=args.diff_deep,
                          min_seq_len=args.min_sentence,
                          max_seq_len=args.max_sentence,
                          warmup=args.warmup_steps,
                          n_layers=args.n_layers,
                          retrain=args.retrain,
                          bleu=args.bleu,
                          debug=args.debug)

    text = "Enter the sentence to translate:"
    print(translate(text))
    # python translation.py --inp-lang="dataset/seq2seq/train.en.txt" --tar-lang="dataset/seq2seq/train.vi.txt"
