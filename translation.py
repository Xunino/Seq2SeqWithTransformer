import os
import io
import json
import time

import numpy as np
import tensorflow as tf
from metrics import CustomSchedule
from argparse import ArgumentParser
from models.model import Transformer
from loader import remove_punctuation
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


class Translate:
    def __init__(self, max_seq_len, header_size, diff_deep, d_model, n_layers):

        home = os.getcwd()
        self.max_seq_len = max_seq_len
        self.save_dict = home + "/saved_checkpoint/{}_vocab.json"

        self.inp_builder = self.load_tokenizer(name_vocab="input")
        self.tar_builder = self.load_tokenizer(name_vocab="target")
        self.keys = list(self.tar_builder.word_docs.keys())

        self.start = self.tar_builder.word_index["<sos>"]
        self.end = self.tar_builder.word_index["<eos>"]

        # Initialize Seq2Seq model
        input_vocab_size = len(self.inp_builder.word_index) + 1
        target_vocab_size = len(self.tar_builder.word_index) + 1

        # Initialize transformer model
        self.transformer = Transformer(inp_vocab_size=input_vocab_size,
                                       tar_vocab_size=target_vocab_size,
                                       n_layers=n_layers,
                                       header_size=header_size,
                                       diff_deep=diff_deep,
                                       d_model=d_model,
                                       pe_input=max_seq_len,
                                       pe_target=max_seq_len)

        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # Initialize check point
        self.saved_checkpoint = os.getcwd() + "/saved_checkpoint/"
        ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                   optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.saved_checkpoint, max_to_keep=5)

        print("[INFO] Load models", end="")
        self.ckpt_manager.restore_or_initialize()
        print(" --> DONE!")

    def load_tokenizer(self, name_vocab):
        path_save = self.save_dict.format(name_vocab)
        with io.open(path_save) as f:
            data = json.load(f)
            token = tokenizer_from_json(data)
        return token

    def __call__(self, text: str):
        text = remove_punctuation(text)
        vector = [self.inp_builder.word_index[w] for w in text.split() if w in self.keys]
        vector = tf.expand_dims(vector, axis=0)
        tensor = pad_sequences(vector,
                               maxlen=self.max_seq_len,
                               padding="post",
                               truncating="post")
        encode_input = tf.convert_to_tensor(tensor, dtype=tf.int64)

        decode_input = tf.convert_to_tensor([self.start], dtype=tf.int64)
        decode_input = tf.expand_dims(decode_input, 0)

        text = ""
        for _ in range(self.max_seq_len):
            predicted = self.transformer(encode_input, decode_input, False)
            predicted = predicted[:, -1:, :]
            predicted_id = tf.argmax(predicted, axis=-1)
            decode_input = tf.concat([decode_input, predicted_id], axis=-1)

            if predicted_id[0] not in [self.start, self.end]:
                text += self.tar_builder.index_word[predicted_id[0][0].numpy()] + " "

            if predicted_id == [self.end]:
                break

        return text


if __name__ == '__main__':
    # TrainTransformer("dataset/seq2seq/train.en.txt", "dataset/seq2seq/train.vi.txt").fit()
    parser = ArgumentParser()

    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--n_layers", default=3, type=int)
    parser.add_argument("--header-size", default=8, type=int)
    parser.add_argument("--d-model", default=512, type=int)
    parser.add_argument("--diff-deep", default=1024, type=int)
    parser.add_argument("--max-sentence", default=50, type=int)

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
    translate = Translate(d_model=args.d_model,
                          header_size=args.header_size,
                          diff_deep=args.diff_deep,
                          max_seq_len=args.max_sentence,
                          n_layers=args.n_layers)
    while True:
        print("\n===========================================")
        text = str(input("[INFO] Enter text : "))
        print("[INFO] Translate  :", translate(text.lower()))
    # python translation.py
