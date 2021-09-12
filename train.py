import os
import time

import numpy as np
import tensorflow as tf
from loader import DatasetLoader
from argparse import ArgumentParser
from models.model import Transformer
from keras_preprocessing.sequence import pad_sequences
from metrics import BleuScore, CustomSchedule, MaskedSoftmaxCELoss, accuracy_function
from sklearn.model_selection import train_test_split


class TrainTransformer:
    def __init__(self,
                 inp_lang_path,
                 tar_lang_path,
                 min_seq_len,
                 max_seq_len,
                 epochs,
                 header_size,
                 diff_deep,
                 d_model,
                 warmup,
                 batch_size,
                 n_layers,
                 test_size,
                 retrain,
                 bleu,
                 debug):
        # Initialize params
        self.inp_lang_path = inp_lang_path
        self.tar_lang_path = tar_lang_path
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

        self.batch_size = batch_size
        self.epochs = epochs
        self.header_size = header_size
        self.diff_deep = diff_deep
        self.d_model = d_model
        self.n_layers = n_layers
        self.retrain = retrain
        self.bleu = bleu
        self.debug = debug
        self.test_size = test_size

        # Initialize dataset
        self.loader = DatasetLoader(self.inp_lang_path,
                                    self.tar_lang_path,
                                    self.min_seq_len,
                                    self.max_seq_len)
        self.inp_vector, self.tar_vector, self.inp_builder, self.tar_builder = self.loader.build_dataset()
        inp_vocab_size = len(self.inp_builder.word_index) + 1
        tar_vocab_size = len(self.tar_builder.word_index) + 1

        self.token_end = tf.constant(self.tar_builder.word_index["<eos>"])[tf.newaxis]

        # Initialize transformer model
        self.transformer = Transformer(inp_vocab_size=inp_vocab_size,
                                       tar_vocab_size=tar_vocab_size,
                                       n_layers=self.n_layers,
                                       header_size=self.header_size,
                                       diff_deep=self.diff_deep,
                                       d_model=self.d_model,
                                       pe_input=self.max_seq_len,
                                       pe_target=self.max_seq_len)

        # Initialize learning rate scheduler
        learning_scheduler = CustomSchedule(d_model, warmup)

        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # Initialize
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

        # Initialize check point
        self.saved_checkpoint = os.getcwd() + "/saved_checkpoint/"
        if not os.path.exists(self.saved_checkpoint):
            os.mkdir(self.saved_checkpoint)
        ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                   optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.saved_checkpoint, max_to_keep=5)

        if retrain:
            print("[INFO] Start Retrain...")
            print("[INFO] Loading model...")
            time.sleep(1)
            self.ckpt_manager.restore_or_initialize()
            print("[INFO] DONE!")

        # Initialize Bleu score
        self.bleu_score = BleuScore()

    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions = self.transformer(inp, tar_inp, is_train=True)
            loss = MaskedSoftmaxCELoss(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(accuracy_function(tar_real, predictions))

    def evaluation(self, inp, tar):
        score = 0
        all_items = len(inp)
        for i, (encode_input, target) in enumerate(zip(inp, tar)):
            # Target text
            target_sentence = " ".join(self.tar_builder.sequences_to_texts([target.numpy()]))

            # Encode input
            input_sentence = " ".join(self.inp_builder.sequences_to_texts([encode_input.numpy()]))
            encode_input = tf.expand_dims(encode_input, axis=0)

            # Decode input
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
            pred_sentence = " ".join(self.tar_builder.sequences_to_texts(np.array(decode_input)))
            score += self.bleu_score(pred_sentence, target_sentence)

            if i < 5:
                print("Input   : ", input_sentence)
                print("Predict : ", pred_sentence)
                print("Target  : ", target_sentence)
                print("-----------------------------------------------------------")

        return score / all_items

    def fit(self):

        # Padding in sequences
        input_data = pad_sequences(self.inp_vector,
                                   maxlen=self.max_seq_len,
                                   padding="post",
                                   truncating="post")
        target_data = pad_sequences(self.tar_vector,
                                    maxlen=self.max_seq_len,
                                    padding="post",
                                    truncating="post")

        # Add to tensor
        train_x, test_x, train_y, test_y = train_test_split(input_data, target_data, test_size=self.test_size)

        train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        val_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))

        train_ds = train_ds.batch(self.batch_size)
        train_ds = train_ds.shuffle(42)

        train_len = len(train_x)
        self.N_batch = train_len // self.batch_size

        for epoch in range(self.epochs):
            print("===========================================================")

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            for batch, (inp, tar) in enumerate(train_ds):
                self.train_step(inp, tar)

                if batch % 50 == 0:
                    print('Epoch {} -- Batch: {} -- Loss: {:.4f} -- Accuracy: {:.4f}'.format(epoch + 1, batch,
                                                                                             self.train_loss.result(),
                                                                                             self.train_accuracy.result()))
            print("-----------------------------------------------------------")
            if self.bleu:
                for batch, (inp, tar) in enumerate(val_ds):
                    bleu_score = self.evaluation(inp, tar)
                    print('Epoch {} -- Loss: {:.4f} -- Accuracy: {:.4f} -- Bleu_score: {:.4f}'.format(epoch + 1,
                                                                                                      self.train_loss.result(),
                                                                                                      self.train_accuracy.result(),
                                                                                                      bleu_score))
            else:
                print('Epoch {} -- Loss: {:.4f} -- Accuracy: {:.4f} '.format(epoch + 1,
                                                                             self.train_loss.result(),
                                                                             self.train_accuracy.result()))

            if (epoch + 1) % 10 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print(f'[INFO] Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

            print("===========================================================")


if __name__ == '__main__':
    # TrainTransformer("dataset/seq2seq/train.en.txt", "dataset/seq2seq/train.vi.txt").fit()
    parser = ArgumentParser()

    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--target-path", required=True, type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--n_layers", default=1, type=int)
    parser.add_argument("--d-model", default=256, type=int)
    parser.add_argument("--header-size", default=8, type=int)
    parser.add_argument("--diff-deep", default=512, type=int)
    parser.add_argument("--min-sentence", default=5, type=int)
    parser.add_argument("--max-sentence", default=10, type=int)
    parser.add_argument("--warmup-steps", default=4000, type=int)
    parser.add_argument("--test-size", default=0.1, type=float)
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
    TrainTransformer(inp_lang_path=args.input_path,
                     tar_lang_path=args.target_path,
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
                     test_size=args.test_size,
                     bleu=args.bleu,
                     debug=args.debug).fit()
    # python train.py --inp-lang="dataset/seq2seq/train.en.txt" --tar-lang="dataset/seq2seq/train.vi.txt" --bleu=True
