from argparse import ArgumentParser

import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm
from models.model import Transformer
from metrics import BleuScore, CustomSchedule, MaskedSoftmaxCELoss
from loader import DatasetLoader
from models.layers.Masking import create_masks


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
                 split_test,
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
        self.split_test = split_test
        self.bleu = bleu
        self.debug = debug

        # Initialize learning rate scheduler
        learning_scheduler = CustomSchedule(d_model, warmup)

        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # Initialize dataset
        loader = DatasetLoader(self.inp_lang_path,
                               self.tar_lang_path,
                               self.min_seq_len,
                               self.max_seq_len)
        self.inp_vector, self.tar_vector, self.inp_builder, self.tar_builder = loader.build_dataset()
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
        # Initialize
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions = self.transformer(inp, tar_inp, is_train=True)
            loss = MaskedSoftmaxCELoss(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(self.accuracy_function(tar_real, predictions))

    def accuracy_function(self, real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    def fit(self):

        # Padding in sequences
        train_x = pad_sequences(self.inp_vector,
                                maxlen=self.max_seq_len,
                                padding="post",
                                truncating="post")
        train_y = pad_sequences(self.tar_vector,
                                maxlen=self.max_seq_len,
                                padding="post",
                                truncating="post")

        train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(42).batch(self.batch_size)

        train_len = len(train_x)
        split_test = int(train_len * self.split_test)
        self.N_batch = train_len // self.batch_size

        for epoch in range(self.epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            for batch, (inp, tar) in enumerate(train_ds):
                self.train_step(inp, tar)

                if batch % 50 == 0:
                    print('Epoch {} -- Batch {} -- Loss: {.4f} -- Accuracy: {.4f}'.format(epoch + 1, batch,
                                                                                          self.train_loss.result(),
                                                                                          self.train_accuracy.result()))

            print('Epoch {} -- Loss: {.4f} -- Accuracy: {.4f}'.format(epoch + 1,
                                                                      self.train_loss.result(),
                                                                      self.train_accuracy.result()))


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
    parser.add_argument("--split-test", default=0.001, type=float)
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
    TrainTransformer(inp_lang_path=args.inp_lang,
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
                     split_test=args.split_test,
                     retrain=args.retrain,
                     bleu=args.bleu,
                     debug=args.debug).fit()
    # python train.py --inp-lang="dataset/seq2seq/train.en.txt" --tar-lang="dataset/seq2seq/train.vi.txt"
