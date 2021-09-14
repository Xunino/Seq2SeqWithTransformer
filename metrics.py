import numpy as np
import collections
import tensorflow as tf


class BleuScore:
    """
        We can evaluate a predicted sequence by comparing it with the label sequence.
        BLEU (Bilingual Evaluation Understudy) "https://aclanthology.org/P02-1040.pdf",
        though originally proposed for evaluating machine translation results,
        has been extensively used in measuring the quality of output sequences for different applications.
        In principle, for any n-grams in the predicted sequence, BLEU evaluates whether this n-grams appears
        in the label sequence.
    """

    def __init__(self):
        super().__init__()

    def remove_oov(self, sentence):
        return [i for i in sentence.split(" ") if i not in ["<sos>", "<eos>"]]

    def __call__(self, pred, target, n_grams=3):
        pred = self.remove_oov(pred)
        target = self.remove_oov(target)
        pred_length = len(pred)
        target_length = len(target)

        if pred_length < n_grams:
            return 0
        else:
            score = np.exp(np.minimum(0, 1 - target_length / pred_length))
            for k in range(1, n_grams + 1):
                label_subs = collections.defaultdict(int)
                for i in range(target_length - k + 1):
                    label_subs[" ".join(target[i:i + k])] += 1

                num_matches = 0
                for i in range(pred_length - k + 1):
                    if label_subs[" ".join(pred[i:i + k])] > 0:
                        label_subs[" ".join(pred[i:i + k])] -= 1
                        num_matches += 1
                score *= np.power(num_matches / (pred_length - k + 1), np.power(0.5, k))
            return score


def MaskedSoftmaxCELoss(label, pred):
    """
    :param label: shape (batch_size, max_length, vocab_size)
    :param pred: shape (batch_size, max_length)

    :return: weighted_loss: shape (batch_size, max_length)
    """
    weights_mask = 1 - np.equal(label, 0)
    unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label, pred)
    weighted_loss = tf.reduce_mean(unweighted_loss * weights_mask)
    return weighted_loss


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_step=4000):
        super(CustomSchedule, self).__init__()
        self.warmup_step = warmup_step
        self.d_model = tf.cast(d_model, dtype=tf.float32)

    def __call__(self, step):
        result = self.d_model ** -0.5 * tf.minimum(step ** -0.5, step * self.warmup_step ** -1.5)
        return result


def accuracy_function(real, pred):
    # Check equal same -> True
    real = tf.cast(real, dtype=tf.int64)
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    # Ìf both mask and accuracies is True -> True
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    # Return True value -> 1
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


class ROUGE:
    """
        ROUGE: A Package for Automatic Evaluation of Summaries
        https://aclanthology.org/W04-1013.pdf
    """

    def __init__(self):
        pass

    def _create_ngrams(self, token, n=3):
        token = token.split()
        len_token = len(token)
        ngrams = collections.defaultdict(int)
        for i in range(len_token - n + 1):
            ngrams[" ".join(token[i:i + n])] += 1
        return ngrams

    def _recall_safe(self, x, y):
        return round(max(x / y, 0), 4)

    def _precision_safe(self, x, y):
        return round(max(x / y, 0), 4)

    def _lcs_len(self, x, y):
        """
            This function returns length of longest common sequence of x and y.
        """

        if len(x) == 0 or len(y) == 0:
            return 0

        xx = x[:-1]  # xx = sequence x without its last element
        yy = y[:-1]

        if x[-1] == y[-1]:  # if last elements of x and y are equal
            return self._lcs_len(xx, yy) + 1
        else:
            return max(self._lcs_len(xx, y), self._lcs_len(x, yy))

    def calculate_ngrams(self, predict_sentence, target_sentence, n=3):
        # Do check matches
        target_ngrams = self._create_ngrams(target_sentence, n)
        pred_ngrams = self._create_ngrams(predict_sentence, n)

        matches = 0
        for ngram in target_ngrams.keys():
            matches += min(target_ngrams[ngram], pred_ngrams[ngram])

        # Do recall
        recall = self._recall_safe(matches, len(target_ngrams))

        # Do precision
        precision = self._precision_safe(matches, len(pred_ngrams))

        # Do f_score
        if (recall + precision) > 0:
            f_score = round(2 * recall * precision / (recall + precision), 4)
        else:
            f_score = 0.0
        return recall, precision, f_score

    def calculate_lcs(self, predict_sentence, target_sentence, beta=2):

        predict_sentence = predict_sentence.split()
        target_sentence = target_sentence.split()

        # Do longest common sentence length
        lcs_length = self._lcs_len(predict_sentence, target_sentence)

        # Do recall
        recall = self._recall_safe(lcs_length, len(target_sentence))

        # Do precision
        precision = self._precision_safe(lcs_length, len(predict_sentence))

        # Do f_score
        if (recall + precision) > 0:
            f_score = round((1 + beta ** 2) * recall * precision / (recall + (beta ** 2) * precision), 4)
        else:
            f_score = 0.0
        return recall, precision, f_score


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    temp_learning_rate_schedule = CustomSchedule(512)

    plt.plot(temp_learning_rate_schedule(tf.range(10000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()
