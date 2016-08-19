from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        # Download
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

filename = maybe_download('text8.zip', 31344016)

def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return tf.compat.as_str(f.read(name))
    f.close()

text = read_data(filename)
# print('Data size %d' % len(text))


valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
# print(train_size, train_text[:batch_size])
# print(valid_size, valid_text[:batch_size])

# Map characters to vocabulary IDs
vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0]) # Unicode

def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1 # Values between 1 - 26 (letters)
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0

def id2char(dictid):
    if dictid > 0:
        return chr(dictid + first_letter - 1) # ord() inverse
    else:
        return ' '

# print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
# print(id2char(1), id2char(26), id2char(0))

# Generate training batch
batch_size = 128
num_unrollings = 10 # LSTM unrolling

class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """
        Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = []
        batches.append(self._next_batch())
        return batches


def characters(probabilities):
    """
    Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation.
    """
    return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
    """
    Convert a sequence of batches back into their (most likely) string
    representation.
    """
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

# print(batches2string(train_batches.next()))
# print(batches2string(train_batches.next()))
# print(batches2string(valid_batches.next()))
# print(batches2string(valid_batches.next()))


def logprob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
#     print("LOGPROB")
#     print(predictions)
#     print("==")
#     print(labels)
#     print(labels.shape)
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
    """
    Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1

def sample(prediction):
    """
    Turn a (column) prediction into 1-hot encoded samples.
    """
    p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p

def random_distribution():
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
    return b/np.sum(b, 1)[:,None]


# LSTM model
num_nodes = 128

from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

# Using TensorFlow LSTM cells.
class Model():
    # Similar performance to lstm_udacity
    # TODO Improve performance
    # https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/model.py
    # https://github.com/thtrieu/qclass_dl/blob/f823d9a1739e57bceeec1f1a2cf1a1012313f76b/lstm.py
    # https://github.com/dnlcrl/TensorFlow-Playground/blob/232b97a2316c3249cd41fa076d287dc328ba9c24/1.tutorials/6.Recurrent%20Neural%20Networks/ptb_word_lm.py
    # https://github.com/3sevieria/en_fr/blob/97ec2dc75ba574b92340320dee87cd9bdc855195/main_rnn.py

    # To infer the paremeters batch_size and seq_length must be 1
    def __init__(self, batch_size, vocab_size, seq_length, lstm_size,
                 num_layers=1, cell_fn=rnn_cell.BasicLSTMCell, keep_prob=1,
                 is_training=True):
        infer = False # Add parameter
        grad_clip = 5
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.lstm_size = lstm_size # lstm_size = num_nodes
        self.lstm_cell = cell_fn(lstm_size)
        self.num_layers = num_layers # Unrolling
        self.vocab_size = vocab_size

        self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.targets = tf.placeholder(tf.int32, [batch_size, seq_length])

        if is_training and keep_prob < 1:
            self.lstm_cell = lstm_cell.DropoutWrapper(lstm_cell,
                                                     output_keep_prob=keep_prob)

        self.cell = rnn_cell.MultiRNNCell([self.lstm_cell] * num_layers)

        self.initial_state = self.cell.zero_state(batch_size, tf.float32)

        softmax_w = tf.Variable(
            tf.truncated_normal([self.lstm_size,
                                 vocabulary_size], -0.1, 0.1))
        softmax_b = tf.Variable(tf.zeros([vocabulary_size]))

        embedding = tf.get_variable("embedding", [self.vocab_size, self.lstm_size])
        inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        if is_training and keep_prob < 1:
            inputs = [tf.nn.dropout(input_, keep_prob)
                      for input_ in inputs]

        outputs, last_state = rnn.rnn(self.cell,
                                      inputs,
                                      initial_state=self.initial_state)

        output = tf.reshape(tf.concat(1, outputs), [-1, self.lstm_size])
        self.logits = tf.nn.xw_plus_b(output,
                                      softmax_w,
                                      softmax_b)
        self.probs = tf.nn.softmax(self.logits)

        loss = seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([self.batch_size * self.seq_length])],
            self.vocab_size)
        self.cost = tf.reduce_sum(loss) / self.batch_size
        self.final_state = last_state

        correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.targets,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Optimizer.
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            10.0, global_step, 500, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(self.cost))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        self.optimizer = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step)



num_steps = 5001
summary_frequency = 500

train_batches = BatchGenerator(train_text, batch_size, 0)
valid_batches = BatchGenerator(valid_text, 1, 1)





import os
import collections
from six.moves import cPickle
import numpy as np

class TextLoader():
    def __init__(self, data, batch_size=128, seq_length=1):
        self.data = data
        self.data_size = len(data)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.top = 0
        self.delta_batches = 8
        self.num_batches = int(self.data_size / (self.batch_size *
                                                   self.seq_length))
        self.create_batches()

    def create_batches(self):
        # 1024/128
        self.reset_batch_pointer()

        new_top = (self.delta_batches * self.batch_size * self.seq_length) + self.top

        self.tensor = self.data[self.top:new_top]
        self.top = new_top

        xdata = np.array(list(self.tensor))
        ydata = np.copy(xdata)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(128, -1), 8, 1)
        #print(self.x_batches)
        self.y_batches = np.split(ydata.reshape(128, -1), 8, 1)
        #print(self.y_batches)

    def next_batch(self):
        # TODO if not over ask for more batches
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        if (self.pointer == self.delta_batches):
            self.create_batches()
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-1.0, 1.0)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        m = Model(batch_size, vocabulary_size, seq_length=1,
                  lstm_size=num_nodes, num_layers=num_unrollings,
                  cell_fn=rnn_cell.BasicLSTMCell, keep_prob=1,
                  is_training=True)

    tf.initialize_all_variables().run()
    tl = TextLoader(data=text)

    mean_loss = 0
    for step in range(num_steps):
        x, y = tl.next_batch()

        for i in range(batch_size):
            x[i][0] = char2id(x[i][0])
            y[i][0] = char2id(y[i][0])

        x = x.astype(np.int32)
        y = y.astype(np.int32)

        _, l, predictions, acc = session.run([m.optimizer,
                                              m.cost,
                                              m.probs,
                                              m.accuracy],
                                         feed_dict={m.input_data: x,
                                                    m.targets: y})
        mean_loss += l

        if step % summary_frequency == 0:
            if step > 0:
                mean_loss = mean_loss / summary_frequency
            # The mean loss is an estimate of the loss over the last few batches.
            print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
            mean_loss = 0
            target_nparray = np.array(y, dtype=int)
            target_nparray = y.reshape(1, batch_size)[0]
            one_hot_target = np.zeros((batch_size, vocabulary_size))
            for ind in range(batch_size):
                one_hot_target[ind][target_nparray[ind]] = 1

            print('Minibatch perplexity: %.2f' % float(
                        np.exp(logprob(predictions, one_hot_target))))
#             if step % (summary_frequency ) == 0:
#                 # Generate some samples.
#                 print('=' * 80)
#                 s = ""
#                 for letterid in np.argmax(predictions, 1):
#                     s += id2char(letterid)
#                 print(s)
#                 print('=' * 80)

            # # Measure validation set perplexity.
            # reset_sample_state.run()
            # valid_logprob = 0
            # for _ in range(valid_size):
            #     b = valid_batches.next()
            #     predictions = sample_prediction.eval({sample_input: b[0]})
            #     valid_logprob = valid_logprob + logprob(predictions, b[1])
            # print('Validation set perplexity: %.2f' % float(np.exp(
            #                 valid_logprob / valid_size)))
