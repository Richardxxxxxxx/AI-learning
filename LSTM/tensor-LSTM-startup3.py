from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import urllib3
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import constant_op




def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()


def build_vocab(filename):
  data = _read_words(filename)
  #print(data)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  id_to_world = dict(zip(word_to_id.values(),word_to_id.keys()))

  return word_to_id,id_to_world


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data]


def gen_raw_data(word_to_id, train_data_path=None,valid_data_path=None,test_data_path=None):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_data = _file_to_word_ids(train_data_path, word_to_id) if train_data_path else None
  valid_data = _file_to_word_ids(valid_data_path, word_to_id) if valid_data_path else None
  test_data = _file_to_word_ids(test_data_path, word_to_id) if test_data_path else None
  return train_data, valid_data, test_data


def gen_batchs(data, batch_size, num_steps):
  """Iterate on the raw PTB data.
  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.
  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  data = np.array(data, dtype=np.int32)

  data_len = len(data)
  batch_len = data_len // batch_size
  diveded_data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    diveded_data[i] = data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = diveded_data[:, i*num_steps:(i+1)*num_steps]
    y = diveded_data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)

def gen_epochs(data, num_epochs, num_steps, batch_size):
    for i in range(num_epochs):
        yield gen_batchs(data, batch_size, num_steps)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def train_network(g, data, num_epochs, num_steps = 200, batch_size = 32, verbose = True, save=False, checkpoint=False):
    print("start training")
    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if checkpoint :
            t = time.time()
            g['saver'].restore(sess, checkpoint)
            print("It took", time.time() - t, "seconds to restore")
        
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(data, num_epochs, num_steps, batch_size)):
            t = time.time()
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1

                feed_dict={g['x']: X, g['y']: Y}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                      g['final_state'],
                                                      g['train_step']],
                                                             feed_dict)
                training_loss += training_loss_
            if verbose:
                print("It took", time.time() - t, "seconds to train for epoch ",idx)
                print("Average training loss for Epoch", idx, ":", training_loss/steps)
            training_losses.append(training_loss/steps)
        if isinstance(save, str):
            g['saver'].save(sess, save)

    return training_losses


def ln(tensor, scope = None, epsilon = 1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert(len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return LN_initial * scale + shift

class LayerNormalizedLSTMCell(rnn_cell_impl.BasicLSTMCell):
    def __init__(self, num_units, forget_bias=1.0,
                 state_is_tuple=True, activation=None, reuse=None, name=None):
        super(LayerNormalizedLSTMCell, self).__init__(num_units, forget_bias, state_is_tuple, activation, reuse, name)


    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).

        Args:
          inputs: `2-D` tensor with shape `[batch_size, input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size, self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size, 2 * self.state_size]`.

        Returns:
          A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(
            value=gate_inputs, num_or_size_splits=4, axis=one)

        # add layer normalization to each gate
        i = ln(i, scope='i/')
        j = ln(j, scope='j/')
        f = ln(f, scope='f/')
        o = ln(o, scope='o/')

        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = math_ops.add
        multiply = math_ops.multiply
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                    multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))

        if self._state_is_tuple:
            new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state



def build_graph(
    num_classes,
    cell_type = None,
    num_weights_for_custom_cell = 5,
    state_size = 100,
    batch_size = 32,
    num_steps = 200,
    num_layers = 3,
    build_with_dropout=False,
    learning_rate = 1e-4):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    dropout = tf.constant(1.0)
    """
    [vocabulary_size, embedding_size]  
    embedding_size is the length you want to represent a word
    embedding_size is not necessary to be state_size 
    embedding_matrix is a variable that it learns how to represent word as training proceeding
    """
    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])
    """
    [batch_size, num_steps] -> [batch_size, num_steps, embedding_size], in this example embedding_size = state_size
    """
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    if cell_type == 'GRU':
        cells = [tf.nn.rnn_cell.GRUCell(state_size) for i in range(num_layers)]
    elif cell_type == 'LSTM':
        cells = [tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True) for i in range(num_layers)]
    elif cell_type == 'LN_LSTM':
        cells = [LayerNormalizedLSTMCell(state_size) for i in range(num_layers)]
    else:
        cells = [tf.nn.rnn_cell.BasicRNNCell(state_size) for i in range(num_layers)]

    if build_with_dropout:
        cells = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout) for cell in cells]

    if cell_type == 'LSTM' or cell_type == 'LN_LSTM':
        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    else:
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    if build_with_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)

    init_state = cell.zero_state(batch_size, tf.float32)

    """
    init_state  : [batch_size, cell.state_size]
    rnn_inputs  : [batch_size, num_steps, anylength]
    rnn_outputs : [batch_size, num_steps, cell.state_size]
    final_state : [batch_size, cell.state_size]
    """
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    #reshape rnn_outputs and y
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    predictions = tf.nn.softmax(logits)

    """
    labels: Tensor of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels and result) and dtype int32 or int64. 
    Each entry in labels must be an index in [0, num_classes). Other values will raise an exception when this op is run on CPU, 
    and return NaN for corresponding loss and gradient rows on GPU.
    logits: Unscaled log probabilities of shape [d_0, d_1, ..., d_{r-1}, num_classes] and dtype float32 or float64
    
    WARNING: This op expects unscaled logits, since it performs a softmax on logits internally for efficiency. 
    Do not call this op with the output of softmax, as it will produce incorrect results.
    
    A common use case is to have logits of shape [batch_size, num_classes] and labels of shape [batch_size]. But higher dimensions are supported.
    """
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels = y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step,
        preds = predictions,
        saver = tf.train.Saver()
    )

def generate_characters(g, vocab_to_idx, idx_to_vocab, checkpoint, num_chars, prompt='A', pick_top_chars=None):
    """ Accepts a current character, initial state"""

    vocab_size = len(vocab_to_idx)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess, checkpoint)

        state = None
        current_char = vocab_to_idx[prompt]
        chars = [current_char]

        for i in range(num_chars):
            if state is not None:
                feed_dict={g['x']: [[current_char]], g['init_state']: state}
            else:
                feed_dict={g['x']: [[current_char]]}

            preds, state = sess.run([g['preds'],g['final_state']], feed_dict)

            if pick_top_chars is not None:
                p = np.squeeze(preds)
                p[np.argsort(p)[:-pick_top_chars]] = 0
                p = p / np.sum(p)
                current_char = np.random.choice(vocab_size, 1, p=p)[0]
            else:
                current_char = np.random.choice(vocab_size, 1, p=np.squeeze(preds))[0]

            chars.append(current_char)
            #chars.append("")

    chars = map(lambda x: idx_to_vocab[x] + " ", chars)
    print("".join(chars).replace("<eos>","\n"))
    return("".join(chars))

def download_file(url, path):
    http = urllib3.PoolManager()
    r = http.request('GET', url, preload_content=False)

    with open(path, 'wb') as out:
        while True:
            data = r.read(4096)
            if not data:
                break
            out.write(data)

    r.release_conn()

if __name__ == '__main__' :

    file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
    file_name = 'tinyshakespeare.txt'
    if not os.path.exists(file_name):
        download_file(file_url, file_name)

    word_to_id, id_to_world = build_vocab(file_name)
    train_data, valid_data, test_data = gen_raw_data(word_to_id,train_data_path=file_name)
    num_classes = len(word_to_id)
    """
    t = time.time()
    g = build_graph(num_classes, cell_type='GRU', num_steps=80)
    print("It took", time.time() - t, "seconds to build the graph.")
    t = time.time()
    losses = train_network(g, train_data, 5, num_steps=80, save="saves/GRU_30_epochs", checkpoint="saves/GRU_25_epochs")
    print("It took", time.time() - t, "seconds to train for 20 epochs.")
    print("The average loss on the final epoch was:", losses[-1])
    """
    """
    It took 1051.6652357578278 seconds to train for 20 epochs.
    The average loss on the final epoch was: 1.75318197903
    """
    """
    g = build_graph(num_classes, cell_type='LSTM', num_steps=80)
    t = time.time()
    losses = train_network(g, train_data, 20, num_steps=80, save="saves/LSTM_20_epochs")
    print("It took", time.time() - t, "seconds to train for 20 epochs.")
    print("The average loss on the final epoch was:", losses[-1])
    """
    """
    It took 614.4890048503876 seconds to train for 20 epochs.
    The average loss on the final epoch was: 2.02813237837
    """
    """
    g = build_graph(num_classes, cell_type='LN_LSTM', num_steps=80)
    t = time.time()
    losses = train_network(g, train_data, 20, num_steps=80, save="saves/LN_LSTM_20_epochs")
    print("It took", time.time() - t, "seconds to train for 20 epochs.")
    print("The average loss on the final epoch was:", losses[-1])
    """
    """
    It took 3867.550405740738 seconds to train for 20 epochs.
    The average loss on the final epoch was: 1.71850851623
    """

    g = build_graph(num_classes, cell_type='GRU', num_steps=1, batch_size=1)
    print(generate_characters(g, word_to_id, id_to_world, "saves/GRU_30_epochs", 750, prompt='you'))



