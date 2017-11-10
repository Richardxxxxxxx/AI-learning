import numpy as np
import tensorflow as tf

# from https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html

num_steps = 5 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 200
num_classes = 2 # output
state_size = 4 # state vector that pass from one to another
learning_rate = 0.1

def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y) #[size]

# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps] #[batch_size, batch_partition_length] -> [batch_size, num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)   #[batch_size, num_steps]

def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps) #[batch_size, num_steps]



def train_network(num_epochs, num_steps, state_size=4, verbose=True):
    """
    Placeholders
    """

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
    init_state = tf.zeros([batch_size, state_size])

    """
    RNN Inputs
    """

    # Turn our x placeholder into a list of one-hot tensors:
    # rnn_inputs is a list of num_steps tensors with shape [batch_size, num_classes]
    """
    one_hot(
        indices,
        depth,
        on_value=None,
        off_value=None,
        axis=None,
        dtype=None,
        name=None
    )
    """
    x_one_hot = tf.one_hot(x, num_classes) #[batch_size,num_steps,num_classes]
    rnn_inputs = tf.unstack(x_one_hot, axis=1) #num_steps X [batch_size,num_classes]

    cell = tf.contrib.rnn.BasicRNNCell(state_size)
    """
    The simplest form of RNN network generated is:
    state = cell.zero_state(...)
    outputs = []
    for input_ in inputs:
        output, state = cell(input_, state)
        outputs.append(output)
        return (outputs, state)
    However, a few other options are available:
    static_rnn(
        cell,
        inputs,
        initial_state=None,
        dtype=None,
        sequence_length=None,
        scope=None
    )
    Args:

    cell: An instance of RNNCell.
    inputs: A length T list of inputs, each a Tensor of shape [batch_size, input_size], or a nested tuple of such elements.
    initial_state: (optional) An initial state for the RNN. If cell.state_size is an integer, this must be a Tensor of appropriate type and shape [batch_size, cell.state_size]. If cell.state_size is a tuple, this should be a tuple of tensors having shapes [batch_size, s] for s in cell.state_size.
    dtype: (optional) The data type for the initial state and expected output. Required if initial_state is not provided or RNN state has a heterogeneous dtype.
    sequence_length: Specifies the length of each sequence in inputs. An int32 or int64 vector (tensor) size [batch_size], values in [0, T).
    scope: VariableScope for the created subgraph; defaults to "rnn".
    Returns:

    A pair (outputs, state) where:

    outputs is a length T list of outputs (one for each input), or a nested tuple of such elements.
    state is the final state
    """

    rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)#rnn_outputs:[num_steps X [batch_size,state_size]]

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]  #[num_steps X [batch_size,num_classes]]
    predictions = [tf.nn.softmax(logit) for logit in logits] #[num_steps X [batch_size,num_classes]]

    # Turn our y placeholder into a list of labels
    """
    Args:
    unstack(
        value,
        num=None,
        axis=0,
        name='unstack'
    )
    value: A rank R > 0 Tensor to be unstacked.
    num: An int. The length of the dimension axis. Automatically inferred if None (the default).
    axis: An int. The axis to unstack along. Defaults to the first dimension. Supports negative indexes.
    name: A name for the operation (optional).
    """
    y_as_list = tf.unstack(y, num=num_steps, axis=1) #[num_steps X [batch_size]]

    # losses and train_step
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
              logit, label in zip(logits, y_as_list)]#[num_steps,batch_size]
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print("\nEPOCH", idx)
            for step, (X, Y) in enumerate(epoch):
                print(X.get_shape())
                tr_losses, training_loss_, training_state, _ = \
                    sess.run([losses,
                              total_loss,
                              final_state,
                              train_step],
                                  feed_dict={x:X, y:Y, init_state:training_state})
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step", step,
                              "for last 250 steps:", training_loss/100)
                    training_losses.append(training_loss/100)
                    training_loss = 0

    return training_losses

if __name__ == '__main__' :
    training_losses = train_network(1,num_steps)
    """
    x = tf.constant(2, shape=[4,3])
    x_one_hot = tf.one_hot(x, 2)
    rnn_inputs = tf.unstack(x_one_hot, axis=1)
    sess = tf.InteractiveSession()
    print(sess.run(x))
    print(sess.run(x_one_hot))
    print(sess.run(rnn_inputs))
    """