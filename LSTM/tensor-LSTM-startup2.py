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

    rnn_inputs = tf.one_hot(x, num_classes) #[batch_size,num_steps,num_classes]
    cell = tf.contrib.rnn.BasicRNNCell(state_size)
    """
    dynamic_rnn(
        cell,
        inputs,
        sequence_length=None,
        initial_state=None,
        dtype=None,
        parallel_iterations=None,
        swap_memory=False,
        time_major=False,
        scope=None
    )
    Args:

    cell: An instance of RNNCell.
    inputs: The RNN inputs.

    If time_major == False (default), this must be a Tensor of shape: [batch_size, max_time, ...], or a nested tuple of such elements.

    If time_major == True, this must be a Tensor of shape: [max_time, batch_size, ...], or a nested tuple of such elements.
    
    
    
    outputs: The RNN output `Tensor`.

    If time_major == False (default), this will be a `Tensor` shaped:
        `[batch_size, max_time, cell.output_size]`.

    If time_major == True, this will be a `Tensor` shaped:
        `[max_time, batch_size, cell.output_size]`.
    """
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    """
    Predictions, loss, training step
    """

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.reshape(
        tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W) + b,
        [batch_size, num_steps, num_classes])
    predictions = tf.nn.softmax(logits)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
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