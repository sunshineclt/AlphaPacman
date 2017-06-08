import tensorflow as tf


def build_shared_network(x):
    """
    Builds a 3-layer network conv -> conv -> fc as described
    in the A3C paper. This network is shared by both the policy and value net.
    
    Args:
        x: Inputs    
    Returns:
        Final layer activations.
    """

    # Three convolutional layers
    conv1 = tf.contrib.layers.conv2d(
        x, 16, 8, 4, activation_fn=tf.nn.relu, scope="conv1", padding='SAME')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.contrib.layers.conv2d(
        pool1, 32, 4, 2, activation_fn=tf.nn.relu, scope="conv2", padding='SAME')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer
    fc1 = tf.contrib.layers.fully_connected(
        inputs=tf.contrib.layers.flatten(pool2),
        num_outputs=256,
        scope="fc1")

    return fc1


class PolicyEstimator:
    """
    Policy Function approximator. Given a observation, returns probabilities
    over all possible actions.
    
    Args:
    num_outputs: Size of the action space.
    reuse: If true, an existing shared network will be re-used.
    trainable: If true we add train ops to the network.
      Actor threads that don't update their local models and don't need
      train ops would set this to false.
    """

    def __init__(self, num_outputs, reuse=False, trainable=True):
        self.num_outputs = num_outputs

        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        # Normalize
        x = tf.to_float(self.states) / 255.0
        batch_size = tf.shape(self.states)[0]

        # Graph shared with Value Net
        with tf.variable_scope("shared", reuse=reuse):
            fc1 = build_shared_network(x)

        with tf.variable_scope("policy_net"):
            self.logits = tf.contrib.layers.fully_connected(fc1, num_outputs, activation_fn=None)
            self.probs = tf.nn.softmax(self.logits) + 1e-8

            self.predictions = {
                "logits": self.logits,
                "probs": self.probs
            }

            # We add entropy to the loss to encourage exploration
            self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), 1, name="entropy")
            self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

            # Get the predictions for the chosen actions only
            gather_indices = tf.range(batch_size) * tf.shape(self.probs)[1] + self.actions
            self.picked_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)

            self.losses = - (tf.log(self.picked_action_probs) * self.targets + 0.01 * self.entropy)
            self.loss = tf.reduce_sum(self.losses, name="loss")

            if trainable:
                # self.optimizer = tf.train.AdamOptimizer(1e-4)
                self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
                                                               global_step=tf.contrib.framework.get_global_step())


class ValueEstimator:
    """
    Value Function approximator. Returns a value estimator for a batch of observations.
    
    Args:
    reuse: If true, an existing shared network will be re-used.
    trainable: If true we add train ops to the network.
      Actor threads that don't update their local models and don't need
      train ops would set this to false.
    """

    def __init__(self, reuse=False, trainable=True):
        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        x = tf.to_float(self.states) / 255.0

        # Graph shared with Value Net
        with tf.variable_scope("shared", reuse=reuse):
            fc1 = build_shared_network(x)

        with tf.variable_scope("value_net"):
            self.logits = tf.contrib.layers.fully_connected(
                inputs=fc1,
                num_outputs=1,
                activation_fn=None)
            self.logits = tf.squeeze(self.logits, squeeze_dims=[1], name="logits")

            self.losses = tf.squared_difference(self.logits, self.targets)
            self.loss = tf.reduce_sum(self.losses, name="loss")

            self.predictions = {
                "logits": self.logits
            }

            if trainable:
                # self.optimizer = tf.train.AdamOptimizer(1e-4)
                self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
                                                               global_step=tf.contrib.framework.get_global_step())
