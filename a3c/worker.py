import collections
import itertools

import numpy as np
import tensorflow as tf

from estimators import ValueEstimator, PolicyEstimator
from state_processor import StateProcessor
from utils import make_copy_params_op, make_train_op

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class Worker(object):
    """
    An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.
    
    Args:
        name: A unique name for this worker
        env: The Gym environment used by this worker
        policy_net: Instance of the globally shared policy net
        value_net: Instance of the globally shared value net
        global_counter: Iterator that holds the global step
        discount_factor: Reward discount factor
        summary_writer: A tf.train.SummaryWriter for Tensorboard summaries
        max_global_steps: If set, stop coordinator when global_counter > max_global_steps
    """

    def __init__(self, name, env, policy_net, value_net, global_counter, discount_factor=0.99,
                 max_global_steps=None):
        self.name = name
        self.discount_factor = discount_factor
        self.max_global_steps = max_global_steps
        self.global_step = tf.contrib.framework.get_global_step()
        self.global_policy_net = policy_net
        self.global_value_net = value_net
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.sp = StateProcessor()
        self.env = env

        # Create local policy/value nets that are not updated asynchronously
        with tf.variable_scope(name):
            self.policy_net = PolicyEstimator(policy_net.num_outputs)
            self.value_net = ValueEstimator(reuse=True)

        # Op to copy params from global policy/valuenets
        self.copy_params_op = make_copy_params_op(
            tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
            tf.contrib.slim.get_variables(scope=self.name, collection=tf.GraphKeys.TRAINABLE_VARIABLES))

        self.vnet_train_op = make_train_op(self.value_net, self.global_value_net)
        self.pnet_train_op = make_train_op(self.policy_net, self.global_policy_net)

        self.state = None

    def run(self, sess, coord, t_max):
        with sess.as_default(), sess.graph.as_default():
            # Initial state
            self.state = self.sp.process(self.env.reset())
            self.state = np.stack([self.state] * 4, axis=2)
            try:
                while not coord.should_stop():
                    # Copy Parameters from the global networks
                    sess.run(self.copy_params_op)

                    # Collect some experience
                    transitions, global_t = self.run_n_steps(t_max, sess)

                    if self.max_global_steps is not None and global_t >= self.max_global_steps:
                        tf.logging.info("Reached global step {}. Stopping.".format(global_t))
                        coord.request_stop()
                        return

                    # Update the global networks
                    self.update(transitions, sess)

            except tf.errors.CancelledError:
                return

    def _policy_net_predict(self, state, sess):
        predicts = sess.run(self.policy_net.predictions, {
            self.policy_net.states: [state]
        })
        return predicts["probs"][0]

    def _value_net_predict(self, state, sess):
        predicts = sess.run(self.value_net.predictions, {
            self.value_net.states: [state]
        })
        return predicts["logits"][0]

    def run_n_steps(self, n, sess):
        transitions = []
        for _ in range(n):
            # Take a step
            action_probabilities = self._policy_net_predict(self.state, sess)
            action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)
            next_state, reward, done, info = self.env.step(action)
            next_state = self.sp.process(next_state)
            next_state = np.append(self.state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
            if info["ale.lives"] == 2:
                done = True

            # Store transition
            transitions.append(Transition(
                state=self.state, action=action, reward=reward, next_state=next_state, done=done))

            # Increase local and global counters
            local_t = next(self.local_counter)
            global_t = next(self.global_counter)

            if local_t % 100 == 0:
                tf.logging.info("{}: local Step {}, global step {}".format(self.name, local_t, global_t))

            if done:
                self.state = self.sp.process(self.env.reset())
                self.state = np.stack([self.state] * 4, axis=2)
                break
            else:
                self.state = next_state
        return transitions, global_t

    def update(self, transitions, sess):
        """
        Updates global policy and value networks based on collected experience
    
        Args:
          transitions: A list of experience transitions
          sess: A Tensorflow session
        """

        # If we episode was not done we bootstrap the value from the last state
        reward = 0.0
        if not transitions[-1].done:
            reward = self._value_net_predict(transitions[-1].next_state, sess)

        # Accumulate mini-batch examples
        states = []
        policy_targets = []
        value_targets = []
        actions = []

        for transition in transitions[::-1]:
            reward = transition.reward + self.discount_factor * reward
            policy_target = (reward - self._value_net_predict(transition.state, sess))
            # Accumulate updates
            states.append(transition.state)
            actions.append(transition.action)
            policy_targets.append(policy_target)
            value_targets.append(reward)

        feed_dict = {
            self.policy_net.states: np.array(states),
            self.policy_net.targets: policy_targets,
            self.policy_net.actions: actions,
            self.value_net.states: np.array(states),
            self.value_net.targets: value_targets,
        }

        # Train the global estimators using local gradients
        global_step, pnet_loss, vnet_loss, _, _ = sess.run([
            self.global_step,
            self.policy_net.loss,
            self.value_net.loss,
            self.pnet_train_op,
            self.vnet_train_op
        ], feed_dict)

        return pnet_loss, vnet_loss
