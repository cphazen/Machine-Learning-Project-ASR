import random
import copy
import numpy as np

class RNN(object):
    def __init__(self, input_count, input_dimension, output_count, output_dimension,
                 memory_dimension, activate, d_activate):
        # DIMENSIONS
        self.id = (input_count, input_dimension)
        self.od = (output_count, output_dimension)
        self.md = memory_dimension

        # WEIGHTS
        self.wm = np.random.randn(memory_dimension, memory_dimension)
        self.w1 = np.random.randn(input_dimension, memory_dimension)
        self.w2 = np.random.randn(memory_dimension, output_dimension)
        self.w3 = np.random.randn(input_count, output_count)

        # ACTIVATION
        self.activate = np.vectorize(activate)
        self.d_activate = np.vectorize(d_activate)

    def forward_propagate(self, data):
        self.ai = []
        self.am = []
        self.ao1 = []

        self.am.append(np.zeros(self.md))    # initialize memory for first node

        for i in xrange(self.id[0]):
            self.ai.append(np.array(data[i]))               # convert data into np array

            l1_in = np.dot(self.ai[i], self.w1)             # get weights from new data
            l1_mem = np.dot(self.am[-1], self.wm)           # get weights from old data
            l1 = self.activate(l1_in + l1_mem)              # sum + activate

            l2_in = np.dot(l1, self.w2)                     # reshape output pt1
            l2 = self.activate(l2_in)                       # activate

            self.ao1.append(copy.deepcopy(l2))              # store the output of l2
            self.am.append(copy.deepcopy(l1))               # commit l1 to memory

        self.ao2 = np.dot(np.array(self.ao1).T, self.w3).T  # reshape output into final output
        self.ao2 = self.activate(self.ao2)                  # activate

        return self.ao2

    def back_propagate(self, expected_output, eta):
        #delta = self.d_activate(actual[0]) * (data - actual[0])
        #for i in range(1, self.s.shape[0]):
        #    delta += self.d_activate(actual[i]) * (data - actual[i]).T
        #
        #self.w += eta * (delta * input[2:])
        # Error
        # LAYER 3 Error:
        error_w3 = expected_output - self.ao2
        delta_w3 = self.d_activate(self.ao2) * error_w3
        slope_w3 = np.dot(delta_w3, np.array(self.ao1).T)

        # LAYER 2 Error:
        #for i in xrange(self.id[0]):
        #error_w2 = np.dot(delta_w3, self.w3)
        #delta_w2 = self.d_activate(self.am[-1])


        # Update
        self.w3 += eta * slope_w3.T


    def train(self, given_input, expected_output, epoch, learning_rate):
        # Performs feed forward and back propagate for all input
        #
        # Parameters:
        #   given_input         input for feed forward
        #   expected_output     expected output of feed forward
        #   epoch               times to repeat training
        #   learning_rate       learning rate
        #
        # Returns:
        #   Saved output to feed into next layer
        cached_output = []
        for i in xrange(epoch):
            for j in xrange(len(given_input)):
                output = self.forward_propagate(given_input[j])
                cached_output.append(output)
                self.back_propagate(expected_output[j], learning_rate)

        return cached_output


    def test(self, data):
        # Performs feed forward
        # TODO: Change this function to test a list of data
        #
        # Parameters:
        #   data            input for feed forward
        #
        # Returns:
        #   Result of feed forward
        return self.feed_forward(data)
