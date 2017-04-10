# Built-in imports
import random
import copy

# External libraries:
import numpy as np

# Our own stuff:
from progress import Progress

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
        # Unfolds network for the number of items provided, then reshapes result
        # to match expected output
        #
        # Parameters:
        #   data        data to transform into approximation of classification
        #
        # Returns:
        #   approximation of classfication
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
        # Uses chain rule and gradient descent to account for error and
        # reset weights through time
        #
        # Parameters:
        #   expected_output     expected output of feed forward
        #   eta                 learning rate
        #
        # Returns:
        #   None

        # Error
        # LAYER 3 Error:
        error_w3 = expected_output - self.ao2
        delta_w3 = self.d_activate(self.ao2) * error_w3
        slope_w3 = np.dot(delta_w3, np.array(self.ao1).T)

        # LAYER 2/1/M Error:
        error_w2 = np.dot(delta_w3.T, self.w3.T)
        delta_w2 = self.d_activate(self.ao1) * error_w2.T

        slope_w2 = np.zeros_like(self.w2)
        slope_wm = np.zeros_like(self.wm)
        slope_w1 = np.zeros_like(self.w1)

        this_dw1 = np.zeros(self.md)
        for i in xrange(self.id[0]):
            this_dw2 = delta_w2[-i-1]
            this_l1 = self.am[-i-1]
            prev_l1 = self.am[-i-2]

            error_wm = np.dot(this_dw1, self.wm.T)
            error_w1 = np.dot(this_dw2, self.w2.T)
            delta_w1 = self.d_activate(this_l1) * error_w1
            delta_w1h = error_wm + error_w1

            this_dw1 = delta_w1h

            slope_w2 += np.dot(this_l1[:,None], this_dw2[None,:])
            slope_wm += np.dot(prev_l1[:,None], delta_w1h[None,:])
            slope_w1 += np.dot(self.ai[i][:,None], delta_w1h[None,:])

        # Update
        self.w3 += eta * slope_w3.T
        self.w2 += eta * slope_w2
        self.wm += eta * slope_wm
        self.w1 += eta * slope_w1

    def save(self, name):
        # Save current state
        #
        # Parameters:
        #   name            name of file to save arrays to
        # Returns:
        #   None

        name = name + '.npz'
        np.savez(name, wm=self.wm, w1=self.w1, w2=self.w2, w3=self.w3)
        return

    def load(self, name):
        # Load previous state. Prints a warning if loaded state dimensions don't match
        #
        # Parameters:
        #   name            name of file to save arrays to
        # Returns:
        #   None

        name = name + '.npz'
        a = np.load(name)
        res = False
        if((self.w3.shape == a['w3'].shape) and (self.w2.shape == a['w2'].shape) and \
           (self.w1.shape == a['w1'].shape) and (self.wm.shape == a['wm'].shape)):
            self.w3 = a['w3']
            self.w2 = a['w2']
            self.w1 = a['w1']
            self.wm = a['wm']
            res = True
        else:
            print('[ERROR] File dimensions do not match this network')
        return res

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

        p = Progress(30, epoch*len(given_input))
        p.start_progress()

        cached_output = []
        for i in xrange(epoch):
            for j in xrange(len(given_input)):
                # Train
                output = self.forward_propagate(given_input[j])
                cached_output.append(output)
                self.back_propagate(expected_output[j], learning_rate)

                # Print Progress
                p.update_progress()
        p.complete_progress()
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
