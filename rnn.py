import random
import numpy as np

class RNN(object):
    def __init__(self, ni, no, activate, d_activate):
        ni += 1
        self.w = np.random.randn(ni, no)

        self.activate = np.vectorize(activate)
        self.d_activate = np.vectorize(d_activate)

    def forward_propagate(self, data, t):
        self.ai = np.matrix(data).T
        # self.a1 = self.activate(np.dot(self.w.T, self.ai))
        self.s = np.zeros((t+1, self.w.shape[1]))
        for i in xrange(1,t+1):
            self.s[i] = self.activate(np.dot(self.w.T, self.ai)).T
            self.ai[:2] = self.s[i][0].T
            self.ai[2] = 1.0
        return self.s, self.s[t]

    def back_propagate(self, input, data, actual, eta):
        delta = self.d_activate(actual[0]) * (data - actual[0])
        for i in range(1, self.s.shape[0]):
            delta += self.d_activate(actual[i]) * (data - actual[i]).T

        self.w += eta * (delta * input[2:])


    def train(self, data, epoch, eta):
        return


    def test(self, data):
        return
