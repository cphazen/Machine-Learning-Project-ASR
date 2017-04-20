# Built-in imports
import random

# External libraries:
import numpy as np

# Our own stuff:
from progress import Progress

def data_to_col(data, filter_size, stride_length):
    # Realigns data into a vector to be used for convolution
    #
    # Parameters:
    #   data            data to be realigned
    #   filter_size     size of filter
    #   stride_length   length of stride (that is, how much the filter moves)
    #
    # Returns:
    #   data reformatated as col

    # Print warning if filter and stride combination will not cover all data

    # Dimensions
    d, w, h = data.shape                                # data
    fw, fh = filter_size                                # filter

    # Debugging only
    # if ((w-fh) % stride_length != 0) or ((h-fw) % stride_length != 0): or \
    #    (fw < stride_length) or (fh < stride_length):
    #    print("[INFO] Invalid filter size and stride length. Data may be lost.")

    i = (w - fh)/stride_length + 1                      # number of filters taken horizontally
    j = (h - fw)/stride_length + 1                      # number of filters taken vertically

    # Get indices of data_to_col block
    base = np.arange(fh)[:,None] * h                    # Get vertical indices
    base = np.tile(base, fw) + np.arange(fw)            # Get horizontal indices
    base = base.ravel()[:,None]                         # Turn into vector

    # Get indices of data_to_col block for entire depth
    block = np.tile(base, d) + w * h * np.arange(d)     # Tile base across depth
    block = block.T.ravel()[:,None]                     # Turn into vector

    # Move block around
    hor = np.arange(j) * stride_length                      # Get horizontal starting indices for each block
    col = np.tile(block, j) + hor                           # Get first row of blocks
    ver = np.repeat(np.arange(i) * stride_length * h, j)    # Get vertical starting indices for each block
    col = np.tile(col, i) + ver                             # Tile down rows
    col = col.ravel()                                       # Flatten vector

    col = np.take(data, col).reshape(fw*fh*d, i*j)      # Get data and reshape vector
    return col

def col_to_data(col, data_size, filter_size, stride_length):
    # Undo col_to_data
    # NOTE: We're not using this, but it's a base to see what d_col_to_data
    #       does
    #
    # Parameters:
    #   col             col to convert back into data
    #   data            size of data to be realigned back into
    #   filter_size     size of filter
    #   stride_length   length of stride (that is, how much the filter moves)
    #
    # Returns:
    #   Reverse of the data_to_col function

    # Dimensions
    cw, ch = col.shape                              # col
    d, w, h = data_size                             # data
    fd, fw, fh = filter_size                        # filter

    i = (w - fh)/stride_length + 1                  # number of filters taken horizontally
    j = (h - fw)/stride_length + 1                  # number of filters taken vertically

    # Reshape
    data = np.zeros(data_size)                      # fill data array with zeros
    for k in xrange(ch):                            # for each column in col...
        block = col[:,k].reshape((d, fh, fw))       #   reshape into a block
        x = k / j * stride                          #   get horizontal starting index
        y = k % j * stride                          #   get vertical starting index
        data[:, x:x+fh, y:y+fw] = block             #   replace block

    return data

def d_col_to_data(col, data, filter_size, stride_length):
    # Calculates mean slope over all cols
    #
    # Parameters:
    #   col             deltas in the format of a col
    #   data            data to be used to calculate slope
    #   filter_size     size of filter
    #   stride_length   length of stride (that is, how much the filter moves)
    #
    # Returns:
    #   Filter-sized slopes

    # Dimensions
    cw, ch = col.shape                              # col
    d, w, h = data.shape                            # data
    fd, fw, fh = filter_size                        # filter

    i = (w - fh)/stride_length + 1                  # number of filters taken horizontally
    j = (h - fw)/stride_length + 1                  # number of filters taken vertically

    # Stretch column to propagate deltas
    stretched_col = np.tile(col, fd*fw*fh).reshape(fd*fw*fh,ch*cw)

    # Reshape
    deltas = np.zeros(filter_size)                  # fill delta array with zeros
    og_sum = np.zeros(filter_size)                  # fill og_sum array with zeros
    for k in xrange(ch):                            # for each column in col...
        block = stretched_col[:,k].reshape((fd, fw, fh)) #   reshape deltas into a block
        deltas += block

        x = k / j * stride_length                   #   get horizontal starting index
        y = k % j * stride_length                   #   get vertical starting index
        og_block = data[:, x:x+fh, y:y+fw]          #   get original block
        og_sum += block                             #   sum the deltas?

    slope = og_sum * deltas                         # calculate slope
    return slope

class CNN(object):
    def __init__(self, input_width, input_height, input_depth,
                       filter_width, filter_height,
                       filter_count, filter_stride, filter_padding,
                       output_width, output_height,
                       activate, d_activate):
        # DIMENSIONS
        self.id = (input_depth, input_height, input_width)
        self.fd = (input_depth, filter_height, filter_width)
        self.cd = (filter_count,
                   (input_height - filter_width + (2 * filter_padding))/filter_stride + 1,
                   (input_width - filter_height + (2 * filter_padding))/filter_stride + 1)
        self.bd = (filter_count, 1, 1)
        self.od = (1, output_height, output_width)

        self.fs = filter_stride
        self.fp = filter_padding

        # WEIGHTS
        self.fw = []
        for i in xrange(filter_count):
            #filter.append(np.random.randn(fd[0], fd[1], fd[2]))
            self.fw.append(np.random.randn(1, self.fd[0]*self.fd[1]*self.fd[2]))
        self.o1w = np.random.randn(self.od[1],1)
        self.o2w = np.random.randn(self.cd[0]*self.cd[1]*self.cd[2], self.od[2])

        # ACTIVATION
        self.activate = np.vectorize(activate)
        self.d_activate = np.vectorize(d_activate)

    def convolve(self, data, filter):
        # Perform convolution on layers
        #
        # Parameters:
        #   data        pre-padded data to convolve
        #   filter      filter weights to apply
        #
        # Returns:
        #   convolved data

        # Original convolve:
        # res = zeros(self.cd[0], self.cd[1], 1)
        # for j in xrange(0, self.cd[1], self.fs):
        #     for i in xrange(0, self.cd[0], self.fs):
        #         res[i, j, 1] = np.sum(self.data[i:i+self.fd[0], j:j+self.fd[1], :] * filter) + bias

        # Convolve using data_to_col:
        col_data = data_to_col(data, (self.fd[1],self.fd[2]), self.fs)  # convert data to columns
        res = np.dot(filter,col_data)                                   # multiply by filter
        return res

    def feed_forward(self, data):
        # Pads data, performs convolution, and fully connects layers into output
        #
        # Parameters:
        #   data        data to transform into approximation of classification
        #
        # Returns:
        #   approximation of classification

        # Add padding to data
        self.data = np.matrix(data).reshape(self.id)
        pd = self.id[0]
        pw = self.id[1] + 2 * self.fp
        ph = self.id[2] + 2 * self.fp
        self.padded_data = np.pad(data, [(self.fp,self.fp),(self.fp,self.fp)], 'constant', constant_values = 0.0)
        self.padded_data = self.padded_data.reshape((pd, pw, ph))

        # Convolve
        self.convolution_layer = np.zeros((self.cd[0], self.cd[1] * self.cd[2]))
        for i in xrange(self.cd[0]):
            self.convolution_layer[i,:] = self.convolve(self.padded_data, self.fw[i])
            self.convolution_layer[i,:] = self.activate(self.convolution_layer[i,:])

        # NOTE: If you want to pool, you do it here
        #       You'll need to change some dimensions around, though

        # Reformat/fully connect
        self.fc = self.convolution_layer.ravel()

        self.fc1 = self.o1w * self.fc
        self.fc1 = self.activate(self.fc1)

        self.fc2 = np.dot(self.fc1, self.o2w)
        self.fc2 = self.activate(self.fc2)
        return self.fc2.T

    def back_propagate(self, expected_output, eta):
        # Uses chain rule and gradient descent to account for error and
        # reset weights
        #
        # Parameters:
        #   expected_output     expected output of feed forward
        #   eta                 learning rate
        #
        # Returns:
        #   None

        # Error
        # FC LAYER 2 Error:
        error_o2w = expected_output - self.fc2.T
        delta_o2w = self.d_activate(self.fc2) * error_o2w.T
        slope_o2w = np.dot(delta_o2w.T, self.fc1).T
        # FC LAYER 1 Error:
        error_o1w = np.dot(delta_o2w, self.o2w.T)
        delta_o1w = self.d_activate(self.fc1) * error_o1w
        slope_o1w = np.dot(delta_o1w, self.fc.reshape(self.fc.size,1))
        # CONV Error:
        error_fc = np.dot(delta_o1w.T, self.o1w)
        delta_fc = self.d_activate(self.fc) * error_fc.T
        deconnect = delta_fc.reshape((self.cd[0], self.cd[2] * self.cd[1]))             # reformat data for loop

        slope_f = []
        for i in xrange(self.cd[0]):
            c_slice = np.matrix(deconnect[i,:])                                         # get slice of reformatted data
            slope_slice = d_col_to_data(c_slice, self.padded_data, self.fd, self.fs)    # calculate slope of summed deltas
            slope_f.append(slope_slice.reshape(1,self.fd[0]*self.fd[1]*self.fd[2]))     # save slop

        # Update
        self.o2w += eta * slope_o2w
        self.o1w += eta * slope_o1w
        for i in xrange(len(slope_f)):
            self.fw[i] += eta * slope_f[i]

        return

    def save(self, name):
        # Save current state
        #
        # Parameters:
        #   name            name of file to save arrays to
        # Returns:
        #   None
        name = name + '.npz'
        np.savez(name, fw=np.array(self.fw), o1w=self.o1w, o2w=self.o2w)
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
        if((np.array(self.fw).shape == a['fw'].shape) and (self.o1w.shape == a['o1w'].shape) and \
           (self.o2w.shape == a['o2w'].shape)):
            self.fw = list(a['fw'])
            self.o1w = a['o1w']
            self.o2w = a['o2w']
            res = True
        else:
            print('[ERROR] File dimensions do not match this network')
        return res

    def train(self, given_input, expected_output, epoch, learning_rate, reader):
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
                output = self.feed_forward(given_input[j])
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
