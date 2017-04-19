# https://pypi.python.org/pypi/numpy
# http://aka.ms/vcpython27

import math, os, sys, time, io, wave, struct
import numpy as np

from sinusoid import Sinusoid, Phasor, ImproperlyConfigured

class WaveFileDFT(object):
    """
    Uses numpy's fft function and
    Python's "Wave" class to extract
    the sound vector from the wav
    file and perform frequency analysis
    on any given set of points from
    the vector.
    """

    def __init__(self, filename):
        """
        For simplicity reasons, we will only open
        the file once. We will extract the sound
        vector into an array, and the metadata
        will be preserved as well.
        We will just assume that the files it
        will be given in this version are not
        so large that they overflow the process's
        allotted memory.
        """

        self._filename = filename

        w1 = wave.open(filename, 'r') # change back to 'self._filename' once we find the bug

        self._nchannels = w1.getnchannels()
        self._samplewidth = w1.getsampwidth()
        self._framerate = w1.getframerate()
        self._nframes = w1.getnframes()
        self._comptype = w1.getcomptype()
        self._compname = w1.getcompname()
        self._params = w1.getparams()

        self._audio_byte_vector = []
        self._audio_vector = []

        for i in range(0, self._nframes):
            self._audio_byte_vector.append(w1.readframes(1))

        w1.close()

        if (len(self._audio_byte_vector) != self._nframes):
            raise ImproperlyConfigured("The correct number " \
                                        "of frames were not " \
                                        "copied into the " \
                                        "audio vector!")

        if (self._nchannels == 1):
            for x in self._audio_byte_vector:
                self._audio_vector.append(float(struct.unpack("<h", x)[0]))
        elif (self._nchannels == 2):
            for x in self._audio_byte_vector:
                self._audio_vector.append(float(struct.unpack("<h", x[0:2])[0]))
        else:
            raise ImproperlyConfigured("wav file must be either stereo or mono")

    def get_number_of_audio_frames(self):
        return len(self._audio_vector)

    def get_section_of_vector(self, starting_index, ending_index):
        if (starting_index > ending_index):
            raise ImproperlyConfigured("starting_index must be less \
    	                                than or equal to ending_index");
        return self._audio_vector[starting_index : (ending_index+1)]

    def get_n_point_frequency_vector(self, starting_index, number_of_points, first_half=True):
        time_domain_vector = self.get_section_of_vector(starting_index, starting_index + number_of_points)
        ndarray_result = np.fft.fft(time_domain_vector, number_of_points)
        frequency_domain_vector = ndarray_result.tolist()
        frequency_domain_abs_vector = []
        for element in frequency_domain_vector:
            frequency_domain_abs_vector.append(abs(element))
        if first_half: return frequency_domain_abs_vector[0:(len(frequency_domain_abs_vector)/2)]
        return frequency_domain_abs_vector

    def plot_audio_vector_with_matplotlib(self, starting_index, ending_index):
        import matplotlib.pyplot as plt
        plt.plot(self.get_section_of_vector(starting_index, ending_index))
        plt.show()

    def plot_frequency_spectrum_with_matplotlib(self, starting_index, number_of_points, first_half=True):
        import matplotlib.pyplot as plt
        plt.plot(self.get_n_point_frequency_vector(starting_index, number_of_points))
        plt.show()

    def get_2d_array_of_frequency_spectrum_windows(self, window_size, overlap_points):

        """
        For simplicity reasons, if the
        window size minus the overlap
        does not evenly divide the total
        number of frames in the sound
        vector, we just generate enough
        windows so that the last one
        goes off the end of the sound
        vector (same length as the
        other windows, only with some
        zero-padding).
        """

        total_vec_len = self.get_number_of_audio_frames()
        if (window_size > total_vec_len):
            raise ImproperlyConfigured("window size must be less than " \
                                       "or equal to the total length of " \
                                       "the sound vector")

        number_of_windows = int(math.ceil(float(total_vec_len)/float(window_size-overlap_points)))

        matrix = []

        for x in range(0, number_of_windows):
            matrix.append(self.get_n_point_frequency_vector(x*(window_size-overlap_points), window_size, first_half=False))

        return matrix

    def get_number_of_windows_for_window_size_and_overlap_points(self, window_size, overlap_points):
        total_vec_len = self.get_number_of_audio_frames()
        return int(math.ceil(float(total_vec_len)/float(window_size-overlap_points)))

    #