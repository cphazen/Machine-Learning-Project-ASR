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

        for x in self._audio_byte_vector:
            self._audio_vector.append(float(struct.unpack("<h", x)[0]))

    def get_number_of_audio_frames(self):
        return len(self._audio_vector)

    def get_section_of_vector(self, starting_index, ending_index):
        if (starting_index > ending_index):
            raise ImproperlyConfigured("starting_index must be less \
    	                                than or equal to ending_index");
        return self._audio_vector[starting_index : (ending_index+1)]

    def get_n_point_fft(self, starting_index, number_of_points):
        time_domain_vector = self.get_section_of_vector(starting_index, starting_index + number_of_points)
        ndarray_result = np.fft.fft(time_domain_vector, number_of_points)
        frequency_domain_vector = ndarray.tolist()
        return abs(frequency_domain_vector)

    def plot_audio_vector_with_matplotlib(self, starting_index, ending_index):
        import matplotlib.pyplot as plt
        plt.plot(self.get_section_of_vector(starting_index, ending_index))
        plt.show()

    def plot_frequency_spectrum_with_matplotlib(self, starting_index, number_of_points):
        import matplotlib.pyplot as plt
        plt.plot(self.get_n_point_fft(starting_index, number_of_points))
        plt.show()

    #