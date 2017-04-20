# https://pypi.python.org/pypi/numpy
# http://aka.ms/vcpython27

import math, os, sys, time, io, wave, numpy

from sinusoid import Sinusoid, Phasor # probably wont need these, will delete them later if that be the case

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

		self._filename = str(filename)

		w1 = wave.open(self._filename, 'r')

        self._nchannels = w1.getnchannels()
        self._samplewidth = w1.getsampwidth()
        self._framerate = w1.getframerate()
        self._nframes = w1.getnframes()
        self._comptype = w1.getcomptype()
        self._compname = w1.getcompname()
        self._params = w1.getparams()

        self._audio_vector = []

        for i in range(0, self._nframes):
        	self._audio_vector.append(w1.readframes(1))

        if (len(self._audio_vector) != self._nframes):
        	raise ImproperlyConfigured("The correct number \
        		                        of frames were not \
        		                        copied into the \
        		                        audio vector!")

        #
