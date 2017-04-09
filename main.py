# Built-in imports:
import hashlib
import os
import math
import struct

# External libraries:
import numpy as np
np.seterr(all="ignore")         # NOTE: You may want to comment this out for testing

# Our own stuff:
from cnn import CNN
#from rnn import RNN

# TODO LIST:
# JOSH:
#   IMPORTANT: Custom audio -> feature vector
#   Play around with the variables
#   Just play around with everything I did and make it better
#   (Optional) Add pooling to CNN
#   (Optional) Add more layers to CNN
#   (Optional) Alter data to allow for isolated word
#   (Optional) Alter data to add phone alignments
#
# COURTNEY:
#   Make a readme.md
#   IMPORTANT: RNN for phones -> words
#   IMPORTANT: feature vectors -> words
#   IMPORTANT: feature vectors -> phones
#   Progress messages
#   Load/save functions
#   Make sure LibriSpeech imports properly
#   Testing function
#   Demo function

# READ & FORMAT DATA
def one_hot(item, dictionary):
    # The number of phones is finite, so we can just convert phones to separate
    # properties
    #
    # Parameters:
    #   item        item to be converted into a vector
    #   dictionary  list of all items
    #
    # Returns:
    #   vectorized item as a list of booleans
    return map(lambda x: 1 if x == item  else 0, dictionary)

def binary(i, length):
    # There are a ton of words, so we're going to use binary encoding
    #
    # Parameters:
    #   i           ordinal representation of vector
    #   length      number of bits
    #
    # Returns:
    #   vectorized word as a list of booleans
    m = bin(i)[2:].zfill(length)
    return map(lambda x: 1 if x == '1' else 0, m)

def load_list(file):
    # Read file into a list, delimited by new lines
    #
    # Parameters:
    #   file        name of file to be read
    #
    # Returns:
    #   each line of the file as an item in a list
    with open(file) as f:
        lines = f.readlines()
    lines = [x.split()[0].strip() for x in lines]
    return lines

def load_dictionary(file):
    # Read file into a hash map
    # Format:
    # key   values
    #
    # Parameters:
    #   file        name of file to be readlines
    #
    # Returns:
    #   hashmap where the first word of line is key and following words are
    #   values
    hashmap = {}
    with open(file) as f:
        for line in f:
            parsed = line.split()
            key = parsed[0].strip()
            value = parsed[1:]
            hashmap[key] = value
    return hashmap

def load_transcript(file, style):
    # Load transcript into an exiting hash map
    # stye = 'CMU AN4'
    #   file should be a text file with the format:
    #       transcription (key)
    #
    # style = 'LibriSpeech'
    #   file should be a directory containing all files, with lowest-level
    #   directories containing their own text file with the format:
    #       key transcription
    #
    # style = 'VoxForge'
    #   file should be a text file with the format:
    #       key transcription
    #
    # Parameters:
    #   file        file or directory containing transcripts (see above)
    #   style       one of the styles listed above, determines how to read file
    hashmap = {}
    if style == 'CMU AN4':
        with open(file) as f:
            for line in f:
                parsed = line.split()
                key = parsed[-1].strip()[1:-1]
                value = parsed[:-1]
                hashmap[key] = value

    elif style == 'LibriSpeech':
        for path, dirs, files in os.walk(file):
            for filename in [x for x in files if x.name.endswith('.trans.txt')]:
                curr_path = os.path.join(path, filename)
                with open(curr_path) as f:
                    for line in f:
                        key = line[0].strip()
                        value = line[1:].strip()
                        hashmap[key] = value

    elif style == 'VoxForge':
        with open(file) as f:
            for line in f:
                key = line[0].strip()
                vlaue = line[1:].strip()
                hashmap[key] = value

    else:
        print('Transcript style not supported')
    return hashmap

def mfcc_to_features(file):
    # TESTING ONLY - We'll want to be creating our own feature vectors for
    #                audio later!
    #
    # Parameters:
    #   file            name of mfc file to convert
    features = []
    with open(file, 'rb') as f:
        header = struct.unpack('i', f.read(4))[0]
        fl = f.read(4)
        while (len(fl)) == 4:
            feature = []
            for j in xrange(13):
                value = struct.unpack('f', fl)[0]
                if math.isnan(value):
                    value = 0.0
                feature.append(value)
                fl = f.read(4)
            features.append(feature)
    return features

def setup_features(directory, phones, dictionary, transcript, words_to_phones):
    # Sets up feature vectors
    #
    # Parameters:
    #   directory       directory containing audio files
    #   phones          list of possible phones
    #   dictionary      list of posible words
    #   transcript      hashmap mapping file to words
    #   words_to_phones hashmap mapping words to phones
    #
    # Returns:
    #   list of all audio feature vectors
    #   list of all phone feature vectors
    #   list of all word feature vectors
    audio_features = []
    word_features = []
    phone_features = []
    bits = int(math.ceil(math.log(len(dictionary),2)))

    # Look through directory to find all audio files
    for path, dirs, files in os.walk(directory):
        for filename in [x for x in files if x.endswith('.mfc')]:  # TODO: Change .mfc to .wav
            curr_path = os.path.join(path, filename)

            # Converts raw audio to feature vector
            # Adds feature vector to list of feature vectors
            audio_features.append(mfcc_to_features(curr_path))     # TODO: Replace with our own audio to feature vector

            # Matches audio file to transcript file
            name, ext = os.path.splitext(filename)

            # Converts transcript into words and adds them to feaure vector
            curr_word = []
            words = transcript[name]
            for word in words:
                ordinal = dictionary.index(word)
                curr_word.append(binary(ordinal, bits))         # NOTE: You could probably use one_hot for smaller dictionaries
            word_features.append(curr_word)

            # Converts words into phones and adds them to feature vector
            curr_phone = []
            for word in words:
                curr_phone.append(one_hot('SIL', phones))
                word_phones = words_to_phones[word]
                for phone in word_phones:
                    curr_phone.append(one_hot(phone, phones))
            phone_features.append(curr_phone)

    # Set all matrices to the same size
    max_audio = len(max(audio_features, key=len))
    empty_audio = [0.0] * 13                                # TODO: This will need to change based on our own audio format
    audio_features = regularize_length(max_audio + 1, audio_features, empty_audio)  # +1 to add bias

    max_phones = len(max(phone_features, key=len))
    empty_phone = one_hot('SIL', phones)
    phone_features = regularize_length(max_phones + 1, phone_features, empty_phone) # +1 to add bias

    max_words = len(max(word_features, key=len))
    empty_word = binary(0, bits)
    word_features = regularize_length(max_words, word_features, empty_word)         # words will never be inputs, so no need for bias

    return audio_features, phone_features, word_features

def regularize_length(length, vectors, empty):
    for vector in vectors:
        empty_vectors = [empty] * (length - len(vector))
        vector += empty_vectors
    return vectors

# SAVE/LOAD STATE
# TODO: Training takes a lot of time

# ACTIVATION FUNCTIONS
def sign(x):
    # Sign
    return 1 if x >= 0 else -1

def sigmoid(x):
    # Sigmoid
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(x):
    # Sigmoid (derivative)
    return x * (1.0 - x)

def relu(x):
    # ReLU
    return x if x > 0 else 0

def d_relu(x):
    # ReLU (derivative)
    return 1 if x > 0 else 0

def tanh(x):
    # Tanh
    return np.tanh(x)

def d_tanh(x):
    # Tanh (derivative)
    return 1.0 - x**2


# MAIN
def main():
    # VARIABLES - FILES
    phone_file = 'an4\\etc\\an4.phone'                        # path to file of phone list
    dictionary_file = 'an4\\etc\\an4.dic'                     # path to file of word list
    transcript_file = 'an4\\etc\\an4_train.transcription'     # path/directory of transcript file(s)
    transcript_type = 'CMU AN4'                               # library being used (to find & parse transcript file)
    words_to_phones_file = 'an4\\etc\\an4.dic'                # path to file matching words with phones
    directory = 'an4\\feat\\an4_clstk'                        # directory containing audio files

    # VARIABLES - NUMBERS
    learning_rate = .2
    epoch = 1
    # Layer 1:
    filter_width = 2
    filter_height = 2
    filter_count = 3
    filter_stride = 1
    filter_padding = 0
    # Layer 2:

    # VARIABLES - FUNCTIONS
    activate = sigmoid
    d_activate = d_sigmoid

    # SETUP
    print("[INFO] Loading files")
    phones = load_list(phone_file)
    dictionary = ['<sil>'] + load_list(dictionary_file)
    transcript = load_transcript(transcript_file, transcript_type)
    words_to_phones = load_dictionary(dictionary_file)
    words_to_phones['<sil>'] = 'SIL'

    print("[INFO] Setting up features")
    audio_features, phone_features, word_features = setup_features(directory, phones, dictionary, transcript, words_to_phones)
    # TODO: Split up our features for testing?

    # TRAINING
    # LAYER 1: AUDIO -> PHONES
    print("[INFO] Training LAYER 1: Audio -> phones")
    l1_input = np.matrix(audio_features[0]).shape
    l1_output = np.matrix(phone_features[1]).shape
    l1 = CNN(l1_input[0], l1_input[1], 1,
             filter_width, filter_height,
             filter_count, filter_stride, filter_padding,
             l1_output[0], l1_output[1],
             activate, d_activate)
    output_features = l1.train(audio_features, phone_features, epoch, learning_rate)

    # NOTE: You can reformat output_features (a list of phones) to be
    #       fed into the next layer here

    # LAYER 2: PHONES -> WORDS
    print("[INFO] Training LAYER 2: Phones-> words")


    # TESTING
    #print("[INFO] Beginning testing...")

    return

main()
