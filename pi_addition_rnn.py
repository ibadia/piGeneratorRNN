# -*- coding: utf-8 -*-
#This code is just the little modification of https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py
'''An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)

Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.

Two digits inverted:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs

Three digits inverted:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs

Four digits inverted:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs

Five digits inverted:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs

'''
from __future__ import print_function
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
import numpy as np
from six.moves import range
from numpy import genfromtxt
from keras.utils.np_utils import to_categorical
class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)
class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


#chars = '0123456789+ ' 
#ctable = CharacterTable(chars, MAXLEN)


def GenerateData(terms, DIGITS, sign):
	


	MAXLEN=20
	chars = '0123456789'
	ctable = CharacterTable(chars, MAXLEN)
	questions = []
	expected = []
	seen = set()
	print('Generating data...')
	DATA=genfromtxt('pi_data.txt',delimiter=',', dtype=str)
	ss=int(DATA.shape[1])
	ss-=1
	questions=DATA[:,:ss]
	expected=DATA[:,ss:]
	print (questions.shape)
	print (expected.shape)

	
	print('Total addition questions:', len(questions))
	#print (questions[0])
	MAXANS=1
	
	print('Vectorization...')
	X = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
	y = np.zeros((len(questions), MAXANS, len(chars)), dtype=np.bool)
	for i, sentence in enumerate(questions):
		X[i] = ctable.encode(sentence, maxlen=MAXLEN)
		
	print ("sdfsd")
	print (X[0])
	for i, sentence in enumerate(expected):
		y[i] = ctable.encode(sentence, maxlen=MAXANS)

	#print ("asdasdadsa")
	print (expected[0])
	print (y[0])

	# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
	#indices = np.arange(len(y))
	#np.random.shuffle(indices)
	#X = X[indices]
	#y = y[indices]
	return X,y,MAXLEN,chars,MAXANS, expected


# Parameters for the model and dataset
TRAINING_SIZE = 50000
INVERT = False
# Try replacing GRU, or SimpleRNN
RNN = recurrent.LSTM
HIDDEN_SIZE = 150
BATCH_SIZE = 200
LAYERS = 2

X,y,MAXLEN,chars,DIGITS, expected=GenerateData(3,3,'+')
'''
y=expected
y=y.astype(int)
y=to_categorical(y,10)'''
#y=y.reshape(y.shape[0],-1,1)

# Explicitly set apart 10% for validation data that we never train over
split_at = len(X) - len(X) / 10
(X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
(y_train, y_val) = (y[:split_at], y[split_at:])

print(X_train.shape)
print(y_train.shape)
chars = '0123456789'
ctable = CharacterTable(chars, MAXLEN)
	
print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(DIGITS))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print (model.summary())
print (X_train[0])
print (y_train[0])
print ("above were printed for sanity check")
#model.load_weights('piWeights.h5')
#print("Weights loaded Sucessfully")
# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 200):

	
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(X_val, y_val))
    model.save_weights('piWeights.h5', overwrite=True)
    print ("weights SAved SUCESSFULLy")
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        q = ctable.decode(rowX[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if INVERT else q)
        print('T', correct)
        print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
        print('---')
