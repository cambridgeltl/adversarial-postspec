import sys, os, codecs, operator, time
reload(sys)
sys.setdefaultencoding('utf8')

import keras
import tensorflow as tf
import numpy as np
import theano
import random

from sklearn.preprocessing import normalize

from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras import losses
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

# Hyper-parameters
encoding_dim = 512
initial = 'glorot_normal'
NUM_NEG = 25
BATCH_SIZE = 32
EPOCHS = 15
DEPTH = 4
delta_margin = 0.6


# Define custom activation: SWISH
def swish(x):
	return (x*K.sigmoid(x))

get_custom_objects().update({'swish': Activation(swish)})

# Define custom objective: this is standard hinge without negative examples
def false_hinge(y_true, y_pred):
    return K.mean(K.maximum(delta_margin - y_true * y_pred, 0.), axis=-1)

# This is a slightly deficient max-margin objective: random confounders are used as negative examples (and I also do not control for the rare case scenario where it can turn out that some of the items in new_true are exactly at the same position as in y_true
def max_margin(y_true, y_pred):
	cost = 0.0
	for i in xrange(0, NUM_NEG):
		new_true = tf.random_shuffle(y_true)
		cost += K.maximum(delta_margin - y_true * y_pred + new_true * y_pred, 0.)
		

	return K.mean(cost, axis=-1)


# Step 1: Prepare the input and output data
# Step 1a: Prepare the input vectors: _distrib contains only distributional vectors that later got adjusted by the Attract-Repel procedure (i.e., seen words)
with open("../vectors/glove300_distrib.vectors", "r") as in_file:
        lines = in_file.readlines()

in_file.close()

input_dic = {}
output_dic = {}

for line in lines:
	item = line.strip().split()
	dkey = item.pop(0)
	vector = np.array(item, dtype='float32')
	norm = np.linalg.norm(vector)
	input_dic[dkey] = vector/norm

# Step 1b: Prepare the input vectors: _ar contains only AR-adjusted vectors (i.e., seen words): the same number of words has to be in _distrib.vectors and _ar.vectors
with open("../vectors/glove300_ar.vectors", "r") as in_filear:
        linesar = in_filear.readlines()

in_filear.close()

for line in linesar:
        item = line.strip().split()
        dkey = item.pop(0)
        vector = np.array(item, dtype='float32')
        norm = np.linalg.norm(vector)
        output_dic[dkey] = vector/norm

# Step 1c
x_train_corrupted = []
x_train = []
for key in input_dic:
	distorted_instance = input_dic[key]
	correct_instance = output_dic[key]
	
	x_train_corrupted.append(distorted_instance)
	x_train.append(correct_instance)

counter = 0

x_train_corrupted = np.asarray(x_train_corrupted)
x_train = np.asarray(x_train)

# Activation functions, pick one
lrelu = keras.layers.advanced_activations.LeakyReLU(alpha=0.5)
prelu = keras.layers.advanced_activations.PReLU(init='zero', weights=None)
elu = keras.layers.advanced_activations.ELU(alpha=1.0)

activ_function = lrelu

### MODEL DEFINITION STARTS HERE ###
model = Sequential()

input_vec = Input(shape=(300,))
model.add(Dense(encoding_dim,input_dim=300,kernel_initializer=initial))
#model.add(BatchNormalization())
model.add(Activation(activ_function))

for i_depth in xrange(0,DEPTH):
	model.add(Dense(encoding_dim, kernel_initializer=initial))
	model.add(Activation(activ_function))
# I experimented with dropouts and batch normalization: it does not really affect the results (dropout makes it worse, which makes sense to me); batch normalization: no effect
##model.add(Dropout(0.3))
##model.add(BatchNormalization())

last_hidden = Dense(300, kernel_initializer=initial)
#model.add(BatchNormalization())
model.add(last_hidden)

# Compile the model
model.compile(optimizer='adam', loss=max_margin)

es = [keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=1, mode='auto')]

model.fit(x_train_corrupted, x_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
		verbose=True,
		validation_split=0.1,
		callbacks = es)


# Now the next step is to encode the words from SimVerb and SimLex (as I do only word similarity tests)
# Step 2a: Get the entire input embedding space (the distributional embeddings)

with open("../vectors/glove300_prefix.vectors", "r") as in_file:
        lines = in_file.readlines()

in_file.close()

input_dic = {}

print >> sys.stderr, "Reading original vectors..."
for line in lines:
        item = line.strip().split()
        dkey = item.pop(0)
        vector = np.array(item, dtype='float32')
        norm = np.linalg.norm(vector)
        input_dic[dkey] = vector/norm

print >> sys.stderr, "Reading SimLex and SimVerb words..."
with open("../vocab/simlexsimverb.words", "r") as in_file:
        unseen_words = in_file.readlines()
in_file.close()


# Now translate the unseen words to the target AR-specialised vector space
all_keys = input_dic.keys()


outfile = sys.argv[1]
outfilestr = "../results/" + str(outfile)

fenc = open(outfilestr, "w")
for item in unseen_words:
	key = item.strip()
	if key not in all_keys:
		continue
	vector = [input_dic[key]]
	vector = np.asarray(vector)
	# Final transformation
	encoded_vector_a = model.predict(vector)
	encoded_vector_nn = encoded_vector_a[0]
	
	# Now normalize the vector
	encoded_vector_n = normalize(encoded_vector_nn.reshape(1,-1), norm='l2', axis=1)
	
	encoded_vector = np.ndarray.tolist(encoded_vector_n)
	encstr = str(key) + " "  + " ".join(map(str,encoded_vector[0])) + "\n"
	fenc.write(encstr)

fenc.close()
