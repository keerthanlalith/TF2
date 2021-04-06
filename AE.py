import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD

import pickle 
import copy


# This is our input image
curr_state = keras.Input(shape=(4,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(16, activation='relu')(curr_state)
encoded = layers.Dense(16, activation='relu')(encoded)
encoded = layers.Dense(4, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(16, activation='relu')(encoded)
decoded = layers.Dense(16, activation='relu')(decoded)
decoded = layers.Dense(4, activation='sigmoid')(decoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(curr_state, decoded)

# This model maps an input to its encoded representation
encoder = keras.Model(curr_state, encoded)

# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(4,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-3]


# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

#opt = SGD(lr=0.005, momentum=0.9)
opt = tf.keras.optimizers.RMSprop(learning_rate=0.0005)
autoencoder.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])

# Import Data

filename = 'Data/NState.npy'
ns = pickle.load(open(filename, 'rb'))
filename = 'Data/Action.npy'
a = pickle.load(open(filename, 'rb'))
filename = 'Data/State.npy'
s = pickle.load(open(filename, 'rb'))
filename = 'Data/Diff.npy'
d = pickle.load(open(filename, 'rb'))

# test set
filename = 'Data/TNState.npy'
test_ns = pickle.load(open(filename, 'rb'))
filename = 'Data/TAction.npy'
test_a = pickle.load(open(filename, 'rb'))
filename = 'Data/TState.npy'
test_s = pickle.load(open(filename, 'rb'))
filename = 'Data/TDiff.npy'
test_d = pickle.load(open(filename, 'rb'))



#history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)

history = autoencoder.fit(s, ns,
                epochs=5000,
                batch_size=256,
                shuffle=True,
                validation_data=(test_s, test_s))

print("All good")

pause = 1
while pause==1:
    pause=1


# Encode and decode some digits
# Note that we take them from the *test* set
#encoded_imgs = encoder.predict(x_test)
#decoded_imgs = decoder.predict(encoded_imgs)
decoded_imgs = autoencoder.predict(x_test)

# Use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()