import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD , Adam, RMSprop
from tensorflow.keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from tensorflow.keras.layers import LeakyReLU 
from matplotlib import pyplot
import pickle 
import copy


# Input state
curr_state = keras.Input(shape=(4,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(16, activation='relu')(curr_state)
encoded = layers.Dense(16, activation='relu')(encoded)
encoded = layers.Dense(4, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the next state
decoded = layers.Dense(16, activation='relu')(encoded)
decoded = layers.Dense(16, activation='relu')(decoded)
n_state = layers.Dense(4, activation='sigmoid')(decoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(curr_state, n_state)

# This model maps an input to its encoded representation
encoder = keras.Model(curr_state, encoded)

# This is our encoded (4-dimensional) input
encoded_input = keras.Input(shape=(4,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-3]


# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

#print(autoencoder.summary())
tf.keras.utils.plot_model(autoencoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4,
    decay_steps=1000,
    decay_rate=0.9)
#opt = keras.optimizers.Adam(learning_rate=lr_schedule)
#opt = SGD(lr=0.005, momentum=0.9)
#opt = Adam(lr=0.0005)
opt = tf.keras.optimizers.RMSprop(learning_rate=0.00015)
autoencoder.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
#autoencoder.compile(loss='mean_squared_logarithmic_error', optimizer=opt, metrics=['mse'])

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


# training
history = autoencoder.fit(x=s, y=ns,
                epochs=100,
                batch_size=32,
                shuffle=True,
                verbose=0,
                validation_data=(test_s, test_ns))
'''
history = autoencoder.fit(x=s, y=ns,
                epochs=50,
                batch_size=64,
                shuffle=True)

'''

print("Training Done")

pause = 0
while pause==1:
    pause=1


# Encode and decode some states
# Note that we take them from the *test* set
encoded_state = encoder.predict(test_s)
decoded_nstate = decoder.predict(encoded_state)
decoded_n_state = autoencoder.predict(test_s)

print("Training Done")
history_dict = history.history
#print(history_dict.keys())

pause = 0 
while pause == 1:
    pause = 1

# evaluate the model
_, train_mse = autoencoder.evaluate(s, ns, verbose=0)
_, test_mse = autoencoder.evaluate(test_s, test_ns, verbose=0)
print('Train: %.6f, Test: %.6f' % (train_mse, test_mse))

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot mse during training
pyplot.subplot(212)
pyplot.title('Mean Squared Error')
pyplot.plot(history.history['mse'], label='train')
pyplot.plot(history.history['val_mse'], label='test')
pyplot.legend()
pyplot.show()

pause = 0 
while pause == 1:
    pause = 1


# Predict next state 
# Note that we take them from the *test* set
pred_ns = autoencoder.predict(test_s)
d_err = np.abs(test_ns - pred_ns)
#pred_ns = denormalise (pred_ns)
d_err_1 = np.abs(test_ns - pred_ns)
p_err =np.mean(d_err*d_err,axis=0)
p_err2 =np.mean(d_err_1*d_err_1,axis=0)
    
print(p_err,p_err2)

n=len(test_s)
n = int(input('Enter your number of test data to predict:'))

print ("autoencoder diff output, Normalised Test diff,  Difference")
    
for i in range(n):
    for j in range(4):
        pred_ns[i][j]=float("{:5.6f}".format(pred_ns[i][j]))
        test_ns[i][j]=float("{:5.6f}".format(test_ns[i][j]))
        d_err[i][j]=float("{:5.6f}".format(d_err[i][j]))

for i in range(n):
    print(pred_ns[i],test_ns[i],d_err[i])

print("------")
    
# Denormalise the data
#pred_ns = denormalise (pred_ns)
d_err_1 = np.abs(test_ns - pred_ns)

for i in range(n):
    for j in range(4):
        pred_ns[i][j]=float("{:5.6f}".format(pred_ns[i][j]))
        test_ns[i][j]=float("{:5.6f}".format(test_ns[i][j]))
        d_err_1[i][j]=float("{:5.6f}".format(d_err_1[i][j]))

for i in range(n):
    print(pred_ns[i],test_ns[i],d_err_1[i])
print("------")

