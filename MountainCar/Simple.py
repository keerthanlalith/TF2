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
curr_state = keras.Input(shape=(2,))

# 2layer neural network to predict the next state
encoded = Dense(32)(curr_state)
encoded = LeakyReLU(alpha=0.2)(encoded)
encoded = Dense(32)(encoded)
encoded = LeakyReLU(alpha=0.2)(encoded)
n_state = layers.Dense(2)(encoded)

# This model maps an input to its next state
AE = keras.Model(inputs=curr_state, outputs=n_state)

#print(AE.summary())
tf.keras.utils.plot_model(AE, to_file='AE_model_plot.png', show_shapes=True, show_layer_names=True)

#opt = keras.optimizers.Adam(learning_rate=lr_schedule)
#opt = SGD(lr=0.0005, momentum=0.9)
opt = Adam(lr=0.0005)
#opt = tf.keras.optimizers.RMSprop(learning_rate=0.0005)
AE.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
#AE.compile(loss='mean_squared_logarithmic_error', optimizer=opt, metrics=['mse'])

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

pause = 0 
while pause == 1:
    pause = 1

# training
history = AE.fit(x=s, y=ns,
                epochs=100,
                batch_size=64,
                shuffle=True,
                verbose=1,
                validation_data=(test_s, test_ns))

AE.save('saved_model/AE')

print("Training Done")
history_dict = history.history
#print(history_dict.keys())

pause = 0 
while pause == 1:
    pause = 1

# evaluate the model
_, train_mse = AE.evaluate(s, ns, verbose=0)
_, test_mse = AE.evaluate(test_s, test_ns, verbose=0)
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
pred_ns = AE.predict(test_s)
d_err = np.abs(test_ns - pred_ns)
#pred_ns = denormalise (pred_ns)
d_err_1 = np.abs(test_ns - pred_ns)
p_err =np.mean(d_err*d_err,axis=0)
p_err2 =np.mean(d_err_1*d_err_1,axis=0)
    
print(p_err,p_err2)

n=len(test_s)
n = int(input('Enter your number of test data to predict:'))

print ("AE next state, True next state,  Difference(true - predicted)")
    
for i in range(n):
    for j in range(2):
        pred_ns[i][j]=float("{:5.6f}".format(pred_ns[i][j]))
        test_ns[i][j]=float("{:5.6f}".format(test_ns[i][j]))
        d_err[i][j]=float("{:5.6f}".format(d_err[i][j]))

for i in range(n):
    print(pred_ns[i],test_ns[i],d_err[i])

print("------")
print(p_err,p_err2)

    