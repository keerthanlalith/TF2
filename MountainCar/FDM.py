import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD , Adam
from tensorflow.keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from tensorflow.keras.layers import LeakyReLU 
from matplotlib import pyplot
import pickle 
import copy

state_dim=2
action_dim =3

# This is our input image
curr_state = keras.Input(shape=(state_dim,))
curr_action = keras.Input(shape=(action_dim,))
# "encoded" is the encoded representation of the input
curr_state_action = concatenate([curr_state, curr_action])
encoded = Dense(16)(curr_state_action)
encoded = LeakyReLU(alpha=0.2)(encoded)
encoded = Dense(16)(encoded)
encoded = LeakyReLU(alpha=0.2)(encoded)
n_state = layers.Dense(state_dim)(encoded)

# This model maps an input to its encoded representation
FDM = keras.Model(inputs=[curr_state,curr_action], outputs=n_state)

#print(FDM.summary())
tf.keras.utils.plot_model(FDM, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


#opt = SGD(lr=0.005, momentum=0.9)
opt = Adam(lr=0.0005)
#opt = tf.keras.optimizers.RMSprop(learning_rate=0.0005)
FDM.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])

# Import Data

filename = 'Data/NState.npy'
ns = pickle.load(open(filename, 'rb'))
filename = 'Data/Action.npy'
a = pickle.load(open(filename, 'rb'))
filename = 'Data/State.npy'
s = pickle.load(open(filename, 'rb'))
filename = 'Data/Diff.npy'
d = pickle.load(open(filename, 'rb'))

temp = np.zeros((a.size,action_dim))
for i in range(len(a)):
    temp[i][a[i]] = 1
a = temp
# test set
filename = 'Data/TNState.npy'
test_ns = pickle.load(open(filename, 'rb'))
filename = 'Data/TAction.npy'
test_a = pickle.load(open(filename, 'rb'))
filename = 'Data/TState.npy'
test_s = pickle.load(open(filename, 'rb'))
filename = 'Data/TDiff.npy'
test_d = pickle.load(open(filename, 'rb'))

temp = np.zeros((test_a.size,action_dim))
for i in range(len(test_a)):
    temp[i][test_a[i]] = 1
test_a = temp


pause = 0 
while pause == 1:
    pause = 1


#history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
#FDM = keras.models.load_model('saved_model/FDM')

history = FDM.fit(x=[s,a], y=ns,
                epochs=100,
                batch_size=32,
                shuffle=True,
                validation_data=([test_s,test_a], test_ns))

print("Training Done")
history_dict = history.history
#print(history_dict.keys())
FDM.save('saved_model/FDM')

pause = 0 
while pause == 1:
    pause = 1

# evaluate the model
_, train_mse = FDM.evaluate([s,a], ns, verbose=0)
_, test_mse = FDM.evaluate([test_s,test_a], test_ns, verbose=0)
print('Train: %.8f, Test: %.8f' % (train_mse, test_mse))

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

# Encode and decode some digits
# Note that we take them from the *test* set
#encoded_imgs = encoder.predict(x_test)
#decoded_imgs = decoder.predict(encoded_imgs)

pred_ns = FDM.predict([test_s,test_a])

print("prediction ")
for i in range(100):
    print(test_ns[i],pred_ns[i],np.abs(test_ns[i]-pred_ns[i]))
