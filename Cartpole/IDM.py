import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD , Adam
from tensorflow.keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from tensorflow.keras.layers import LeakyReLU 
from matplotlib import pyplot

#from keras.utils.vis_utils import plot_model
import pickle 
import copy


# Import Data

filename = 'Data/NState.npy'
ns = pickle.load(open(filename, 'rb'))
filename = 'Data/Action.npy'
a = pickle.load(open(filename, 'rb'))
filename = 'Data/State.npy'
s = pickle.load(open(filename, 'rb'))
filename = 'Data/Diff.npy'
d = pickle.load(open(filename, 'rb'))

temp = np.zeros((a.size,2))
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

temp = np.zeros((test_a.size,2))
for i in range(len(test_a)):
    temp[i][test_a[i]] = 1
test_a = temp


pause = 0 
while pause == 1:
    pause = 1

# This is input current state and next state
curr_state = keras.Input(shape=(4,))
n_state = keras.Input(shape=(4,))
# "encoded" is the encoded representation of the input
state_n_state = concatenate([curr_state, n_state])
encoded = Dense(16)(state_n_state)
encoded = LeakyReLU(alpha=0.2)(encoded)
encoded = Dense(16)(encoded)
encoded = LeakyReLU(alpha=0.2)(encoded)
pred_action = layers.Dense(2)(encoded)
pred_action_hot = tf.one_hot(tf.argmax(pred_action, axis=1), 2, name="one_hot")

p=1
while p==1:
    p=1

#loss_fn = tf.keras.losses.sparse_categorical_crossentropy( true_action?????????, pred_action_hot, from_logits=True, axis=-1)
#tf.keras.utils.plot_model(IDM, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

loss_fn = tf.reduce_mean(tf.nn.softmax(labels=a, logits=pred_action))
#opt = SGD(lr=0.005, momentum=0.9)
opt = Adam(lr=0.0005)
#opt = tf.keras.optimizers.RMSprop(learning_rate=0.0005)

# This model maps an input to its encoded representation
IDM = keras.Model(inputs=[curr_state,n_state], outputs=pred_action_hot)
print(IDM.summary())
IDM.compile(loss=loss_fn, optimizer=opt, metrics=['accuracy'])

pause = 0
while pause ==1:
    pause =1


history = IDM.fit(x=[s,ns], y=a,
                epochs=50,
                batch_size=32,
                shuffle=True)
            

print("Training Done")
history_dict = history.history
#print(history_dict.keys())

pause = 1 
while pause == 1:
    pause = 1

# evaluate the model
_, train_mse = IDM.evaluate([s,ns], a, verbose=0)
_, test_mse = IDM.evaluate([test_s,test_a], test_ns, verbose=0)
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


# Encode and decode some digits
# Note that we take them from the *test* set
#encoded_imgs = encoder.predict(x_test)
#decoded_imgs = decoder.predict(encoded_imgs)
pred_ns = IDM.predict([test_s,test_a])

print("prediction ")
for i in range(10):
    print(test_ns[i],pred_ns[i],np.abs(test_ns[i]-pred_ns[i]))