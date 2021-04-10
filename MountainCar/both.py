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
import os



# Input state
AE_state = keras.Input(shape=(4,),name="AE_state")

# 2layer neural network to predict the next state
encoded = Dense(32,name="dense1_NS")(AE_state)
encoded = LeakyReLU(alpha=0.2,name="LeakyRelu1_NS")(encoded)
encoded = Dense(32,name="dense2_NS")(encoded)
encoded = LeakyReLU(alpha=0.2,name="LeakyRelu2_NS")(encoded)
n_state = layers.Dense(4,name="dense3_NS")(encoded)


# Input state
curr_state = keras.Input(shape=(4,),name="curr_state")
curr_action = keras.Input(shape=(2,),name="curr_action")
# FDM model
curr_state_action = concatenate([curr_state, curr_action])
fdm_h1 = Dense(16,name="dense1_FDM")(curr_state_action)
fdm_h1 = LeakyReLU(alpha=0.2,name="LeakyRelu1_FDM")(fdm_h1)
fdm_h2 = Dense(16,name="dense2_FDM")(fdm_h1)
fdm_h2 = LeakyReLU(alpha=0.2,name="LeakyRelu2_FDM")(fdm_h2)
fdm_pred_state = layers.Dense(4,name="dense3_FDM")(fdm_h2)


# This model maps an input to its next state
AE = keras.Model(inputs=AE_state, outputs=n_state,name="AE")


# This model maps an input & action to its next state
FDM = keras.Model(inputs=[curr_state,curr_action], outputs=fdm_pred_state,name="FDM")


#print(AE.summary())
#print(FDM.summary())


pause = 0 
while pause == 1:
    pause = 1

tf.keras.utils.plot_model(AE, to_file='AE_model_plot.png', show_shapes=True, show_layer_names=True)
tf.keras.utils.plot_model(FDM, to_file='FDM_model_plot.png', show_shapes=True, show_layer_names=True)

opt_AE = tf.keras.optimizers.RMSprop(learning_rate=0.00015)
AE.compile(loss='mean_squared_error', optimizer=opt_AE, metrics=['mse'])

opt_FDM = tf.keras.optimizers.RMSprop(learning_rate=0.00015)
FDM.compile(loss='mean_squared_error', optimizer=opt_FDM, metrics=['mse'])

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


if not os.path.exists("saved_model"):
    os.makedirs("saved_model")

reuse =True
if reuse ==True:
    print("Using pretrained models")
    AE = keras.models.load_model('saved_model/AE')
    FDM = keras.models.load_model('saved_model/FDM')
else:
    # training
    print("Training started")
    history_AE = AE.fit(x=s, y=ns,
                    epochs=100,
                    batch_size=32,
                    shuffle=True,
                    verbose=1,
                    validation_data=(test_s, test_ns))

    print("AE Training Done")
    history_FDM = FDM.fit(x=[s,a], y=ns,
                    epochs=100,
                    batch_size=32,
                    shuffle=True,
                    verbose=1,
                    validation_data=([test_s,test_a], test_ns))

    print("FDM Training Done")
    history_dict_AE = history_AE.history
    history_dict_FDM = history_FDM.history
    AE.save('saved_model/AE')
    FDM.save('saved_model/FDM')

print("Eval")
# evaluate the model
_, train_mse = AE.evaluate(s, ns, verbose=0)
_, test_mse = AE.evaluate(test_s, test_ns, verbose=0)
print('Train: %.6f, Test: %.6f' % (train_mse, test_mse))


_, train_mse = FDM.evaluate([s,a], ns, verbose=0)
_, test_mse = FDM.evaluate([test_s,test_a], test_ns, verbose=0)
print('Train: %.6f, Test: %.6f' % (train_mse, test_mse))

pause = 0 
while pause == 1:
    pause = 1
'''
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
'''
pause = 0 
while pause == 1:
    pause = 1


# Predict next state 
# Note that we take them from the *test* set
#pred_ns = AE.predict(test_s)
p_state = np.zeros((1,4))
left = np.zeros((1,2))
left[0][0] = 1
right = np.zeros((1,2))
right[0][1] = 1
repeat = True
while repeat:
    print("Enter index to test")
    m = int(input())

    for i in range(4):
        p_state[0][i] = test_s[m][i]
    
    pred_ns = AE.predict(p_state)

    FDM_ns_l = np.squeeze(FDM.predict([p_state,left]))
    FDM_ns_r = np.squeeze(FDM.predict([p_state,right]))

    FDM_ns_both = np.array([FDM_ns_l,FDM_ns_r])

    print("True Next state      ",test_ns[m])
    print("AE pred Nstate       ",pred_ns)

    cost = np.abs(FDM_ns_both-pred_ns)
    print("state diff           ",cost)
    p_cost =np.array([cost[0][0]+cost[0][2],cost[1][0]+cost[1][2]])
    cost = np.sum(cost,axis=1)
    print("cost                 ",cost)
    print("p_cost               ",p_cost)
    
    action_from_IDM = np.argmin(cost, axis=0)
    print("True action          ",test_a[m])
    print("action from IDM      ",action_from_IDM)
    action_from_IDM = np.argmin(p_cost, axis=0)
    print("p_action from IDM      ",action_from_IDM)
    

