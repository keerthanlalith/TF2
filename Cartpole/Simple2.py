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


print(AE.summary())
print(FDM.summary())


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
pred_ns = AE.predict(test_s)
dummystate = np.zeros((1,4))
left = np.zeros((1,2))
left[0][0] = 1
right = np.zeros((1,2))
right[0][1] = 1

for i in range(4):
    dummystate[0][i] = test_s[0][i]

pred_ns_l = np.squeeze(FDM.predict([dummystate,left]))
pred_ns_r = np.squeeze(FDM.predict([dummystate,right]))

pred_ns_1 = np.array([pred_ns_l,pred_ns_r])

print(test_ns[0])
print("action",test_a[0])
cost = np.abs(pred_ns_1-test_ns[0])
print(cost)
cost = np.sum(cost,axis=1)
print(cost)

print(np.argmin(cost, axis=0))


pause = 1 
while pause == 1:
    pause = 1

'''
left = np.zeros((len(test_s),2))
right = np.zeros((len(test_s),2))

for i in range(len(test_s)):
    left = 0


all_actions = np.eye(2)
rigth = all_actions[0]
left = all_actions[1]
testing = test_s[0]
print(rigth,left, test_s[0],test_a[0])
#FDM_Nstates1 = FDM.predict([testing,left])
#FDM_Nstates2 = FDM.predict([test_s,right])

FDM_Nstates1 =AE.predict(test_s)

print(FDM_Nstates1)
'''

pause = 1
while pause ==1:
    pause =1


d_err = np.abs(test_ns - pred_ns)
#pred_ns = denormalise (pred_ns)
d_err_1 = np.abs(test_ns - pred_ns)
p_err =np.mean(d_err*d_err,axis=0)
p_err2 =np.mean(d_err_1*d_err_1,axis=0)
    
print(p_err,p_err2)

n=len(test_s)
n = int(input('Enter your number of test data to predict:'))

print ("AE diff output, Normalised Test diff,  Difference")
    
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


    