### Iterative model for 1-step and multi-step ahead forecasting
# Author: Efrain Noa-Yarasca
# Texas A&M University


import random, csv, math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandas import read_csv
from math import sqrt
from numpy import array, mean, std, median
from pandas import DataFrame
from pandas import concat

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping
from keras.layers import TimeDistributed          

from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

def colored(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def smape(actual, forecast):
    return 1/len(actual) * np.sum(2 * np.abs(forecast-actual) / (np.abs(actual) + np.abs(forecast))*100)

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

def series_to_supervised2(data, n_steps_in, n_steps_out):
  X, y = list(), list()
  for i in range(len(data)):
    end_ix = i + n_steps_in
    out_end_ix = end_ix + n_steps_out
    if out_end_ix > len(data):
      break
    seq_x, seq_y = data[i:end_ix], data[end_ix:out_end_ix]
    X.append(seq_x)
    y.append(seq_y)
  return array(X), array(y)

def reshape_data_cnn(train_x2, train_y2):
    train_y2 = train_y2.reshape((train_y2.shape[0], train_y2.shape[1])) 
    return train_x2, train_y2
# ----------------------------------------------------------

### Get denoised time series

#site = 'S_369' # 'S_369' # 'S_372' # 'S_391' # 'S_426' # 'S_21642' 
site ='S_445' # 'S_370' # 'S_376' # 'S_416' # 'S_434' # 'S_445' #'S_21895' #
dfgd = pd.read_csv('D:/work/research_t/forecast_dec/003_753ts_data_for_decomp.csv', header=0, index_col=0)
#dfgd = pd.read_csv('D:/work/research_t/forecast_dec/003_fourier_denoised_ts.csv', header=0, index_col=0)
#site = 'S_369'# 'S_357'# 'S_369'# 'S_372'# 'S_391'# 'S_426'# 'S_21642' # 'S_21928'#                 #  ----> SET
df_m = dfgd.copy() # make a copy
dfm_serie = df_m[site]

max_v = max(dfgd[site].values)
min_v = min(dfgd[site].values)
# -------------------- Time-series based on sine function --------------------


x_range = 496 # 496 # 6000# 496# 
x = list(range(0, x_range))
m1 = 750
a1 = 250
m2 = 750
a2 = 250

f1 = []
f2 = []
fs = []
for i in range(x_range):
  f1.append(m1 + a1*math.sin(2*math.pi*i/24))
  f2.append(m2 + a2*math.sin(2*math.pi*i/12))
  fs.append(m1 + a1*math.sin(2*math.pi*i/24) + m2 + a2*math.sin(2*math.pi*i/12))
'''
plt.figure(figsize=(20,3))
plt.plot(x, f1, color='green')
plt.plot(x, f2, color='darkblue')
plt.plot(x, fs, color='red')
plt.show() #'''

# ---------------------- Get Normal random values -------------------------
from random import seed
from random import gauss
#seed(1)
rnd_gs1 = []
for _ in range(x_range): # range(6000): # 
  rnd_gs1.append(gauss(0, 1))
f_rnd = 0
rnd_gs2 = [x * f_rnd for x in rnd_gs1]
#print(rnd_gs1) # Random numbers between 0 to 1
#print(rnd_gs2) # Random numbers multiplied by a factor

# -------- Plot the random numbers

'''x = np.array(list(range(0, x_range)))
y = np.array(rnd_gs2)
plt.figure(figsize=(4,3))
plt.scatter(x, y)
plt.show() #'''

# -------------------- Plot Sine-function + random numbers --------------------
sum_fn_rd = [f1[i] + rnd_gs2[i] for i in range(len(f1))]
print(f1[0:5])
print(rnd_gs2[0:5])
print(sum_fn_rd[0:5])
'''
plt.figure(figsize=(15,3))
plt.plot(x, sum_fn_rd,'--', color='blue')
plt.plot(x[:-24], f1[:-24], color='red') #'darkblue'
plt.title('Time series using Sin function')
plt.ylabel('Value')
plt.xlabel('Time')
plt.legend(['sum_fn_rd', 'f1'], loc='upper left')
#plt.xlim([300, 510])
plt.grid()
plt.show() #'''


#sum_fn_rd = ts_n
sum_fn_rd = dfm_serie.tolist()
#sum_fn_rd = ts_denoised

#x_range = 471
f11 = np.array(sum_fn_rd).reshape(x_range,1) # Convert to array and reshape. From (496,) to (496,1) 
print(f11.shape) 

# ------------------------------------------------------------------------
### Normalization
from sklearn.preprocessing import MinMaxScaler
min_sc, max_sc = 0, 1
sc_f11 = MinMaxScaler(feature_range = (min_sc, max_sc))
data_sc_f11 = sc_f11.fit_transform(f11) # Scaling
data_f11 = data_sc_f11 #
data_f11 = data_f11.ravel() # Make it plain array

# ------------------------------------------------------------------------
def model_cnn2 (n_in, n_out, activ_m, n_nodes, n_filter, k_size):
    model = Sequential()
    n_features = 1
    model.add(Conv1D(filters=n_filter, kernel_size=3, strides=1, 
                     activation='relu', input_shape=(n_in, n_features), 
                     padding = 'valid'))
    #model.add(Conv1D(filters=n_filter, kernel_size=k_size, strides=1, 
    #                 activation='relu', padding = 'valid')) # Causal padding. Ref.: https://analyticsindiamag.com/guide-to-different-padding-methods-for-cnn-models/
    model.add(MaxPooling1D(pool_size=2)) # Deafult=2
    #model.add(Conv1D(filters=n_filter, kernel_size=9, strides=1, activation='relu', input_shape=(n_in, n_features), padding = 'valid'))
    #model.add(Conv1D(filters=n_filter, kernel_size=9, strides=1, activation='relu', padding = 'valid')) # Causal padding. Ref.: https://analyticsindiamag.com/guide-to-different-padding-methods-for-cnn-models/
    #model.add(MaxPooling1D()) # Deafult=2
    model.add(Flatten())
    
    #model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_out))
    #model.compile(loss='mse', optimizer= Adamax(learning_rate = 0.01), metrics=['mse']) # Adamax        metrics=['mse', 'mae', 'mape'])
    #model.compile(loss='mse', optimizer= Adamax(learning_rate = 0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-07), metrics=['mse', 'mae', 'mape']) # Adamax
    
    #model.compile(loss='mse', optimizer = SGD(learning_rate=0.1), metrics=['mse']) # metrics=['mse', 'mae', 'mape'])
    model.compile(loss='mse', optimizer = SGD(learning_rate=0.005), metrics=['mse']) # with decay
    
    #model.compile(loss='mse', optimizer = SGD(learning_rate=0.025, momentum=0.999), metrics=['mse']) # SGD with Momentum. metrics=['mse', 'mae', 'mape'])
    #model.compile(loss='mse', optimizer = SGD(lr=0.01), metrics=['mse', 'mae', 'mape']) # SGD for Learning-Function function
    #model.summary()
    return model

# ------------------------------------------------------------------------
data = data_f11


from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(1)

import time
#import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
#from tensorflow.keras import backend as K
start_time = time.perf_counter () 

def learning_rate_1(epoch, lrate): # Ref.: https://neptune.ai/blog/how-to-choose-a-learning-rate-scheduler
  epochs = 1000
  decay = lrate/epochs
  new_lr = lrate * 1 / (1 + decay * epoch) # Eq. by default (Time-base dacay)
  return new_lr

def learning_rate_2(epoch, lrate):
  k = 0.1
  new_lr = lrate * math.exp(-k*epoch) # Exponential decay
  return new_lr

def learning_rate_3(epoch, lrate): # Step-based decay
  drop_rate = 0.95
  epochs_drop = 20.0
  new_lr = lrate * math.pow(drop_rate, math.floor(epoch/epochs_drop))
  return new_lr

# Define the Required Callback Function
import tensorflow as tf
from tensorflow.keras import backend as K
class printlearningrate(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    optimizer = self.model.optimizer
    lr = K.eval(optimizer.lr)
    Epoch_count = epoch + 1
    print('\n', "Epoch:", Epoch_count, ', LR: {:.5f}'.format(lr)) #'''

def scheduler(epoch):
  optimizer = model.optimizer
  return K.eval(optimizer.learning_rate + 0.0001) #'''


#list_n_input = [12,24,36,48,60,72,84,96,108,120,132,144,156,168,180,192,204,216,228,240,252,264,276,288,300,312,324,336,348,360,372,384,396,408,420,432,444, 456,468]                # < ---------- INPUTS
list_n_input = [48]
nf_list = [8, 16, 32, 64, 128, 256]
ks_list = [1,2,3,4,6,8,12,24] # [24] # 
ks_list = [1,2,3,6] # [24] # 

all_rmse2, all_rmse2_inv = [],[]
all_mape2, all_mape2_inv = [],[]
all_smape2, all_smape2_inv = [],[]

all_rmse2_inv_train, all_mape2_inv_train, all_smape2_inv_train = [],[],[]

pred2_inv_r_n3 = []
cc1 = 1
#for ni in list_n_input:
#for nf in nf_list:
for ks in ks_list:
  ni = 24
  print(colored(255, 105, 60, (site + "  N_INPUTs: ........... "+ str(ni)+'  n_out:'+str(ks)))) #Magenta

  n_test = 6                                      # <----------Horizon
  n_repeats = 2                                   # <-------------------------- number REPEAT         
  scores_rmse2, scores_rmse2_inv = [],[]
  scores_mape2, scores_mape2_inv = [],[]
  scores_smape2, scores_smape2_inv = [],[]
  rmse_tr_inv_list, mape_tr_inv_list, smape_tr_inv_list = [],[],[]
  pred2_inv_r_n2 = []
  for i in range(n_repeats):
    print(colored(0, 255, 255, (site + '  n_repeat ..... '+ str(i+1)+"/"+ str(n_repeats)+
                              '    n_in...'+ str(cc1)+"/"+ str(len(list_n_input))+' ('+str(ni)+')')))
    n_input = ni # 
    n_out = ks #24
    n_features = 1
    train, test = train_test_split(data, n_test)  
    test_m = data[-(n_input+n_test):]
    test_gr = []
    newlist = [test_gr.append(test[x:x+n_out]) for x in range(0, n_test, n_out)] # Create a list grouping items

    # ------ Converting data into supervised
    train_x2, train_y2 = series_to_supervised2(train, n_input, n_out)       # ---------> Call Fn
    test_x2, test_y2 = series_to_supervised2(test_m, n_input, n_out)        # ---------> Call Fn
    train_x2, train_y2 = reshape_data_cnn(train_x2, train_y2)
    print ("Shapes (train_x2, train_y2,test_x2, test_y2): >>>>> ", train_x2.shape, train_y2.shape, test_x2.shape, test_y2.shape) 
    #print ("-------------------------------------------------------------------------------")  # 

    # ------ Setting the model - define config
    activat_set = 'relu'
    n_nodes0 = 50 # 

    ### ------------------------------------   For CNN model
    print(colored(255, 0, 0, ('CNN ....... ')))
    n_filter = 64 #nf #64  # +++++++++++++++++++++++++++++++++++++++++++++++
    k_size = 3
    model = model_cnn2(n_input, n_out, activat_set,  n_nodes0, n_filter, k_size) 
    
    ### ------------------------------------   For CNN-LSTM model
    '''
    print(colored(255, 0, 0, ('CNN-LSTM ....... ')))
    n_seq = 2                                                                       # ------------> NEW  SETTINGS
    n_sub_in = 12
    model = model_cnn_lstm(n_sub_in, n_out, activat_set,  n_nodes0) 
    train_x3 = train_x2.reshape(train_x2.shape[0], n_seq, n_sub_in, n_features)
    train_y3 = train_y2.ravel()
    test_x3 = test_x2.reshape(test_x2.shape[0], n_seq, n_sub_in, n_features)
    test_y3 = test_y2.ravel()
    print("------------- shapes for CNN-LSTM: ", train_x3.shape, train_y3.shape, test_x3.shape, test_y3.shape) # '''


    # ------ Fit the model
    n_batch = 16 #  Ref.: https://pub.towardsai.net/what-is-the-effect-of-batch-size-on-model-learning-196414284add
    n_epochs = 500 
    early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
    #(deprecated-no used) rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=1E-7) # Reduce learning rate when a metric has stopped improving
    #updatelr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    #lrs = LearningRateScheduler(learning_rate_3) # Adjust the l-rate during training by reducing the l-rate according to a pre-defined schedule
    #printlr = printlearningrate() # To print l-rate. Ref.: https://stackoverflow.com/questions/59680252/add-learning-rate-to-history-object-of-fit-generator-with-tensorflow
    

    #hist_model = model.fit(train_x2, train_y2, epochs=n_epochs, batch_size = n_batch, verbose=0, validation_data=(test_x2, test_y2), callbacks=[early_stop])   #  -----   For CNN
    
    hist_model = model.fit(train_x2, train_y2, 
                           epochs=n_epochs, batch_size = n_batch, verbose=0, 
                           validation_data=(test_x2, test_y2), 
                           callbacks=[early_stop])
    # callbacks=[early_stop, rlrp, lrs, printlr, updatelr])

    #hist_model = model.fit(train_x3, train_y3, epochs=n_epochs, batch_size = n_batch, verbose=0, validation_data=(test_x3, test_y3), callbacks=[early_stop])  #  -----   For CNN-LSTM
    

    # list all data in history
    print(hist_model.history.keys())
    
    # summarize history for accuracy
    plt.plot(hist_model.history['mse'])
    plt.plot(hist_model.history['val_mse'])
    plt.title('Learning curve - model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.yscale('log')
    plt.show()#

    # summarize history for loss
    '''
    plt.plot(hist_model.history['loss'])
    plt.plot(hist_model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show() #'''

    ### Prediction on test period
    prediction2 = list()
    history2 = [x for x in train] # 
    for j in range(int(np.ceil(len(test)/n_out))):  ## Iterates through the test data
      x_input2 = array(history2[-n_input:]).reshape(1, n_input,1)            # For CNN.
      #print("x_input2: ... ", x_input2[0])
      #print("x_input2: ... ", history2[-n_input:])
      #x_input2 = array(history2[-n_input:]).reshape(1, n_seq, n_sub_in,1)     # For CNN-LSTM

      yhat2 = model.predict(x_input2, verbose=0)      # prediction
      #print(str(j+1)+ "/"+str(int(np.ceil(len(test)/n_out)))+ "...... yhat2:"+str(yhat2)+ '   expected:'+str(test_gr[j]))
      prediction2.append(yhat2[0])

      ee2 = yhat2[0].reshape(n_out) # Reshape the just predicted values
      aa1 = np.r_[history2, ee2] # Add the just predicted values to the last 24 values from dataset
      history2 = aa1[-n_input:] # Take the last 24 values to input into the next prediction
    
    ### -------------------  Start: Prediction on Training data ----------------- #
    print('Prediction in training data:')
    '''
    yhat_tr_all = []
    for k in range(0,len(train)-ni, n_out):
      input_tr = array(train[k:k+ni]).reshape(1, n_input,1)            # For CNN
      #print(input_tr)
      yhat_tr = model.predict(input_tr, verbose=0)      # prediction
      yhat_tr_all.append(yhat_tr[0])
      #if k<4:
      #  print(yhat_tr, '   expected:', train[k+24:k+24+1]) #

    pred_tr_flat = np.concatenate(yhat_tr_all).ravel().tolist()
    #print(pred_tr_flat)
    pred_tr_inv_flat = sc_f11.inverse_transform(yhat_tr_all).reshape(len(train)-ni,1) # reshape(448,1)
    train_invt = sc_f11.inverse_transform(train.reshape(len(train),1)) 
    rmse_tr_inv = sqrt(mean_squared_error(train_invt[ni:], pred_tr_inv_flat)) # Error using inversed-transformed data 
    mape_tr_inv = mean_absolute_percentage_error(train_invt[ni:], pred_tr_inv_flat)
    smape_tr_inv = smape(train_invt[ni:], pred_tr_inv_flat.reshape(len(train)-ni,1))
    rmse_tr_inv_list.append(rmse_tr_inv)
    mape_tr_inv_list.append(mape_tr_inv)
    smape_tr_inv_list.append(smape_tr_inv)
    print("Train-pred: ", "RMSE_tr:", rmse_tr_inv, " MAPE_tr:", mape_tr_inv, "SMAPE_tr:",smape_tr_inv) #'''
    ### ----------- End: Complement for prediction in Train --------- ##'''

    print("-------------------------------------------------------------------")
    pred2_flat = np.concatenate(prediction2).ravel().tolist()
    #print(pred2_flat)

    # estimate prediction error
    rmse2 = sqrt(mean_squared_error(test, pred2_flat)) 
    mape2 = mean_absolute_percentage_error(test, pred2_flat)
    smape2 = smape(test, pred2_flat) 

    scores_rmse2.append(rmse2)  
    scores_mape2.append(mape2) 
    scores_smape2.append(smape2) 

    pred2_inv_flat = sc_f11.inverse_transform(prediction2).reshape(n_test,1) 
    test_invt = sc_f11.inverse_transform(test.reshape(n_test,1)) 

    rmse2_inv = sqrt(mean_squared_error(test_invt, pred2_inv_flat)) # Error using inversed-transformed data 
    mape2_inv = mean_absolute_percentage_error(test_invt, pred2_inv_flat)
    smape2_inv = smape(test_invt, pred2_inv_flat.reshape(n_test,1))

    scores_mape2_inv.append(mape2_inv)
    scores_rmse2_inv.append(rmse2_inv)
    scores_smape2_inv.append(smape2_inv)
    
    ### ----------------   Plot predicted and observed.
    '''
    plt.figure(figsize=(20,4))
    #plt.plot(test_invt)
    #plt.plot(pred2_inv_flat)
    #plt.plot(np.append(train*(max_v-min_v)+min_v, test_invt*(max_v-min_v)+min_v)) # plt.plot(test_invt)
    #plt.plot(np.append(train*(max_v-min_v)+min_v, pred2_inv_flat*(max_v-min_v)+min_v)) # plt.plot(pred2_inv_flat)
    plt.plot(np.append(train*(max_v-min_v)+min_v, test_invt)) # plt.plot(test_invt)
    plt.plot(np.append(train*(max_v-min_v)+min_v, pred2_inv_flat), linestyle='--', color='red') # plt.plot(pred2_inv_flat)
    plt.title('Time series '+str(site)+' - '+str(i+1)+"/"+ str(n_repeats))
    plt.legend(['Expected+', 'Predicted+'], loc='upper left')
    plt.show() #'''


    print('> RMSE2 %.5f'% rmse2, ' > RMSE2_inv %.5f'% rmse2_inv)
    print('> MAPE2 %.3f'% mape2, ' > MAPE2_inv %.5f'% mape2_inv)
    print('> SMAPE2 %.5f'% smape2, ' > SMAPE2_inv %.5f'% smape2_inv)

    pred2_inv_r_n1 = [item for sublist in pred2_inv_flat.tolist() for item in sublist]  # ***
    pred2_inv_r_n2.append(pred2_inv_r_n1) 
    print("-------------------------------------------------------------------")
    print()
  
  # summarize scores_rmse1 (summarize model performance)    
  #print('%s: %.5f SD (+/- %.5f)' % ('RMSE2', mean(scores_rmse2), std(scores_rmse2)))
  #print('%s: %.3f SD (+/- %.3f)' % ('MAPE2', mean(scores_mape2), std(scores_mape2)))
  #print('%s: %.5f SD (+/- %.5f)' % ('SMAPE2', mean(scores_smape2), std(scores_smape2)))
  
  print('%s: %.5f SD (+/- %.5f)' % ('RMSE2_i', mean(scores_rmse2_inv), std(scores_rmse2_inv)))
  print('%s: %.5f SD (+/- %.5f)' % ('MAPE2_i', mean(scores_mape2_inv), std(scores_mape2_inv)))
  print('%s: %.5f SD (+/- %.5f)' % ('SMAPE2_i', mean(scores_smape2_inv), std(scores_smape2_inv)))
  
  ### Append for train data
  all_rmse2_inv_train.append(rmse_tr_inv_list) # ***
  all_mape2_inv_train.append(mape_tr_inv_list)
  all_smape2_inv_train.append(smape_tr_inv_list)
  
  ### Append for test data
  #all_rmse2.append(scores_rmse2)
  all_rmse2_inv.append(scores_rmse2_inv) # ***
  #all_mape2.append(scores_mape2)
  all_mape2_inv.append(scores_mape2_inv)
  #all_smape2.append(scores_smape2)
  all_smape2_inv.append(scores_smape2_inv)
  pred2_inv_r_n3.append(pred2_inv_r_n2)     # ***  recurrent   

  cc1 += 1
  end_time = time.perf_counter ()
  print("------> Partial Time: ", end_time - start_time, "seconds")
  print ()

print("Summary:")
#print(mean(rmse_tr_inv_list), mean(mape_tr_inv_list), mean(smape_tr_inv_list))
print(mean(scores_rmse2_inv), mean(scores_mape2_inv), mean(scores_smape2_inv))
# --------------- Print running time ----------- #
end_time = time.perf_counter ()
print("------> Running Time: ", end_time - start_time, "seconds")
print()
# ------------- End Print running time ----------- #


print("--------------------------------------------------------------------")
# ------------------------------------------------------------------------
print("All Summary ......................")
print(all_rmse2_inv)
print(all_mape2_inv)
print(all_smape2_inv) #'''
##### print(pred2_inv_r_n3)     # ***  recurrent  


print("Done........")


# ------------------------------------------------------------------------
### Plot predicted and observed
'''
plt.plot(test_invt)
plt.plot(pred2_inv_flat)
plt.legend(['Expected', 'Predicted'], loc='upper left')
plt.show() #'''
# ------------------------------------------------------------------------

'''
plt.figure(figsize=(20,3))
plt.plot(x, ts_n, color='green')
#plt.plot(x, f1, color='red') #'darkblue'
plt.show()'''

### Print the model structure
'''
model = Sequential()
n_features = 1
model.add(Conv1D(filters=n_filter, kernel_size=9, strides=1, 
                 activation='relu', input_shape=(24, n_features), 
                 padding = 'valid'))
#model.add(Conv1D(filters=n_filter, kernel_size=9, strides=1, 
#                 activation='relu', padding = 'valid')) # Causal padding. Ref.: https://analyticsindiamag.com/guide-to-different-padding-methods-for-cnn-models/
model.add(MaxPooling1D()) # Deafult=2
#model.add(Conv1D(filters=n_filter, kernel_size=9, strides=1, activation='relu', input_shape=(n_in, n_features), padding = 'valid'))
#model.add(Conv1D(filters=n_filter, kernel_size=9, strides=1, activation='relu', padding = 'valid')) # Causal padding. Ref.: https://analyticsindiamag.com/guide-to-different-padding-methods-for-cnn-models/
#model.add(MaxPooling1D()) # Deafult=2
model.add(Flatten())

model.add(Dense(50, activation='relu'))
model.add(Dense(n_out))
#model.compile(loss='mse', optimizer= Adamax(learning_rate = 0.01), metrics=['mse']) # Adamax        metrics=['mse', 'mae', 'mape'])
#model.compile(loss='mse', optimizer= Adamax(learning_rate = 0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07), metrics=['mse', 'mae', 'mape']) # Adamax

#model.compile(loss='mse', optimizer = SGD(learning_rate=0.1), metrics=['mse']) # metrics=['mse', 'mae', 'mape'])
model.compile(loss='mse', optimizer = SGD(learning_rate=0.025), metrics=['mse']) # with decay

#model.compile(loss='mse', optimizer = SGD(learning_rate=0.01, momentum=0.995), metrics=['mse']) # SGD with Momentum. metrics=['mse', 'mae', 'mape'])
#model.compile(loss='mse', optimizer = SGD(lr=0.01), metrics=['mse', 'mae', 'mape']) # SGD for Learning-Function function
model.summary()
'''


'''
### Save as Exccel file.  Ref.: https://github.com/Sven-Bo/export-dataframes-to-excel/blob/master/df-to-excel.py
# Ref.: https://www.youtube.com/watch?v=DroafWQXqDw
# Ref.: https://www.youtube.com/watch?v=or2ibHcZkSY&t=50s

import pandas as pd
heads = [('sim'+str(i)) for i in range(1, n_repeats + 1)]
hor = 6 #6 # 12
divisors_h = [x for x in range(1,hor+1) if hor/x==int(hor/x)]
ids = [('k'+str(j)) for j in divisors_h]        # Get: ['k1', 'k2', 'k3', 'k6']

ds1 = pd.DataFrame(all_rmse2_inv, columns = heads, index=ids) # For RMSE
ds2 = pd.DataFrame(all_mape2_inv, columns = heads, index=ids)            # For MAPE
ds3 = pd.DataFrame(all_smape2_inv, columns = heads, index=ids)           # For SMAPE

time_h = [('t'+str(i)) for i in range(1, hor + 1)]
dout_1 = pd.DataFrame(pred2_inv_r_n3[0], columns = time_h, index=heads)
dout_2 = pd.DataFrame(pred2_inv_r_n3[1], columns = time_h, index=heads)
dout_3 = pd.DataFrame(pred2_inv_r_n3[2], columns = time_h, index=heads)
dout_4 = pd.DataFrame(pred2_inv_r_n3[3], columns = time_h, index=heads)
#dout_5 = pd.DataFrame(pred2_inv_r_n3[4], columns = time_h, index=heads)
#dout_6 = pd.DataFrame(pred2_inv_r_n3[5], columns = time_h, index=heads)

with pd.ExcelWriter("ts5_m1_h6_20sim.xlsx", engine="openpyxl") as writer:
    sh1_name = 'rmse_mape_smape_h6'
    pd.DataFrame(['RMSE']).to_excel(writer, sheet_name = sh1_name, header=False, index=False)
    ds1.to_excel(writer, sheet_name = sh1_name, startcol=0, startrow=1, header=True, index=True)
    pd.DataFrame(['MAPE']).to_excel(writer, sheet_name = sh1_name, startcol=0, startrow=10, header=False, index=False)
    ds2.to_excel(writer, sheet_name = sh1_name, startcol=0, startrow=11, header=True, index=True)
    pd.DataFrame(['SMAPE']).to_excel(writer, sheet_name = sh1_name, startcol=0, startrow=20, header=False, index=False)
    ds3.to_excel(writer, sheet_name = sh1_name, startcol=0, startrow=21, header=True, index=True)
    
    sh2_name = 'ts_pred_h6'
    dout_1.to_excel(writer, sheet_name = sh2_name)
    dout_2.to_excel(writer, sheet_name = sh2_name, startcol= 1*(hor+2), startrow=0, header=True, index=True)
    dout_3.to_excel(writer, sheet_name = sh2_name, startcol=2*(hor+2), startrow=0, header=True, index=True)
    dout_4.to_excel(writer, sheet_name = sh2_name, startcol=3*(hor+2), startrow=0, header=True, index=True)
    #dout_5.to_excel(writer, sheet_name = sh2_name, startcol=4*(hor+2), startrow=0, header=True, index=True)
    #dout_6.to_excel(writer, sheet_name = sh2_name, startcol=5*(hor+2), startrow=0, header=True, index=True)

#'''








