### Recursive method for 1-step and multi-step ahead forecasting
# When k = H, the method becomes MIMO
# Author: Efrain Noa-Yarasca
# Texas A&M University


import random, csv, math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from math import sqrt
from numpy import array, mean, std, median

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping

from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from numpy.random import seed
from tensorflow import random

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

### Get Time series
site ='S_445' # 'S_370' # 'S_376' # 'S_21885' # 'S_434' # 'S_445' #'S_21895' #
dfgd = pd.read_csv('D:/work/research_t/forecast_dec/003_753ts_data_for_decomp.csv', header=0, index_col=0)
df_m = dfgd.copy() # make a copy
dfm_serie = df_m[site]

max_v = max(dfgd[site].values)
min_v = min(dfgd[site].values)
x_range = 496 # 496 # 6000# 496# 

sum_fn_rd = dfm_serie.tolist()
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
    model.add(MaxPooling1D(pool_size = 1)) # Deafult=2
    model.add(Flatten())
    
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_out))
    model.compile(loss='mse', optimizer = SGD(learning_rate=0.005), metrics=['mse']) # with decay
    return model
# ------------------------------------------------------------------------
data = data_f11

seed(1)
random.set_seed(1)

import time
start_time = time.perf_counter () 

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

    ###  For CNN model
    print(colored(255, 0, 0, ('CNN ....... ')))
    n_filter = 64 #nf #64  # +++++++++++++++++++++++++++++++++++++++++++++++
    k_size = 3
    model = model_cnn2(n_input, n_out, activat_set,  n_nodes0, n_filter, k_size) 
    
    # ------ Fit the model
    n_batch = 16 #  Ref.: https://pub.towardsai.net/what-is-the-effect-of-batch-size-on-model-learning-196414284add
    n_epochs = 500 
    early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
    hist_model = model.fit(train_x2, train_y2, 
                           epochs=n_epochs, batch_size = n_batch, verbose=0, 
                           validation_data=(test_x2, test_y2), 
                           callbacks=[early_stop])
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

    ### Prediction on test period
    prediction2 = list()
    history2 = [x for x in train] # 
    for j in range(int(np.ceil(len(test)/n_out))):  ## Iterates through the test data
      x_input2 = array(history2[-n_input:]).reshape(1, n_input,1)            # For CNN.
      yhat2 = model.predict(x_input2, verbose=0)      # prediction
      prediction2.append(yhat2[0])

      ee2 = yhat2[0].reshape(n_out) # Reshape the just predicted values
      aa1 = np.r_[history2, ee2] # Add the just predicted values to the last 24 values from dataset
      history2 = aa1[-n_input:] # Take the last 24 values to input into the next prediction
    
    print("-------------------------------------------------------------------")
    pred2_flat = np.concatenate(prediction2).ravel().tolist()

    # Estimate prediction error
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
    
    print('> RMSE2 %.5f'% rmse2, ' > RMSE2_inv %.5f'% rmse2_inv)
    print('> MAPE2 %.3f'% mape2, ' > MAPE2_inv %.5f'% mape2_inv)
    print('> SMAPE2 %.5f'% smape2, ' > SMAPE2_inv %.5f'% smape2_inv)

    pred2_inv_r_n1 = [item for sublist in pred2_inv_flat.tolist() for item in sublist]  # ***
    pred2_inv_r_n2.append(pred2_inv_r_n1) 
    print("-------------------------------------------------------------------")
    print()
  
  print('%s: %.5f SD (+/- %.5f)' % ('RMSE2_i', mean(scores_rmse2_inv), std(scores_rmse2_inv)))
  print('%s: %.5f SD (+/- %.5f)' % ('MAPE2_i', mean(scores_mape2_inv), std(scores_mape2_inv)))
  print('%s: %.5f SD (+/- %.5f)' % ('SMAPE2_i', mean(scores_smape2_inv), std(scores_smape2_inv)))
  
  ### Append for train data
  all_rmse2_inv_train.append(rmse_tr_inv_list) # ***
  all_mape2_inv_train.append(mape_tr_inv_list)
  all_smape2_inv_train.append(smape_tr_inv_list)
  
  ### Append for test data
  all_rmse2_inv.append(scores_rmse2_inv) # ***
  all_mape2_inv.append(scores_mape2_inv)
  all_smape2_inv.append(scores_smape2_inv)
  pred2_inv_r_n3.append(pred2_inv_r_n2)     # ***  recurrent   

  cc1 += 1
  end_time = time.perf_counter ()
  print("------> Partial Time: ", end_time - start_time, "seconds")
  print ()

print("Summary:")
print(mean(scores_rmse2_inv), mean(scores_mape2_inv), mean(scores_smape2_inv))
# --------------- Print running time ----------- #
end_time = time.perf_counter ()
print("------> Running Time: ", end_time - start_time, "seconds")
print()
# ------------- End Print running time ----------- #

print("All Summary ......................")
print(all_rmse2_inv)
print(all_mape2_inv)
print(all_smape2_inv) #'''

print("Done........")


