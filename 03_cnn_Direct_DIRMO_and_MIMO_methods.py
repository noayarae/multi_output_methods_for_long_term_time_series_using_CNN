### Direct model
### Author: Efrain Noa-Yarasca
### Texas A&M University

import random, csv
import matplotlib.pyplot as plt
import numpy as np

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
from keras.utils.vis_utils import plot_model
from keras.layers import Bidirectional
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# -----------------------------------------------------------------------------
def colored(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def smape(actual, forecast):
    return 1/len(actual) * np.sum(2 * np.abs(forecast-actual) / (np.abs(actual) + np.abs(forecast))*100)

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

#train_x2, train_y2 = series_to_supervised2(train, n_input, n_out, tg) # tg: target
def series_to_supervised2(data, n_steps_in, n_steps_out, tg):   # tg: target
  X, y = list(), list()
  for i in range(len(data)):
    end_ix = i + n_steps_in
    out_str_ix = end_ix + n_steps_out *(tg-1)
    out_end_ix = end_ix + n_steps_out *tg
    if out_end_ix > len(data):
      break
    #print(end_ix, out_end_ix, end_ix+(tg-1), out_end_ix)
    seq_x = data[i:end_ix]
    seq_y = data[out_str_ix: out_end_ix]
    #if tg < 2 and  i < 2:   # To debug
        #print(i, end_ix, out_str_ix, out_end_ix)
    X.append(seq_x)
    y.append(seq_y)
  return array(X), array(y)

def reshape_data_cnn(train_x2, train_y2):
  train_y2 = train_y2.reshape((train_y2.shape[0], train_y2.shape[1]))
  print ("train_x2.shape (reshape_data_cnn) ---> ", train_x2.shape) #
  print ("train_y2.shape (reshape_data_cnn) ---> ", train_y2.shape)
  return train_x2, train_y2

# -----------------------------------------------------------------------------
def model_cnn (n_in, n_out, activ_m, n_nodes):
  model = Sequential()
  n_features = 1
  model.add(Conv1D(filters=64, kernel_size=3, strides=1, 
                   activation='relu', input_shape=(n_in, n_features), padding = 'valid'))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Flatten())

  model.add(Dense(100, activation='relu'))
  model.add(Dense(n_out))
  model.compile(loss='mse', optimizer = SGD(learning_rate=0.005), metrics=['mse'])
  #model.compile(loss='mse', optimizer= Adamax(learning_rate = 0.001), metrics=['mse']) # Adamax        metrics=['mse', 'mae', 'mape'])
  #model.compile(loss='mse', optimizer= Adamax(learning_rate = 0.01, beta_1=0.999, beta_2=0.990, epsilon=1e-07), metrics=['mse', 'mae', 'mape']) # Adamax
  #model.summary()
  return model

def model_cnn_lstm(n_sub_in, n_out, activ_m, n_nodes):
  model = Sequential()
  n_features = 1
  model.add(TimeDistributed(Conv1D(16, 3, strides=1, activation='relu', padding = 'same'), input_shape=(None, n_sub_in, n_features)))
  model.add(TimeDistributed(MaxPooling1D()))
  model.add(TimeDistributed(Flatten()))

  model.add(LSTM(50, activation='relu'))
  model.add(Dense(n_out))
  model.compile(loss='mse', optimizer= Adam(learning_rate = 0.001))
  return model

# -----------------------------------------------------------------------------
#series = read_csv('/content/drive/MyDrive/5000_data/003_340_ts_data_for_decomp_v1.csv', header=0, index_col=0)  # From local folder
series = read_csv('D:/work/research_t/forecast_dec/003_753ts_data_for_decomp.csv', header=0, index_col=0)
#site = 'S_369' # 'S_21642' # 'S_432' # 'S_21928' #  'S_21928' #'S_21581' # 'S_21661' #                 #  ----> SET
site = 'S_445' # 'S_370' # 'S_376' # 'S_416' # 'S_434' # 'S_445' #'S_21895' #
'''
dif1 = series[site]-series[site].shift(1)
dif1 = dif1.dropna()
dif2 = dif1-dif1.shift(24)
dif2 = dif2.dropna()
data = dif2.values
dd = data.reshape(471,1) # data.reshape(495,1) #'''

data = series[site].values
dd = data.reshape(496,1)

max_v = max(series[site].values)
min_v = min(series[site].values)

### Normalization ----------------------
from sklearn.preprocessing import MinMaxScaler
min_sc, max_sc = 0, 1
sc = MinMaxScaler(feature_range = (min_sc, max_sc))
data_sc = sc.fit_transform(dd) # Scaling
data = data_sc #
# --------------------------------------
data = data.ravel()
print (type(data), data.shape)
print () #'''

# -----------------------------------------------------------------------------

import time
start_time = time.perf_counter ()

list_inter = [1,2,3,4,6,8,12,24] # [24] #               # < ---------------------------- INPUTS
list_inter = [1,2]
all_rmse2, all_rmse2_inv = [],[]
all_mape2, all_mape2_inv = [],[]
all_smape2, all_smape2_inv = [],[]
pred2_inv_r_n3 = []
cc1 = 1
for iter_n in list_inter:
  ni = 24
  print(colored(153, 51, 255, (site + "  Case: ........... "+ str(iter_n))))

  n_test = 2                    # Horizon
  n_repeats = 20                # Number of repetitions
  scores_rmse2, scores_rmse2_inv = [],[]
  scores_mape2, scores_mape2_inv = [],[]
  scores_smape2, scores_smape2_inv = [],[]
  pred2_inv_r_n2 = []
  for i in range(n_repeats):
    print(colored(0, 255, 255, (site + '  n_repeat ..... '+ str(i+1)+"/"+ str(n_repeats)+
                              '    n_in...'+ str(cc1)+"/"+ str(len(list_inter))+' ('+str(ni)+')')))
    n_input = ni #
    n_out = iter_n # 6
    #e_iter = int(24/iter_n+0.1) # 4
    e_iter = int(n_test/iter_n+0.1) # 4
    
    n_features = 1
    #target = 1
    target_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    
    '''
    period_length = 24
    period_arr = np.array(list(range(1, period_length+1)))
    period_sets = period_arr.reshape(int(period_length/n_out), n_out)
    print(period_sets) # '''
    
    #target_list = [2]
    
    prediction2 = list()
    #for targ in period_sets:
    #for target in target_list:
    for ext in range(1,e_iter+1): # [8]: # 
      #target = targ[0]
      #print("Target....  ", ext)
      print(colored(255, 100, 155, ('Target ..... '+str(ext)+'/'+str(e_iter)+"   Model "+str(ext))))
      train, test = train_test_split(data, n_test)
      test_m = data[-(n_input+n_test):]

      # ------ Converting data into supervised
      train_x2, train_y2 = series_to_supervised2(train, n_input, n_out, ext)       # ---------> Call Fn
      test_x2, test_y2 = series_to_supervised2(test_m, n_input, n_out, ext)        # ---------> Call Fn
      ###print("train_x2: ", train_x2.shape," \n", train_x2)
      ###print("train_y2: ", train_y2.shape," \n", train_y2)
      #print("test_x2: ", test_x2.shape," \n", test_x2)
      ###print("test_y2: ", test_y2.shape," \n", test_y2)
      train_x2, train_y2 = reshape_data_cnn(train_x2, train_y2)
      print ("Shapes (train_x2, train_y2,test_x2, test_y2): >>>>> ", train_x2.shape, train_y2.shape, test_x2.shape, test_y2.shape)
      print ("--------------------------------------------------------------------------")  #

      # ------ Setting the model - define config
      activat_set = 'relu'
      n_nodes0 = 50 #

      ### ------------------------------------   For CNN model
      print(colored(255, 0, 0, ('CNN ....... ')))
      model = model_cnn(n_input, n_out, activat_set,  n_nodes0)

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
      n_batch = 16 #
      n_epochs = 500
      early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
      #early_stop = EarlyStopping(monitor='loss', patience=50, verbose=1, mode='auto')
      hist_model = model.fit(train_x2, train_y2, epochs=n_epochs, batch_size = n_batch, verbose=0, validation_data=(test_x2, test_y2), callbacks=[early_stop])   #  -----   For CNN
      #hist_model = model.fit(train_x3, train_y3, epochs=n_epochs, batch_size = n_batch, verbose=0, validation_data=(test_x3, test_y3), callbacks=[early_stop])  #  -----   For CNN-LSTM

      # summarize LEarning curve (history-accuracy)
      
      plt.plot(hist_model.history['mse'])
      plt.plot(hist_model.history['val_mse'])
      plt.title('Learning curve - model accuracy - '+str(ext))
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.yscale('log')
      plt.show() #'''

      ### Prediction
      #prediction2 = list()
      history2 = [x for x in train] #
      for j in range(1): # range(int(np.ceil(len(test)/n_out))):
        #print("j ............... ", j)
        x_input2 = array(history2[-n_input:]).reshape(1, n_input,1)            # For CNN
        #x_input2 = array(history2[-n_input:]).reshape(1, n_seq, n_sub_in,1)     # For CNN-LSTM
        print('x_input2...:', x_input2.tolist())
        print('x_input2...:', history2[-n_input:])

        yhat2 = model.predict(x_input2, verbose=0)      # prediction
        #print("Pred:", yhat2, "... Exp:", test_y2[0], "... err%:",100*(yhat2-test_y2[0])/test_y2[0])
        ppi = yhat2*(dd.max()-dd.min())+dd.min()
        yyi = test_y2[0]*(dd.max()-dd.min())+dd.min()
        print("Pred_inv:", ppi, "... Exp_inv:", yyi, "... err%:",100*(ppi-yyi)/yyi)
        #prediction2.append(yhat2[0])

        #history1 = history1 + test[j*n_out:(j+1)*n_out].tolist()
        #ee2 = yhat2[0].reshape(n_out) # Reshape the just predicted values
        #aa1 = np.r_[history2, ee2] # Add the just predicted values to the last 24 values from dataset
        #history2 = aa1[-n_input:] # Take the last 24 values to input into the next prediction 

      print("------------------------------------------------------------------")
      prediction2.append(yhat2[0])
    pred2_flat = np.concatenate(prediction2).ravel().tolist()
    #print(pred2_flat)
    #print(eeee)

    ### estimate prediction error
    rmse2 = sqrt(mean_squared_error(test, pred2_flat))
    mape2 = mean_absolute_percentage_error(test, pred2_flat)
    smape2 = smape(test, pred2_flat)

    scores_rmse2.append(rmse2)
    scores_mape2.append(mape2)
    scores_smape2.append(smape2)

    pred2_inv_flat = sc.inverse_transform(prediction2).reshape(n_test,1)
    test_invt = sc.inverse_transform(test.reshape(n_test,1)) # (6,2)
    
    ### &&&&&&&&&&&&&&&&   Plot predicted and observed. &&&&&&&&&&&&&&&&
    plt.figure(figsize=(20,4))
    plt.plot(np.append(train*(max_v-min_v)+min_v, test_invt)) # plt.plot(test_invt)
    plt.plot(np.append(train*(max_v-min_v)+min_v, pred2_inv_flat)) # plt.plot(pred2_inv_flat)
    plt.title('Predicted vs Explored - '+str(ext))
    plt.legend(['Expected+', 'Predicted+'], loc='upper left')
    plt.grid()
    plt.show() #'''
    

    rmse2_inv = sqrt(mean_squared_error(test_invt, pred2_inv_flat)) # Error using inversed-transformed data
    mape2_inv = mean_absolute_percentage_error(test_invt, pred2_inv_flat)
    smape2_inv = smape(test_invt, pred2_inv_flat.reshape(n_test,1))

    scores_mape2_inv.append(mape2_inv)
    scores_rmse2_inv.append(rmse2_inv)
    scores_smape2_inv.append(smape2_inv)

    print(' > RMSE2 %.3f'% rmse2, ' > RMSE2_inv %.3f'% rmse2_inv)
    print(' > MAPE2 %.3f'% mape2, ' > MAPE2_inv %.3f'% mape2_inv)
    print(' > SMAPE2 %.3f'% smape2, ' > SMAPE2_inv %.3f'% smape2_inv)

    pred2_inv_r_n1 = [item for sublist in pred2_inv_flat.tolist() for item in sublist]  # ***
    pred2_inv_r_n2.append(pred2_inv_r_n1)
    print("--------------------------------------------------------------------")
    print()

  # summarize scores_rmse1 (summarize model performance)
  #print('%s: %.3f SD (+/- %.3f)' % ('RMSE2', mean(scores_rmse2), std(scores_rmse2)))
  print('%s: %.3f SD (+/- %.3f)' % ('RMSE2_i', mean(scores_rmse2_inv), std(scores_rmse2_inv)))
  #print('%s: %.3f SD (+/- %.3f)' % ('MAPE2', mean(scores_mape2), std(scores_mape2)))
  print('%s: %.3f SD (+/- %.3f)' % ('MAPE2_i', mean(scores_mape2_inv), std(scores_mape2_inv)))
  #print('%s: %.3f SD (+/- %.3f)' % ('SMAPE2', mean(scores_smape2), std(scores_smape2)))
  print('%s: %.3f SD (+/- %.3f)' % ('SMAPE2_i', mean(scores_smape2_inv), std(scores_smape2_inv)))

  all_rmse2.append(scores_rmse2)
  all_rmse2_inv.append(scores_rmse2_inv)
  all_mape2.append(scores_mape2)
  all_mape2_inv.append(scores_mape2_inv)
  all_smape2.append(scores_smape2)
  all_smape2_inv.append(scores_smape2_inv)
  pred2_inv_r_n3.append(pred2_inv_r_n2)     # ***  recurrent

  cc1 += 1
  print ()

# --------------- Print running time ----------- #
end_time = time.perf_counter ()
print(end_time - start_time, "seconds")
# ------------- End Print running time ----------- #

# print all values
print(all_rmse2_inv)
print(all_mape2_inv)
print(all_smape2_inv)
print("--------------------------------------------------------------------")

'''
### Save as Exccel file.  Ref.: https://github.com/Sven-Bo/export-dataframes-to-excel/blob/master/df-to-excel.py
# Ref.: https://www.youtube.com/watch?v=DroafWQXqDw
# Ref.: https://www.youtube.com/watch?v=or2ibHcZkSY&t=50s

import pandas as pd
heads = [('sim'+str(i)) for i in range(1, n_repeats + 1)]
hor = 2 #6 # 12
divisors_h = [x for x in range(1,hor+1) if hor/x==int(hor/x)]
ids = [('k'+str(j)) for j in divisors_h]        # Get: ['k1', 'k2', 'k3', 'k6']

ds1 = pd.DataFrame(all_rmse2_inv, columns = heads, index=ids) # For RMSE
ds2 = pd.DataFrame(all_mape2_inv, columns = heads, index=ids)            # For MAPE
ds3 = pd.DataFrame(all_smape2_inv, columns = heads, index=ids)           # For SMAPE

time_h = [('t'+str(i)) for i in range(1, hor + 1)]
dout_1 = pd.DataFrame(pred2_inv_r_n3[0], columns = time_h, index=heads)
dout_2 = pd.DataFrame(pred2_inv_r_n3[1], columns = time_h, index=heads)
#dout_3 = pd.DataFrame(pred2_inv_r_n3[2], columns = time_h, index=heads)
#dout_4 = pd.DataFrame(pred2_inv_r_n3[3], columns = time_h, index=heads)
#dout_5 = pd.DataFrame(pred2_inv_r_n3[4], columns = time_h, index=heads)
#dout_6 = pd.DataFrame(pred2_inv_r_n3[5], columns = time_h, index=heads)

with pd.ExcelWriter("ts5_m3_h2_20s.xlsx", engine="openpyxl") as writer:
    sh1_name = 'rmse_mape_smape_h2'
    ds1.to_excel(writer, sheet_name = sh1_name)
    ds2.to_excel(writer, sheet_name = sh1_name, startcol=0, startrow=10, header=True, index=True)
    ds3.to_excel(writer, sheet_name = sh1_name, startcol=0, startrow=20, header=True, index=True)
    
    sh2_name = 'ts_pred_h2'
    dout_1.to_excel(writer, sheet_name = sh2_name)
    dout_2.to_excel(writer, sheet_name = sh2_name, startcol= 1*(hor+2), startrow=0, header=True, index=True)
    #dout_3.to_excel(writer, sheet_name = sh2_name, startcol=2*(hor+2), startrow=0, header=True, index=True)
    #dout_4.to_excel(writer, sheet_name = sh2_name, startcol=3*(hor+2), startrow=0, header=True, index=True)
    #dout_5.to_excel(writer, sheet_name = sh2_name, startcol=4*(hor+2), startrow=0, header=True, index=True)
    #dout_6.to_excel(writer, sheet_name = sh2_name, startcol=5*(hor+2), startrow=0, header=True, index=True)

#'''













