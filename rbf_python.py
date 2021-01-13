# -*- coding: utf-8 -*-

"""
Created on Thu Oct 17 18:33:06 2019

@author: steve
"""
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# rbf layer from https://github.com/PetraVidnerova/rbf_keras

from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Constant
import numpy as np


class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.
    # Example
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(
                                         value=self.init_betas),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp(-self.betas * K.sum(H**2, axis=1))

        # C = self.centers[np.newaxis, :, :]
        # X = x[:, np.newaxis, :]

        # diffnorm = K.sum((C-X)**2, axis=-1)
        # ret = K.exp( - self.betas * diffnorm)
        # return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
  
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#read input & output data
import pandas as pd #與Panel、DataFrame 與 Series 相呼應
import numpy as np #

#read input & output data
x_train=open("D:/important/TA/ANN/109學期/hw2/hw2_data/pm25/PM25/train_X.txt")
x_train=pd.read_csv(x_train, sep="\t",header=0)
y_train=open("D:/important/TA/ANN/109學期/hw2/hw2_data/pm25/PM25/train_y.txt")
y_train=(pd.read_csv(y_train, sep="\t",header=0)).iloc[:,1]

x_test=open("D:/important/TA/ANN/109學期/hw2/hw2_data/pm25/PM25/test_X.txt")
x_test=pd.read_csv(x_test, sep="\t",header=0)
y_test=open("D:/important/TA/ANN/109學期/hw2/hw2_data/pm25/test_Y_PM25_real.txt")
y_test=(pd.read_csv(y_test, sep="\t",header=0)).iloc[:,1]

#_______________________________________________________________________________________________________________
#using input label
usingLabel=[
     't1-WIND_DIREC','t1-CH4','t1-NMHC','t1-CO','t1-RH','t1-WIND_SPEED','t1-WD_HR','t1-SO2','t1-THC','t1-NO','t1-WS_HR','t1-AMB_TEMP','t1-PM10','t1-O3','t1-PM2.5','t1-NOx','t1-NO2','t1-RAINFALL',
     't2-WIND_DIREC','t2-CH4','t2-NMHC','t2-CO','t2-RH','t2-WIND_SPEED','t2-WD_HR','t2-SO2','t2-THC','t2-NO','t2-WS_HR','t2-AMB_TEMP','t2-PM10','t2-O3','t2-PM2.5','t2-NOx','t2-NO2','t2-RAINFALL',
     't3-WIND_DIREC','t3-CH4','t3-NMHC','t3-CO','t3-RH','t3-WIND_SPEED','t3-WD_HR','t3-SO2','t3-THC','t3-NO','t3-WS_HR','t3-AMB_TEMP','t3-PM10','t3-O3','t3-PM2.5','t3-NOx','t3-NO2','t3-RAINFALL',
     't4-WIND_DIREC','t4-CH4','t4-NMHC','t4-CO','t4-RH','t4-WIND_SPEED','t4-WD_HR','t4-SO2','t4-THC','t4-NO','t4-WS_HR','t4-AMB_TEMP','t4-PM10','t4-O3','t4-PM2.5','t4-NOx','t4-NO2','t4-RAINFALL',
     't5-WIND_DIREC','t5-CH4','t5-NMHC','t5-CO','t5-RH','t5-WIND_SPEED','t5-WD_HR','t5-SO2','t5-THC','t5-NO','t5-WS_HR','t5-AMB_TEMP','t5-PM10','t5-O3','t5-PM2.5','t5-NOx','t5-NO2','t5-RAINFALL',
     't6-WIND_DIREC','t6-CH4','t6-NMHC','t6-CO','t6-RH','t6-WIND_SPEED','t6-WD_HR','t6-SO2','t6-THC','t6-NO','t6-WS_HR','t6-AMB_TEMP','t6-PM10','t6-O3','t6-PM2.5','t6-NOx','t6-NO2','t6-RAINFALL',
     't7-WIND_DIREC','t7-CH4','t7-NMHC','t7-CO','t7-RH','t7-WIND_SPEED','t7-WD_HR','t7-SO2','t7-THC','t7-NO','t7-WS_HR','t7-AMB_TEMP','t7-PM10','t7-O3','t7-PM2.5','t7-NOx','t7-NO2','t7-RAINFALL',
     't8-WIND_DIREC','t8-CH4','t8-NMHC','t8-CO','t8-RH','t8-WIND_SPEED','t8-WD_HR','t8-SO2','t8-THC','t8-NO','t8-WS_HR','t8-AMB_TEMP','t8-PM10','t8-O3','t8-PM2.5','t8-NOx','t8-NO2','t8-RAINFALL',
     't9-WIND_DIREC','t9-CH4','t9-NMHC','t9-CO','t9-RH','t9-WIND_SPEED','t9-WD_HR','t9-SO2','t9-THC','t9-NO','t9-WS_HR','t9-AMB_TEMP','t9-PM10','t9-O3','t9-PM2.5','t9-NOx','t9-NO2','t9-RAINFALL'
     ];

data_in=x_train.loc[:,usingLabel]
data_real_out=y_train
test_in=x_test.loc[:,usingLabel]
test_real_out=y_test
#_______________________________________________________________________________________________________________
#data normalization
data_in_max=data_in.max()
data_in_min=data_in.min()
data_in_norm=(data_in-data_in_min)/(data_in_max-data_in_min)
data_in_norm=np.asarray(data_in_norm)

data_real_out_max=data_real_out.max()
data_real_out_min=data_real_out.min()
data_real_out_norm=(data_real_out-data_real_out_min)/(data_real_out_max-data_real_out_min)
data_real_out_norm=np.asarray(data_real_out_norm)

test_in_max=test_in.max()
test_in_min=test_in.min()
test_in_norm=(test_in-test_in_min)/(test_in_max-test_in_min)
test_in_norm=np.asarray(test_in_norm)

test_real_out_max=test_real_out.max()
test_real_out_min=test_real_out.min()
test_real_out_norm=(test_real_out-test_real_out_min)/(test_real_out_max-test_real_out_min)
test_real_out_norm=np.asarray(test_real_out_norm)


#_______________________________________________________________________________________________________________
#build model
#from keras import backend as K
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers.core import Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
#from keras.layers.normalization import BatchNormalization    
model = Sequential()
model.add(RBFLayer(10,initializer=InitCentersRandom(data_in_norm),betas=1.0,input_shape=(len(data_in_norm[0]),)))
model.add(Activation('tanh'))
model.add(Dense(1))
learning_rate=1e-4 #設定學習速率
adam = Adam(lr=learning_rate) #設定的優化器
model.compile(optimizer=adam,loss="mse") #執行設定好的優化器和誤差函數
earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=0) #設定early stopping的參數 
checkpoint =ModelCheckpoint("D:/important/TA/ANN/109學期/hw2/hw2_data/pm25/PM25/rbf-model.hdf5",save_best_only=True) 
#每次迭代儲存模式，我設定了最終只儲存最佳的模式(當驗證誤差最小的時候)
callback_list=[earlystopper,checkpoint]  
model.fit(data_in_norm,data_real_out_norm,epochs=100, batch_size=8,validation_split=0.2,callbacks=callback_list) 
#執行模式訓練并且把設定好的early stopping以及儲存模式參數套用到訓練中
#more details reference: https://keras.io/
#_______________________________________________________________________________________________________________
#model forecasting result
model.summary()
model=load_model("D:/important/TA/ANN/109學期/hw2/hw2_data/pm25/PM25/rbf-model.hdf5", custom_objects={'RBFLayer': RBFLayer}) #把儲存好的最佳模式讀入

batch_size=8
pred_train=(model.predict(data_in_norm,batch_size=batch_size))*(data_real_out_max-data_real_out_min)+data_real_out_min #針對訓練資料做預測
pred_test=(model.predict(test_in_norm,batch_size=batch_size))*(test_real_out_max-test_real_out_min)+test_real_out_min #針對測試資料做預測
#_______________________________________________________________________________________________________________
#define error function & calculate model's output error
def np_RMSE(output,output_pred):
    rmse=0
    for i in range(0,len(output)):
        rmse=rmse+np.square(output[i]-output_pred[i])
    rmse=np.sqrt(rmse/len(output))
    return rmse

def np_R2(output,output_pred):
    return np.square(np.corrcoef(output.reshape(np.size(output)),output_pred.reshape(np.size(output_pred)), False)[1,0])

rmse=[];r2=[];
rmse.append(np_RMSE(np.asarray(data_real_out),pred_train))
rmse.append(np_RMSE(np.asarray(test_real_out),pred_test))
r2.append(np_R2(np.asarray(data_real_out),pred_train))
r2.append(np_R2(np.asarray(test_real_out),pred_test))

rmse=np.asarray(rmse).reshape(1,2)
rmse=pd.DataFrame(rmse,columns=['train','test'])
r2=np.asarray(r2).reshape(1,2)
r2=pd.DataFrame(r2,columns=['train','test'])
    
