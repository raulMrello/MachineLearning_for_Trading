#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################################################
# incluye google drive api
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials

####################################################################################
# Librerías de manejo de datos 
import pandas as pd
from pandas import concat
import numpy as np

####################################################################################
# Librerías de machine learning
import sklearn
from sklearn import preprocessing
from sklearn import metrics as sk
from sklearn.externals import joblib
import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
from keras.constraints import max_norm

####################################################################################
# Librerías de visualización
import matplotlib.pyplot as plt
from matplotlib import dates, ticker
from matplotlib.dates import (MONDAY, DateFormatter, MonthLocator, WeekdayLocator, date2num)
import matplotlib as mpl
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import Scatter, Layout
import plotly.tools as tls


####################################################################################
# TA-Lib: instalación y carga de la librería
import talib

####################################################################################
# Otras utilidades
import datetime
import time
import os
import sys
import math

import MyUtils as utils



class PredictiveNet:                   
    def __init__(self, name, **kwargs):
        """Constructor
        Args:
            name: Nombre del objeto creado
            kwargs: Lista de parámetros aceptados:
                loopback_window: Time-steps a pasado para utilizar en la predicción (24)
                forward_window: Time-steps a futuro a predecir (4)
                num_lstm_layers: Número de capas LSTM (3)
                num_cells: Número de neuronas en cada capa (128)
                num_dense_layers: Número de capas Dense intermedias (0)
                batch_size: Número de muestras del batch (32)
                suffle_enable: Flag para hacer 'suffling' en el entrenamiento (True)
                tvt_csv_file: Archivo csv con datos históricos train-validate-test
				verbose: Flag de depuración
        """
        self.name = name
        self.lbw = None
        self.fww = None
        self.nll = None
        self.nlc = None
        self.ndl = None
        self.ndc = None
        self.bs = None
        self.sf = None
        _dbg = False
        self.df = None        
        self.dfapp = None
        self.num_outputs = 0
        self.num_inputs = 0
        self.num_in_steps = self.lbw
        self.num_out_steps = self.fww
        self.sts_df = None
        self.sts_src = None
        self.sts_scaled = None
        self.scaler = None
        self.x_train = None
        self.y_train = None
        self.x_validation = None
        self.y_validation = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.callbacks_list = None
        self.columns_to_include = None
        self.columns_to_exclude = None
        self.indicators_to_generate = None
        
        for key,val in kwargs.items():        
            if key=='loopback_window': 
                self.lbw = val
                self.num_in_steps = val
            if key=='forward_window': 
                self.fww = val
                self.num_out_steps = val
            if key=='num_lstm_layers': 
                self.nll = val
            if key=='num_cells': 
                self.nlc = val
                self.ndc = val
            if key=='num_dense_layers': 
                self.ndl = val
            if key=='batch_size': 
                self.bs = val
            if key=='suffle_enabled': 
                self.sf = val
            if key=='verbose':
                if val=='full':
                    _dbg = True
            if key=='tvt_csv_file': 
                if _dbg:
                    print('Cargando histórico...')
                self.df = self.load_hist(val, sep=';', reindex_fillna=True, plot_it=_dbg, debug_it=_dbg)
                if _dbg:
                    print(self.df.head())
                    print('Incluyendo indicadores...')
                self.dfapp = self.add_indicators(self.df, 
                                            'out',
                                            ['CLOSE'], 
                                            ['weekday', 'barsize'], 
                                            ['weekday'], 
                                            ['bollWidthRel', 'bollR', 'atr', 'SMAx3'], 
                                            remove_weekends=True, 
                                            add_applied=True, 
                                            plot_it=_dbg, 
                                            starts=0, 
                                            plot_len=0)
                
                self.num_outputs = 1
                self.num_inputs = self.dfapp.shape[1] - self.num_outputs
                if _dbg:
                    print(self.dfapp.head())
                    print('Parseando a supervisado con ins={}, outs={}...'.format(self.num_inputs,self.num_outputs))
                self.sts_df = self.series_to_supervised(self.dfapp, self.num_inputs, self.num_outputs, self.lbw, self.fww)                
                if _dbg:
                    print(self.sts_df.head())
                    print('Normalizando...')
                self.sts_scaled, self.scaler = self.normalize_data(self.sts_df, None, self.name+'_scaler.save')
                if _dbg:
                    print(self.sts_scaled.head())
                    print('Preparando pares XY...')
                self.x_train, self.y_train, self.x_validation, self.y_validation, self.x_test, self.y_test = self.prepare_training_data(self.sts_scaled, self.bs, 4, 1, 0.8, _dbg)
                if _dbg:
                    print('Contruyendo red...')
                self.model, self.callbacks_list = self.build_net(_dbg) 
                if _dbg:
                    print(self.model.summary())
                

    #--------------------------------------------------------------------------------
    def load_hist(self, filename, **kwargs):
        """Carga un archivo csv con datos train-validate-test
        Args:
            filename: Archivo de datos csv
            kwargs: Lista de parámetros aceptados:
                sep: tipo de separador (,)
                reindex_fillna: Flag para rellenar espacios en blanco y reindexar por fecha (False)
                plot_it: Flag para visualizar las columnas OHLC (False)
                debug_it: Flag para habilitar trazas de depuración (True)
        Returns:
            Dataframe
        """
        sep = ','
        reindex_fillna=False
        plot_it=False  
        debug_it = True
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'sep': sep = value
                if key == 'reindex_fillna': reindex_fillna = value
                if key == 'plot_it': plot_it = value
                if key == 'debug_it': debug_it = value

        df = pd.read_csv(filename, sep=sep) 
        if debug_it:
            print(df.head(),'\r\n--------------------------')
        # crea una nueva columna de timestamp uniendo date & time y crea columnas para el día de la semana, el día del mes, el mes y la hora del día actual 
        df['timestamp'] = df['DATE'] + '  ' + df['TIME'] 
        df['timestamp'] = df['timestamp'].map(lambda x: datetime.datetime.strptime(x, '%Y.%m.%d %H:%M:%S'))  
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if debug_it:
            print(df.head(),'\r\n--------------------------')
        # establece como índice la columna de timestamp con una cadencia de 30min para encontrar los valores NaN
        if reindex_fillna:
            df.set_index('timestamp', inplace=True)
            df = df.reindex(pd.date_range(min(df.index), max(df.index), freq="1H"))
            # elimina los valores nulos y los rellena con el valor anterior
            df.fillna(method='ffill', inplace =True)
        df['weekday'] = df.index.dayofweek
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['hhmm'] = df.index.hour*60 +df.index.minute
        df['barsize'] = df['CLOSE']-df['OPEN']
        df = df[['OPEN','HIGH','LOW','CLOSE','barsize','hhmm','weekday','TICKVOL','VOL','SPREAD']]
        if debug_it:
            print(df.head())

        if plot_it:
            plt.plot(df.OPEN)
            plt.plot(df.HIGH)
            plt.plot(df.LOW)
            plt.plot(df.CLOSE)
            plt.legend(['open','high','low','close'], loc='upper right')        
        return df
                
    #--------------------------------------------------------------------------------
    def add_indicators(self, df, out_applied, applied, add_cols, sub_cols, indicators, **kwargs):
        """Carga un archivo csv con datos train-validate-test
        Args:
            df: DataFrame origen
            out_applied: Columna de salida que se creará a partir de 'applied'
            applied: Columna utilizada para generar la salida y los indicadores
            add_cols: Lista de columnas a generar
            sub_cols: Lista de columnas a eliminar al final de todo el proceso
            indicators: Lista de indicadores a crear
            kwargs: Lista de parámetros aceptados:
                plot_it: Flag para activar la visualización gráfica (False)
                starts: Punto de inicio de la visualización (0)
                plot_len: Punto final de la visualización (0)
                remove_weekends: Flag para habilitar la eliminación de datos en los fines de semana (False)
                add_applied: Flag para habilitar la inclusión de la columna 'applied' como entrada
        Returns:
            Dataframe
        """
        self.columns_to_include = add_cols
        self.columns_to_exclude = sub_cols
        self.indicators_to_generate = indicators
                    
        df = df.copy()    
        plot_it = False
        starts = 0    
        plot_len = 0
        remove_weekends=False
        for key,val in kwargs.items():        
            if key=='plot_it': plot_it=val
            elif key=='starts': starts=val
            elif key=='plot_len': plot_len=val
            elif key=='remove_weekends':
                remove_weekends=val
        if remove_weekends:
            df = df.drop(df[(df['weekday'] > 4)].index)

        cols = list()
        cols = cols + add_cols
        for p in applied:
            if df[p] is None: continue
            upperband, middleband, lowerband = talib.BBANDS(df[p], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            bollinger_width = upperband - lowerband
            sma8 = talib.SMA(df[p], timeperiod=8)
            sma50 = talib.SMA(df[p], timeperiod=50)
            sma100 = talib.SMA(df[p], timeperiod=100)    
            for i in indicators:
                if i+'_'+p in df.columns:
                    continue                
                if i=='bollWidthRel':
                    # obtengo columnas a partir de features TA-Lib            
                    bollinger_width_sma50 = talib.SMA(bollinger_width, timeperiod=50)            
                    df[i+'_'+p] = bollinger_width/(3*bollinger_width_sma50)
                    cols.append(i+'_'+p)

                elif i=='bollR':
                    if i+'_'+p in df.columns:
                        continue                
                    bollR = (df[p] - lowerband)/(upperband - lowerband)
                    bollR[np.isnan(bollR)]=0.5
                    bollR[np.isinf(bollR)]=0.5
                    df[i+'_'+p]=bollR
                    cols.append(i+'_'+p)

                elif i=='atr':
                    df[i+'_'+p] = talib.SMA((df[p] - df.CLOSE.shift(1)),timeperiod=14)
                    cols.append(i+'_'+p)

                elif i=='SMAx3':            
                    df['sma4_'+p] = talib.SMA(df.CLOSE,timeperiod=4)
                    df['sma16_'+p] = talib.SMA(df.CLOSE,timeperiod=16)
                    df['sma40_'+p] = talib.SMA(df.CLOSE,timeperiod=40)
                    cols.append('sma4_'+p)
                    cols.append('sma16_'+p)
                    cols.append('sma40_'+p)

                elif i=='BarSizeSma':            
                    barsize_sma = talib.SMA(df['barsize'],timeperiod=20)
                    df['BarSizeSma'] = df['barsize']/barsize_sma
                    cols.append('BarSizeSma')

                elif i=='HIGH' or i=='LOW' or i== 'CLOSE' or i=='OPEN':
                    if i in cols:
                        continue                
                    cols.append(i)

                else:  
                    print('Indicador "{}" desconocido'.format(i))                    

        for key,val in kwargs.items():        
            if key=='add_applied':
                for i in applied:
                    cols.append(i)

        df[out_applied] = df[applied[0]]
        cols.append(out_applied)
        df = df[cols]    
        df.dropna(inplace=True)   
        dfview = df
        if remove_weekends:
            dfview = df.reset_index()

        if plot_it:
            plt.figure(figsize=(12,24))
            for i in cols:
                plt.subplot(len(cols), 1, cols.index(i)+1)
                plt.plot(dfview[i][starts:starts+plot_len])
                plt.title(i, y=0.5, loc='right')
        df = df.drop(sub_cols,axis=1)
        return df    
                
    #--------------------------------------------------------------------------------
    def series_to_supervised(self, df, num_inputs, num_outputs, n_in=1, n_out=1, dropnan=True):
        """Crea un Dataframe para entranamiento supervisado
        Args:
            df: DataFrame origen
            num_inputs: Número de entradas
            num_outputs: Número de salidas
            n_in: Número de timesteps de entrada (1)
            n_out: Número de timesteps de salida (1)
            dropnan: Flag para habilitar la eliminación de valores NaN (True)
        Returns:
            Dataframe
        """
        # obtengo el dataframe con las entradas y las salidas
        df_in = df[[x for x in df.columns if df.columns.get_loc(x) < num_inputs]]
        df_out = df[[x for x in df.columns if df.columns.get_loc(x)>=num_inputs]]

        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in-1, -1, -1):
            cols.append(df_in.shift(i))
            if i!=0:
              names += [('%s(t-%d)' % (df_in.columns[j], i)) for j in range(0, num_inputs)]
            else:
              names += [('%s(t)' % (df_in.columns[j])) for j in range(0, num_inputs)]

        # forecast sequence (t, t+1, ... t+n)
        for i in range(1, n_out+1):   
            cols.append(df_out.shift(-i))
            if i == 0:
              names += [('%s(t)' % (df_out.columns[j])) for j in range(0, num_outputs)]
            else:
              names += [('%s(t+%d)' % (df_out.columns[j], i)) for j in range(0, num_outputs)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
                
    #--------------------------------------------------------------------------------
    def normalize_data(self, df, csv_to_save=None, scaler_to_save=None):
        """Normaliza los datos utilizando un scaler
        Args:
            df: DataFrame origen
            csv_to_save: Archivo para guardar csv normalizado (None)
            scaler_to_save: Archivo para guardar el scaler (None)
        Returns:
            Dataframe normalizado
            Scaler utilizado
        """
        sts_src = df.copy()
        sts_src_values = sts_src.values
        sts_src_values = sts_src_values.astype('float32')
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        sts_src_scaled_values = scaler.fit_transform(sts_src_values)
        sts_scaled = pd.DataFrame(data=sts_src_scaled_values, columns=sts_src.columns, index=sts_src.index)        
        if csv_to_save is not None:
            sts_scaled.to_csv(csv_to_save, sep=';')
        if scaler_to_save is not None:
            joblib.dump(scaler, scaler_to_save) 
        return sts_scaled,scaler
                
    #--------------------------------------------------------------------------------            
    def walk_forward(self, values, wf_train, wf_validate):
        """Crea los conjuntos de entrenamiento y validación
        Args:
            values: Valores del DataFrame origen con los datos normalizados
            wf_train: Tamaño del conjunto de entrenamiento
            wf_validate: Tamaño del conjunto de validación
        Returns:
            Nº 
        """
        # calculo el número de muestras del walk-forward (wf_size) en base a los wf de entrenamiento y test
        wf_size = wf_train + wf_validate
        # creo arrays para obtener los conjuntos de entranamiento y validación
        nd_train = values[:wf_train,:]
        nd_validate = values[wf_train:wf_size,:]
        count = 1
        # creo un bucle mientras que esté en el rango del número de muestras a procesar
        for i in range(wf_size,values.shape[0], wf_size):
            count += 1
            _t = values[i:i+wf_train,:]
            nd_train = np.append(nd_train, _t).reshape(nd_train.shape[0]+_t.shape[0], _t.shape[1])    
            _v = values[i+wf_train:i+wf_size,:]
            nd_validate = np.append(nd_validate, _v).reshape(nd_validate.shape[0]+_v.shape[0], _v.shape[1])    
        return nd_train,nd_validate,count        
    
    #--------------------------------------------------------------------------------
    def prepare_training_data(self, df, batch_size, train_nb=4, validate_nb=1, trvl_ratio=0.8, debug_it=False):
        """Crea los conjuntos de entrenamiento-validación-test y sus pares XY
        Args:
            df: DataFrame origen con los datos normalizados
            batch_size: Tamaño del batch
            train_nb: Número de batches que forman el conjunto de entrenamiento (4)
            validate_nb: Número de batches que forman el conjunto de validación (1)
            trvl_ratio: Ratio de pares de entrenamiento respecto de validación (0.8)
            debug_it: Flag para habilitar trazas de depuración (False)
        Returns:
            Pares XY de: train, validation, test
        """
        df = df.copy()
        # establezco un ratio train-validation de (80%, 20%), para lo que tomo bloques de 5 batches: 4-train 1-validate
        wf_train = train_nb * batch_size
        wf_validate = validate_nb * batch_size
        # calculo el número de muestras train-validate para que sea un múltiplo entero de 5 batches
        wf_forward = wf_train + wf_validate
        total_samples = df.shape[0]
        trvl_samples = total_samples * trvl_ratio
        num_blocks = int(trvl_samples/wf_forward)
        trvl_samples = num_blocks * wf_forward
        test_samples = total_samples - trvl_samples
        if debug_it:
            print('Total samples............... ', total_samples)
            print('Train-Validate samples...... ', trvl_samples)
            print('Num blocks.................. ', num_blocks)
            print('Train per block............. ', wf_train)
            print('Validate per block.......... ', wf_validate)
            print('Total trains................ ', wf_train * num_blocks)
            print('Total validates............. ', wf_validate * num_blocks)    

        # obtengo los grupos en formato np.ndarray
        train_validate_values = df[0:trvl_samples].values
        self.test = df[trvl_samples:].values

        # obtengo las listas de entrenamiento-validación y el n
        self.train,self.validation,count = self.walk_forward(train_validate_values, wf_train, wf_validate) 

        # obtengo los pares x-y de cada conjunto
        x_train, y_train = self.train[:,:-(self.num_outputs * self.num_out_steps)], self.train[:,-(self.num_outputs * self.num_out_steps):]
        x_validation, y_validation = self.validation[:,:-(self.num_outputs * self.num_out_steps)], self.validation[:,-(self.num_outputs * self.num_out_steps):]
        x_test, y_test = self.test[:,:-(self.num_outputs * self.num_out_steps)], self.test[:,-(self.num_outputs * self.num_out_steps):]
        if debug_it:
            print('x_train shape:', x_train.shape)
            print('y_train shape:', y_train.shape)
            print('x_validation shape:', x_validation.shape)
            print('y_validation shape:', y_validation.shape)
            print('x_test shape:', x_test.shape)
            print('y_test shape:', y_test.shape)
        return x_train, y_train, x_validation, y_validation, x_test, y_test
    
    #--------------------------------------------------------------------------------
    ####################################################################################
    # Callback para descargar pesos de la red neuronal
    class DownloadWeights(keras.callbacks.Callback):
      def __init__(self, filepath):
        self.filepath = filepath
      def on_epoch_end(self, epoch, logs={}):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
          json_file.write(model_json)
        # Serializa los pesos a formato HDF5
        self.model.save_weights(self.filepath)
        # Descargo el archivo al ordenador local
        files.download(self.filepath)    
        print("Saved model to disk")
        return
      '''
      def on_train_begin(self, logs={}):
        pass
      def on_train_end(self, logs={}):
        return
      def on_batch_begin(self, batch, logs={}):
        return
      def on_batch_end(self, batch, logs={}):
        return
      def on_epoch_begin(self, epoch, logs={}):
        return
      '''
    
    #--------------------------------------------------------------------------------
    ####################################################################################
    # callback para hacer backup de los pesos en el ordenador local
    class FitLogger(keras.callbacks.Callback):
      def __init__(self, fitfile):
        self.fitfile = fitfile
        self.loss, self.val_loss, self.acc, self.val_acc = np.ndarray((0,)), np.ndarray((0,)),np.ndarray((0,)), np.ndarray((0,))
        dfx = pd.DataFrame(data=[[0],[0],[0],[0]]).transpose()
        dfx.columns=['loss','val_loss','acc','val_acc']
        dfx.to_csv(self.fitfile)
        print('file "{}" updated'.format(self.fitfile))

      '''
      def on_epoch_end(self, epoch, logs={}):
        return
      def on_train_begin(self, logs={}):
        pass
      def on_train_end(self, logs={}):
        return
      def on_epoch_begin(self, epoch, logs={}):
        return
      def on_batch_begin(self, batch, logs={}):
        print('Starting batch...')
        return
      def on_batch_end(self, batch, logs={}):
        print('... batch Finished!')
        return
      def on_epoch_begin(self, epoch, logs={}):
        print('Starting epoch...', epoch)
        return
      '''
      def on_epoch_end(self, epoch, logs={}):
        self.loss = np.append(self.loss, [logs.get('loss')], axis=0)
        self.val_loss = np.append(self.val_loss, [logs.get('val_loss')], axis=0)
        self.acc = np.append(self.acc, [logs.get('acc')], axis=0)
        self.val_acc = np.append(self.val_acc, [logs.get('val_acc')], axis=0)
        s0,s1,s2,s3 = pd.Series(self.loss), pd.Series(self.val_loss), pd.Series(self.acc[-self.loss.shape[0]:]), pd.Series(self.val_acc[-self.loss.shape[0]:])
        dfx = pd.DataFrame(data=[s0,s1,s2,s3]).transpose()
        dfx.columns=['loss','val_loss','acc','val_acc']
        dfx.to_csv(self.fitfile)
        return
    
    #--------------------------------------------------------------------------------
    def build_net(self, debug_it=False):
        """Crea la red neuronal
        Args:
            debug_it: Flag para habilitar depuración
        Returns:
            model,callback_list
        """
        weights_file= self.name+'_weights.hd5'
        fitlog_to_save= self.name+'_fitlog.csv'
        
        # Inicio definiendo un modelo secuencial
        model = Sequential()

        sequence_flag = False
        if self.nll > 1:
            sequence_flag = True

        # capa de entrada, debe especificar formato 'input_shape'
        model.add(LSTM(self.nlc, return_sequences=sequence_flag, input_shape=(self.lbw, self.num_inputs)))
        #model.add(BatchNormalization())
        model.add(Dropout(0.20))

        for i in range(1, self.nll, 1):
            if i == self.nll-1:
              sequence_flag = False
            # capas intermedias
            model.add(LSTM(self.nlc, return_sequences=sequence_flag))
            #model.add(BatchNormalization())
            model.add(Dropout(0.20))

        for i in range(1, self.ndl, 1):
            model.add(Dense(self.ndc, activation='linear'))    
            
        # la capa de salida es una capa Dense con tantas salidas como timesteps a predecir con activación lineal     
        model.add(Dense(self.num_outputs * self.fww, activation='linear'))

        # compilo con optimizador Adam y pérdida 'mse'
        opt = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=False)
        model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])   

        #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        #download_weights = DownloadWeights(filepath)
        # callback para visualización en tensorboard
        #tbCallBack = TensorBoard(log_dir='./log', histogram_freq=1,write_graph=True,write_grads=True,batch_size=batch_size,write_images=True)        
        checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        fitlogger = self.FitLogger(fitlog_to_save)
        callbacks_list = [checkpoint, fitlogger]
        
        # si existe un modelo previo, lo carga
        if weights_file is not None:
            try:
                model.load_weights(weights_file)
                if debug_it:
                    print('Loaded weights from file: ', weights_file)
            except:
                if debug_it:
                    print('No weights file to load')

        if debug_it:
            model.summary()
        return model, callbacks_list
    
    #--------------------------------------------------------------------------------
    def stock_loss(self, y_true, y_pred):
        """ Métrica loss """
        alpha = 100.
        loss = K.switch(K.less(y_true * y_pred, 0), alpha*y_pred**2 - K.sign(y_true)*y_pred + K.abs(y_true), K.abs(y_true - y_pred))
        return K.mean(loss, axis=-1)
    
    #--------------------------------------------------------------------------------
    def rmse(self, y_true, y_pred):
        """ Métrica RMSE """
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
    
    #--------------------------------------------------------------------------------
    def train_validate(self, num_epochs, plot_results=False):
        """Realiza el proceso de entrenamiento y validación
        Args:
            num_epochs: Epochs a ejecutar
            plot_results: Flag para habilitar la visualización gráfica de resultados
        Returns:
            History result
        """
        try:
            train_x = self.x_train.reshape(self.x_train.shape[0], self.num_in_steps, self.num_inputs)
            train_y = self.y_train.reshape(self.y_train.shape[0], self.num_out_steps * self.num_outputs)
            validation_x = self.x_validation.reshape(self.x_validation.shape[0], self.num_in_steps, self.num_inputs)
            validation_y = self.y_validation.reshape(self.y_validation.shape[0], self.num_out_steps * self.num_outputs)
            history = self.model.fit(train_x, train_y, epochs=num_epochs, batch_size=self.bs, callbacks=self.callbacks_list, validation_data=(validation_x,validation_y), verbose=2, shuffle=self.sf) 
            if plot_results:
                # visualizo el resultado de la ejecución de la celda actual
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper right')
                plt.show()
            return history
        except:
            print('Model fit Exception:', sys.exc_info()[0])
            return None
    
    #--------------------------------------------------------------------------------
    def test_eval(self): 
        """Realiza el proceso de test
        Returns:
            loss: Pérdida media
            acc: Precisión media
        """
        test_x = self.x_test.reshape(self.x_test.shape[0], self.lbw, self.num_inputs)
        test_y = self.y_test.reshape(self.y_test.shape[0], self.fww * self.num_outputs)            
        scores = self.model.evaluate(test_x, test_y, batch_size=1, verbose=2)
        loss = scores[0]
        acc = scores[1]
        return loss,acc

		
    #--------------------------------------------------------------------------------
    def test_rmse(self, plot_results=True, plot_from=70, plot_to=200):
        """Realiza el proceso de test
        Args:
            plot_results: Flag para habilitar la visualización gráfica de resultados
            plot_from: Punto desde el que visualizar
            plot_to: Punto hasta el que visualizar
        Returns:
            inp: Lista con datos de entrada
            target: Lista con datos objetivo
            predict: Lista con datos predichos
        """
        test_x = self.x_test.reshape(self.x_test.shape[0], self.lbw, self.num_inputs)
        test_y = self.y_test.reshape(self.y_test.shape[0], self.fww * self.num_outputs)            
        scores = self.model.evaluate(test_x, test_y, batch_size=1, verbose=2)
        print('Model Loss: ', scores[0])  
        print("Model Accuracy: ",scores[1])
        x_data = test_x
        y_data = test_y

        rmse = list()
        inp,target,pred=list(),list(),list()
        for sample in range(x_data.shape[0]):
            # Realizo predicción del primer conjunto de datos
            x = x_data[sample].reshape(1, self.lbw, self.num_inputs)
            y = y_data[sample].reshape(1,self.fww * self.num_outputs)
            predictions = self.model.predict(x, batch_size=1,verbose=0)

            # deshago el scaling
            xy_values = np.concatenate((x.reshape(1, x.shape[1]*x.shape[2]), y),axis=1)
            xy_values = self.scaler.inverse_transform(xy_values)
            xyhat_values = np.concatenate((x.reshape(1, x.shape[1]*x.shape[2]), predictions),axis=1)
            xyhat_values = self.scaler.inverse_transform(xyhat_values)

            # Calculo el error RMSE
            rmse_val = math.sqrt(sk.mean_squared_error(xy_values[0,-(self.fww * self.num_outputs):], xyhat_values[0,-(self.fww * self.num_outputs):], multioutput = 'uniform_average'))
            rmse.append(rmse_val)
            inp.append(xy_values[0,-(self.fww * self.num_outputs)-1])
            target.append(xy_values[0,-(self.fww * self.num_outputs)])
            pred.append(xyhat_values[0,-(self.fww * self.num_outputs)])    
        
        if plot_results:        
            plt.figure(figsize=(12,12))
            plt.subplot(2,1,1)
            plt.plot(np.asarray(rmse))    
            plt.legend(['RMSE'])
            plt.subplot(2,1,2)
            # visualizo los resultados anteriores en un rango dado
            i,j = plot_from,plot_to
            plt.plot(np.asarray(inp).reshape((len(inp),))[i:j],color='black')
            plt.plot(np.asarray(target).reshape((len(target),))[i:j],color='green')
            plt.plot(np.asarray(pred).reshape((len(pred),))[i:j],color='red')
            plt.legend(['x_test', 'y_target', 'y_predicted'])
        return inp,target,pred
    
    #--------------------------------------------------------------------------------
    def test_with_predictions(self, num_samples=100, sample_step=10):
        """Realiza el proceso de test y visualiza las predicciones
        Args:
            num_samples: Número de muestras a verificar
            sample_step: Lapso de muestras para visualizar las predicciones
        """
        test_x = self.x_test.reshape(self.x_test.shape[0], self.lbw, self.num_inputs)
        test_y = self.y_test.reshape(self.y_test.shape[0], self.fww * self.num_outputs)  
        x_data = test_x
        y_data = test_y

        isample = self.num_in_steps
        num_samples = 100
        step = 10

        target_vals,pred_vals, pred_idx=list(),list(),list()
        for s in range(num_samples):
            sample = isample+s
            # Realizo predicción del primer conjunto de datos
            x = x_data[sample].reshape(1, self.num_in_steps, self.num_inputs)
            y = y_data[sample].reshape(1,self.num_out_steps * self.num_outputs)
            predictions = self.model.predict(x, batch_size=1,verbose=0)

            # deshago el scaling
            xy_values = np.concatenate((x.reshape(1, x.shape[1]*x.shape[2]), y),axis=1)
            xy_values = self.scaler.inverse_transform(xy_values)
            xyhat_values = np.concatenate((x.reshape(1, x.shape[1]*x.shape[2]), predictions),axis=1)
            xyhat_values = self.scaler.inverse_transform(xyhat_values)

            # Calculo el error RMSE
            target_vals.append(xy_values[0,-(self.num_out_steps * self.num_outputs)-1])
            if s>=self.lbw and s % step == 0:
                print(xy_values[0,-(self.num_out_steps * self.num_outputs):], xyhat_values[0,-(self.num_out_steps * self.num_outputs):])
                pred_vals.append(xyhat_values[0,-(self.num_out_steps * self.num_outputs):])    
                pred_idx.append(s)

        plt.figure(figsize=(12,12))
        plt.plot(np.asarray(target_vals).reshape((len(target_vals),)),color='black')
        for i in range(len(pred_vals)):
            values = np.asarray(pred_vals[i])
            pos = np.arange(pred_idx[i], pred_idx[i]+values.shape[0])
            plt.plot(pos, values, color='r')
    
    #--------------------------------------------------------------------------------
    def test_full_predictions(self, **kwargs):
        """Realiza el proceso de test y visualiza las predicciones
        Args:
            print_logs: Flag para activar trazas de depuración de esta celda (False)
            test_len: Establezco el número de muestras a futuro para realizar el test (100)
            predict_with_ctrol: Establezco el modo de predicción (con o sin control) (True)
            ctrol_wdow: Establezco la ventana del dataframe 'df' a seleccionar (trvl_samples)
            ctrol_wdow_prev: Nº de muestras previas de control a visualizar (50)
            add_first_steps: Flag para añadir las num_out_steps filas al final de dfpred antes del test (False)

        """
        sts_src = self.sts_df.copy()
        trvl_samples = self.x_train.shape[0]+self.x_validation.shape[0]
        print_logs = False              # Flag para activar trazas de depuración de esta celda
        test_len = 100                   # Establezco el número de muestras a futuro para realizar el test
        predict_with_ctrol=True         # Establezco el modo de predicción (con o sin control)
        ctrol_wdow = trvl_samples       # Establezco la ventana del dataframe 'df' a seleccionar (df[:-ctrol_wdow])
        ctrol_wdow_prev=50              # Nº de muestras previas de control a visualizar
        add_first_steps=False           # Flag para añadir las num_out_steps filas al final de dfpred antes del test
        for key,val in kwargs.items():        
            if key=='print_logs': 
                print_logs = val
            if key=='test_len': 
                test_len = val
            if key=='predict_with_ctrol': 
                predict_with_ctrol = val
            if key=='ctrol_wdow': 
                ctrol_wdow = val
            if key=='ctrol_wdow_prev': 
                ctrol_wdow_prev = val
            if key=='add_first_steps': 
                add_first_steps = val
        
        # dpred contiene una copia del dataframe origen 'df' que puede ser:
        # 1. Modo predicción con control de resultado (si dfpred+test_len es un slice de df)
        # 2. Modo predicción a futuro real (si dfpred es una copia exacta de df)
        dfpred = self.df.copy()
        if add_first_steps:
            delta = dfpred.index[1] - dfpred.index[0]
            max_index = dfpred.index[-1] + (self.num_out_steps * delta)
            new_index = pd.date_range(min(dfpred.index), max_index, freq="1H")
            dfpred = dfpred.reindex(new_index)
            dfpred.loc[-self.num_out_steps:,:]=0   

        control_data = dfpred[dfpred.shape[0] - self.num_out_steps-ctrol_wdow_prev:dfpred.shape[0] - self.num_out_steps]
        if predict_with_ctrol:
            dfpred = self.df.loc[self.df.index <= sts_src.index[ctrol_wdow+self.num_out_steps]].copy()
            control_data = self.df[dfpred.shape[0] - self.num_out_steps-ctrol_wdow_prev:dfpred.shape[0] - self.num_out_steps+test_len]

        # inicio sin ninguna predicción
        ypred = np.nan

        # creo lista para almacenar las predicciones de la red 
        pred = list()

        # para cada muestra de test
        for sample in range(test_len):    
            # si hay muestra predicha del paso anterior, la reinserto como una nueva muestra actual
            if not np.isnan(ypred):
                dfpred.iloc[-self.num_out_steps]['CLOSE'] = ypred
                # añado una nueva fila al final y pongo sus valores a 0
                delta = dfpred.index[1] - dfpred.index[0]
                max_index = dfpred.index[-1] + delta
                new_index = pd.date_range(min(dfpred.index), max_index, freq="1H")
                dfpred = dfpred.reindex(new_index)
                dfpred.loc[-self.num_out_steps:,:]=0

            # en este punto dfpred es el dataframe original con 'num_out_steps' filas al final todas a 0.
            # añado indicadores técnicos
            dfapp_pred = self.add_indicators(dfpred, 'out',['CLOSE'], self.columns_to_include, self.columns_to_exclude, self.indicators_to_generate, remove_weekends=True, add_applied=True)

            # en este punto dfapp_pred es el dataframe con todos los indicadores técnicos
            # creo modelo con formato de entrenamiento supervisado para el scaler
            sts_src_pred = self.series_to_supervised(dfapp_pred, self.num_inputs, self.num_outputs, self.num_in_steps, self.num_out_steps)

            # en este punto sts_src_pred tiene todos los datos listos antes de la normalización con 'scaler'   
            # normalizo los datos
            sts_src_pred_values = sts_src_pred.values
            sts_src_pred_values = sts_src_pred_values.astype('float32')
            sts_src_pred_scaled_values = self.scaler.transform(sts_src_pred_values)
            sts_scaled_pred = pd.DataFrame(data=sts_src_pred_scaled_values, columns=sts_src_pred.columns, index=sts_src_pred.index)
            sts_scaled_pred_values = sts_scaled_pred.values

            # en este punto sts_scaled_pred tiene los valores listos para la red neuronal
            # genero una predicción
            x = sts_scaled_pred_values[-1,0:self.num_inputs*self.num_in_steps].reshape(1, self.num_in_steps, self.num_inputs)
            predictions = self.model.predict(x, batch_size=1,verbose=0)

            # en este punto ya tengo la siguiente predicción
            # deshago el scaling
            xyhat_values = np.concatenate((x.reshape(1, x.shape[1]*x.shape[2]), predictions),axis=1)
            xyhat_values = self.scaler.inverse_transform(xyhat_values)

            # en este punto tengo el valor denormalizado de la predicción
            # guardo la última entrada aplicada y la predicción en t+1 para repetir el bucle
            xreal = xyhat_values[0,-(self.num_outputs*self.num_out_steps)-1]
            if np.isnan(ypred):
                pred.append(xreal)
            ypred = xyhat_values[0,self.num_inputs*self.num_in_steps]  
            pred.append(ypred)  
            print('sample {} de {}'.format(sample+1, test_len))


        y_pred = np.asarray(pred).reshape((len(pred),))
        x_pred_idx = np.arange(ctrol_wdow_prev-1,ctrol_wdow_prev-1+y_pred.shape[0])
        x_previous = control_data['CLOSE'].values
        #y_trgt = np.asarray(targets).reshape((len(targets),))

        plt.figure(figsize=(12,12)) 
        plt.plot(np.arange(x_previous.shape[0]),x_previous, color='black')
        plt.plot(x_pred_idx, y_pred, color='r')
        if predict_with_ctrol:
            plt.legend(['y_target','y_pred'])
        else:
            plt.legend(['x','y_pred'])     
    
    #--------------------------------------------------------------------------------
