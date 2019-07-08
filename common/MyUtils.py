#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################################################
# Chequeo si está operativo el entorno de Google Colaboratory
import sys
ENABLE_GOOGLE_COLAB = 'google.colab' in sys.modules
ENABLE_GOOGLE_COLAB

####################################################################################
# incluye google drive api
if ENABLE_GOOGLE_COLAB:
  from pydrive.auth import GoogleAuth
  from pydrive.drive import GoogleDrive
  from oauth2client.client import GoogleCredentials

####################################################################################
# Librerías de manejo de datos 
import pandas as pd
from pandas import concat
from pandas.plotting import scatter_matrix
import numpy as np

####################################################################################
# Librerías de machine learning
import sklearn
from sklearn import preprocessing
from sklearn import metrics as sk
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

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
import pickle
from enum import Enum

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def save_to_file(obj, filepath):
  with open(filepath, 'wb') as f:
    try:
      pickle.dump(obj, f)
    except MemoryError as error:
      exc_type, exc_value, exc_traceback = sys.exc_info()
      print('type:', exc_type, 'value:', exc_value)
        
        

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def load_from_file(filepath):
  with open(filepath, 'rb') as f:
    obj = pickle.load(f)        
  return obj   


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def static_vars(**kwargs):
  """ Decorador para definir variables estáticas en funciones
  """
  def decorate(func):
    for k in kwargs:
      setattr(func, k, kwargs[k])
    return func
  return decorate

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def mergeDataframes(df1, df2, df3, prefixes = ['df1_', 'df2_', 'df3_']):
  """ Merges 3 dataframes using column TIME as key for merging with inner
      merging method. Fills NaN with ffill.
  """
  # set same timestamp limits
  first_row = max(_df1.TIME.iloc[0], _df2.TIME.iloc[0], _df3.TIME.iloc[0])
  last_row = max(_df1.TIME.iloc[-1], _df2.TIME.iloc[-1], _df3.TIME.iloc[-1])
  _df1 = df1[(df1.TIME >= first_row) & (df1.TIME <= last_row)].copy()
  _df2 = df2[(df2.TIME >= first_row) & (df2.TIME <= last_row)].copy()
  _df3 = df3[(df3.TIME >= first_row) & (df3.TIME <= last_row)].copy()
  # rename columns
  cols = []
  for c in _df1.columns:
    if c == 'TIME':
      cols.append(c)
    else:
      cols.append('{}{}'.format(prefixes[0], c))
  _df1.columns = cols
  cols = []
  for c in _df2.columns:
    if c == 'TIME':
      cols.append(c)
    else:
      cols.append('{}{}'.format(prefixes[1], c))
  _df2.columns = cols
  cols = []
  for c in _df3.columns:
    if c == 'TIME':
      cols.append(c)
    else:
      cols.append('{}{}'.format(prefixes[2], c))
  _df3.columns = cols
  # merge dataframes
  _df23 = _df2.merge(_df3, on='TIME', how='inner', suffixes=('',''))
  _df123 = _df1.merge(_df23, on='TIME', how='inner', suffixes=('',''))
  _df123.fillna(inplace=True, method='ffill')
  return _df123



#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def load_hist(filename, **kwargs):
  """Carga un archivo csv con datos train-validate-test
    Args:
      filename: Archivo de datos csv
      kwargs: Lista de parámetros aceptados:
        sep: tipo de separador (,)
        remove_weekends: Flag para borrar datos del fin de semana sin cotización
        remove_index: Flag para borrar el índice y obtener el timestamp como una nueva columna
        freq : Frecuencia del rango horario 1H, 1M, ...
  Returns:
    Dataframe
  """
  sep = ','
  freq="1H"
  remove_weekends=False
  remove_index=False
  if kwargs is not None:
    for key, value in kwargs.items():
      if key == 'sep': 
        sep = value
      if key == 'remove_weekends': 
        remove_weekends = value
      if key == 'remove_index': 
        remove_index = value
      if key == 'freq': 
        freq = value

  df = pd.read_csv(filename, sep=sep) 
  # crea una nueva columna de timestamp uniendo date & time y crea columnas para el día de la semana, el día del mes, el mes y la hora del día actual 
  df['timestamp'] = df['DATE'] + '  ' + df['TIME'] 
  df['timestamp'] = df['timestamp'].map(lambda x: datetime.datetime.strptime(x, '%Y.%m.%d %H:%M:%S'))  
  df['timestamp'] = pd.to_datetime(df['timestamp'])
  df.set_index('timestamp', inplace=True)
  df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq=freq))
  # elimina los valores nulos y los rellena con el valor anterior
  df.fillna(method='ffill', inplace =True)
  df['WEEKDAY'] = df.index.dayofweek
  df['day'] = df.index.day
  df['month'] = df.index.month
  df['HHMM'] = df.index.hour*60 +df.index.minute
  df['OC2'] = (df['CLOSE']+df['OPEN'])/2
  df['HLC3'] = (df['HIGH']+df['LOW']+df['CLOSE'])/3
  df['OHLC4'] = (df['OPEN']+df['HIGH']+df['LOW']+df['CLOSE'])/4
  df['CLOSE_DIFF'] = df['CLOSE'] - df['CLOSE'].shift(1)
  df = df.dropna()
  df = df[['OPEN','HIGH','LOW','CLOSE','OC2','HLC3','OHLC4','CLOSE_DIFF','HHMM','WEEKDAY','TICKVOL','VOL','SPREAD']]
  if remove_weekends:
    print('Deleting weekends...')
    df = df.drop(df[(df['WEEKDAY'] > 4)].index)
  if remove_index:
    print('Deleting index...')
    df = df.reset_index()
    df = df.rename(index=str, columns={"index": "TIMESTAMP"})
  return df
  

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def add_indicators(df, **kwargs):
  """Añade diferentes indicadores técnicos al dataframe de referencia, devolviendo un nuevo
     dataframe.
     Ejemplo de uso:
     add_indicators(df, 
                    applied=['CLOSE'], 
                    base_cols=['OPEN','HIGH','LOW','CLOSE','OC2','HLC3','OHLC4'],
                    remove_weekends=True,
                    plot_it=True,
                    starts=1000,
                    plot_len=1000)
  Args:
      df: DataFrame origen
      kwargs: Lista de parámetros aceptados:
          plot_it: Flag para activar la visualización gráfica (False)
          starts: Punto de inicio de la visualización (0)
          plot_len: Punto final de la visualización (0)
          remove_weekends: Flag para habilitar la eliminación de datos en los fines de semana (False)
          add_applied: Flag para habilitar la inclusión de la columna 'applied' como entrada
  Returns:
      Dataframe
  """
  df = df.copy()    
  plot_it = False
  starts = 0    
  plot_len = 0
  remove_weekends=False
  applied = []
  base_cols = []
  for key,val in kwargs.items():        
    if key=='applied': applied=val
    elif key=='base_cols': base_cols=val
    elif key=='plot_it': plot_it=val
    elif key=='starts': starts=val
    elif key=='plot_len': plot_len=val
    elif key=='remove_weekends':
      remove_weekends=val
    
  if remove_weekends:
    df = df.drop(df[(df['weekday'] > 4)].index)

  # lista de columnas añadidas
  cols = base_cols
  
  # Añade diferentes indicadores para cada applied
  for p in applied:
    # Incluye el indicador de bandas bollinger 
    upperband, middleband, lowerband = talib.BBANDS(df[p], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['IND_BB_UPPER_'+p] = upperband
    cols.append('IND_BB_UPPER_'+p)
    df['IND_BB_MIDDLE_'+p] = middleband
    cols.append('IND_BB_MIDDLE_'+p)
    df['IND_BB_LOWER_'+p] = lowerband
    cols.append('IND_BB_LOWER_'+p)
    df['IND_BB_WIDTH_'+p] = upperband - lowerband
    cols.append('IND_BB_WIDTH_'+p)
    bollR = (df[p] - lowerband)/(upperband - lowerband)
    bollR[np.isnan(bollR)]=0.5
    bollR[np.isinf(bollR)]=0.5
    df['IND_BB_PERCENT_'+p] = bollR
    cols.append('IND_BB_PERCENT_'+p)
    
    # Incluye un indicador sintético basado en la anchura de bandas bollinger y su SMA50
    df['IND_BB_WIDTH_SMA4_'+p] = talib.SMA(df['IND_BB_WIDTH_'+p], timeperiod=4)            
    df['IND_BB_WIDTH_SMA12_'+p] = talib.SMA(df['IND_BB_WIDTH_'+p], timeperiod=12)            
    cols.append('IND_BB_WIDTH_SMA4_'+p)
    cols.append('IND_BB_WIDTH_SMA12_'+p)
    
    # Incluye varias medias móviles
    df['IND_SMA4_'+p] = talib.SMA(df[p],timeperiod=4)
    df['IND_SMA16_'+p] = talib.SMA(df[p],timeperiod=16)
    df['IND_SMA40_'+p] = talib.SMA(df[p],timeperiod=40)
    cols.append('IND_SMA4_'+p)
    cols.append('IND_SMA16_'+p)
    cols.append('IND_SMA40_'+p)
    
    # Incluye MACD
    macd, macdsignal, macdhist = talib.MACD(df[p], fastperiod=12, slowperiod=26, signalperiod=9)
    df['IND_MACD_'+p] = macd
    df['IND_MACD_SIG_'+p] = macdsignal
    df['IND_MACD_HIST_'+p] = macdhist
    cols.append('IND_MACD_'+p)
    cols.append('IND_MACD_SIG_'+p)
    cols.append('IND_MACD_HIST_'+p)
    
    # Incluye RSI
    df['IND_RSI_'+p] = talib.RSI(df[p], timeperiod=14)
    cols.append('IND_RSI_'+p)
    
    # APO
    df['IND_APO_'+p] = talib.APO(df[p], fastperiod=12, slowperiod=26, matype=0)
    cols.append('IND_APO_'+p)

    # MOMentum
    df['IND_MOM_'+p] = talib.MOM(df[p], timeperiod=10)
    cols.append('IND_MOM_'+p)

    # ROCP
    df['IND_ROCP_'+p] = talib.ROCP(df[p], timeperiod=10)
    cols.append('IND_ROCP_'+p)

    # ROCR
    df['IND_ROCR_'+p] = talib.ROCR(df[p], timeperiod=10)
    cols.append('IND_ROCR_'+p)
  
  # Ahora incluye indicadores que no dependen del componente 'applied'
  # Incluye Williams %R
  df['IND_WILLR'] = talib.WILLR(df['HIGH'], df['LOW'], df['CLOSE'], timeperiod=14)
  cols.append('IND_WILLR')
  
  # Incluye ATR y una media móvil asociada
  df['IND_ATR'] = talib.ATR(df['HIGH'], df['LOW'], df['CLOSE'], timeperiod=14)
  df['IND_ATR_SMA4'] = talib.SMA(df['IND_ATR'],timeperiod=4)
  df['IND_ATR_SMA12'] = talib.SMA(df['IND_ATR'],timeperiod=12)
  cols.append('IND_ATR')
  cols.append('IND_ATR_SMA4')
  cols.append('IND_ATR_SMA12')

  # Incluye indicador ADX
  df['IND_ADX'] = talib.ADX(df['HIGH'], df['LOW'], df['CLOSE'], timeperiod=14)
  cols.append('IND_ADX')

  # ADXR
  df['IND_ADXR'] = talib.ADXR(df['HIGH'], df['LOW'], df['CLOSE'], timeperiod=14)
  cols.append('IND_ADXR')

  df['IND_CCI'] = talib.CCI(df['HIGH'], df['LOW'], df['CLOSE'], timeperiod=14)
  cols.append('IND_CCI')

  df['IND_SLOWK'], df['IND_SLOWD'] = talib.STOCH(df['HIGH'], df['LOW'], df['CLOSE'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
  cols.append('IND_SLOWK')
  cols.append('IND_SLOWD')

  #df['IND_SAR'] = talib.SAR(df['HIGH'], df['LOW'], acceleration=0, maximum=0)
  #cols.append('IND_SAR')  
    
  # Selecciona únicamente las columnas deseadas más los indicadores creados
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
  return df  

  

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def add_candle_patterns(df):
  """ Agrega todos los identificadores de patrones de velas de la librería TALIB. 
      Comento aquellos que generalmente se producen en muy poca cantidad y por lo
      tanto no son representativos durante el entrenamiento del modelo
  """
  df = df.copy()
  #df['CDL2CROWS'] = talib.CDL2CROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDL3INSIDE'] = talib.CDL3INSIDE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(df.OPEN, df.HIGH, df.LOW, df.CLOSE, 0)
  df['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLBELTHOLD'] = talib.CDLBELTHOLD(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  #df['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  #df['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  #df['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(df.OPEN, df.HIGH, df.LOW, df.CLOSE, 0)
  df['CDLDOJI'] = talib.CDLDOJI(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLDOJISTAR'] = talib.CDLDOJISTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLENGULFING'] = talib.CDLENGULFING(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE, 0)
  df['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE, 0)
  df['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLHAMMER'] = talib.CDLHAMMER(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLHARAMI'] = talib.CDLHARAMI(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLHIKKAKE'] = talib.CDLHIKKAKE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLINNECK'] = talib.CDLINNECK(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLKICKING'] = talib.CDLKICKING(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  #df['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLLONGLINE'] = talib.CDLLONGLINE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLMARUBOZU'] = talib.CDLMARUBOZU(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLMATHOLD'] = talib.CDLMATHOLD(df.OPEN, df.HIGH, df.LOW, df.CLOSE, 0)
  df['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE, 0)
  df['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE, 0)
  df['CDLONNECK'] = talib.CDLONNECK(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLPIERCING'] = talib.CDLPIERCING(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLSHORTLINE'] = talib.CDLSHORTLINE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLTAKURI'] = talib.CDLTAKURI(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLTHRUSTING'] = talib.CDLTHRUSTING(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLTRISTAR'] = talib.CDLTRISTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  # normalizo en el rango (-1,1)
  drop_cols = []
  for c in df.columns:
    if c.startswith('CDL'): 
      m = max(abs(df[c].max()),abs(df[c].min()))
      if m != 0:
        df[c] /= m
      else:
        drop_cols.append(c)
  # elimino las columnas que no aportan ninguna señal
  df.drop(columns=drop_cols, inplace = True)
  return df



#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
@static_vars(i=0)
def drawdown_calculator(values, wdow, just_value = 3, do_plot = False):
  """ Calcula para una serie de valores: la variación de principio a fin, y los
      drawdown_bearish y drawdown_bullish máximos.
      Utiliza la variables interna 'drawdown_calculator.i' como índice incremental
      para realizar el cálculo con una 'lambda'
      Args:
        values  : Array de datos a evaluar
        wdow    : Muestras del array 'values' para operación incrementeal en 'lambda'
        do_plot : Flag para habilitar el pintado
      Returns:
        Variación, drawdown_bullish, drawdown_bearish
  """
  v = values[drawdown_calculator.i:drawdown_calculator.i+wdow]
  drawdown_calculator.i += 1
  try:
    # posición del máximo
    ibull = np.argmax(np.maximum.accumulate(v) - v) 
    # posición del mínimo
    jbull = np.argmax(v[:ibull]) # start of period
  except:
    ibull,jbull = 0,0
  try:  
    ibear = np.argmax(v - np.minimum.accumulate(v))
    jbear = np.argmin(v[:ibear])
  except:
    ibear,jbear = 0,0

  if do_plot:
    plt.plot(v)
    plt.plot([ibull, jbull], [v[ibull], v[jbull]], 'o', color='Green', markersize=10)    
    plt.plot([ibear, jbear], [v[ibear], v[jbear]], 'x', color='Red', markersize=10)
  
  if just_value is 0:
    return (v[-1] - v[0])
  elif just_value is 1:
    return (v[ibull] - v[jbull])
  elif just_value is 2:
    return (v[ibear] - v[jbear])
  else:
    return (v[-1] - v[0]), (v[ibull] - v[jbull]), (v[ibear] - v[jbear])  
  



#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def series_to_supervised(df, num_inputs, num_outputs, n_in=1, n_out=1, dropnan=True):
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
  df = df.copy()
  df_in = df[[x for x in df.columns if df.columns.get_loc(x) < num_inputs]]
  df_out = df[[x for x in df.columns if df.columns.get_loc(x)>=num_inputs]]

  cols, names = list(), list()
  # input sequence (t-n-1, ... t)
  for i in range(n_in-1, -1, -1):
    cols.append(df_in.shift(i))
    if i!=0:
      names += [('%s(t-%d)' % (df_in.columns[j], i)) for j in range(0, num_inputs)]
    else:
      names += [('%s(t)' % (df_in.columns[j])) for j in range(0, num_inputs)]

  # forecast sequence (t+1, ... t+n)
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
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def normalize_data(df, feat_range = (-1,1), csv_to_save=None, scaler_to_save=None):
  """Normaliza los datos utilizando un scaler
  Args:
      df: DataFrame origen
      feat_range : Rango de normalización
      csv_to_save: Archivo para guardar csv normalizado (None)
      scaler_to_save: Archivo para guardar el scaler (None)
  Returns:
      Dataframe normalizado
      Scaler utilizado
  """
  sts_src = df.copy()
  sts_src_values = sts_src.values
  sts_src_values = sts_src_values.astype('float32')
  scaler = preprocessing.MinMaxScaler(feature_range=feat_range)
  sts_src_scaled_values = scaler.fit_transform(sts_src_values)
  sts_scaled = pd.DataFrame(data=sts_src_scaled_values, columns=sts_src.columns, index=sts_src.index)        
  if csv_to_save is not None:
    sts_scaled.to_csv(csv_to_save, sep=';')
  if scaler_to_save is not None:
    joblib.dump(scaler, scaler_to_save) 
  return sts_scaled,scaler

  

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def denormalize_data(df, scaler):
  """DeNormaliza los datos utilizando un scaler
  Args:
      df: DataFrame origen
      scaler : Scaler de de-normalización
  Returns:
      Dataframe denormalizado
  """
  sts_src = df.copy()
  sts_src_values = sts_src.values
  sts_src_values = sts_src_values.astype('float32')
  sts_src_unscaled_values = scaler.inverse_transform(sts_src_values)
  sts_unscaled = pd.DataFrame(data=sts_src_unscaled_values, columns=sts_src.columns, index=sts_src.index)        
  return sts_unscaled

  

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def prepare_training_data(df, in_feats, ratio):
  """Crea los conjuntos de entrenamiento-validación-test y sus pares XY
    Args:
      df: DataFrame origen con los datos normalizados
      in_feats : Entradas
      out_feats: Salidas
      ratio  : Ratio de división de entrenamiento - test
  Returns:
      Pares XY de: train, test
  """
  train_len = int(df.shape[0] * ratio)
  dfxy = df.copy()
  x_train = dfxy.values[:train_len,:in_feats]
  y_train = dfxy.values[:train_len,in_feats:]
  x_test = dfxy.values[train_len:,:in_feats]
  y_test = dfxy.values[train_len:,in_feats:]  
  return x_train, y_train, x_test, y_test



#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def build_lstm_net(num_inputs, lbw, num_outputs, fww, nll, ndl, nlc, ndc, wfile, loss='mae', metrics=['accuracy'], verbose=0):
  """Crea la red neuronal tipo LSTM
    Args:
      num_inputs : Número de entradas
      lbw : timesteps del pasado
      num_outputs : Número de salidas
      fww : timesteps a predecir
      nll : Número de capas LSTM
      ndl : Número de capas Dense
      nlc : Número de neuronas por capa lstm
      ndc : Número de neuronas por capa Dense
      wfile : Archivo de pesos
      loss: Tipo de pérdida utilizada. Por defecto 'mae' ya que he visto que funciona mejor que 'mse'
      metrics : Tipo de métrica utilizada
      verbose : Muestra un resumen si > 0
    Returns:
      model,callback_list
  """
  # Inicio definiendo un modelo secuencial
  model = Sequential()

  sequence_flag = False
  if nll > 1:
      sequence_flag = True

  # capa de entrada, debe especificar formato 'input_shape'
  model.add(LSTM(nlc, return_sequences=sequence_flag, input_shape=(lbw, num_inputs)))
  #model.add(BatchNormalization())
  model.add(Dropout(0.20))

  for i in range(1, nll, 1):
      if i == nll-1:
        sequence_flag = False
      # capas intermedias
      model.add(LSTM(nlc, return_sequences=sequence_flag))
      #model.add(BatchNormalization())
      model.add(Dropout(0.20))

  for i in range(1, ndl, 1):
      model.add(Dense(ndc, activation='linear'))    
      
  # la capa de salida es una capa Dense con tantas salidas como timesteps a predecir con activación lineal     
  model.add(Dense(num_outputs * fww, activation='linear'))

  # compilo con optimizador Adam y pérdida 'mse'
  model.compile(optimizer='adam', loss=loss, metrics=metrics)   

  checkpoint = ModelCheckpoint(wfile, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
  callbacks_list = [checkpoint]
  
  # si existe un modelo previo, lo carga
  if wfile is not None:
      try:
          model.load_weights(wfile)
          if verbose > 0:
            print('Loaded weights from file: ', wfile)
      except:
          if verbose > 0:
            print('No weights file to load')

  if verbose > 0:
    model.summary()
  return model, callbacks_list
  

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def fit(model, x_train, y_train, num_inputs, num_in_steps, num_epochs, callbacks, batchsize, val_ratio=0, shuffle=False, plot_results=False, verbose=0):
  """Realiza el proceso de entrenamiento y validación
  Args:
      model : Modelo a entrenar
      x_train : Features de entrada
      y_train : Features de salida
      num_inputs : Nº de entradas
      num_in_steps  : Nº de timesteps pasados
      num_epochs    : Nº de épocas de entrenamiento
      callbacks     : Lista de callbacks
      batchsize     : Tamaño del batch
      val_ratio     : Tamaño de la partición de validación
      shuffle       : Flag para barajar los pares
      plot_result   : Flag para visualizar el resultado
      verbose       : 0, 1, 2
  Returns:
      History result
  """
  try:
    x = x_train.reshape(x_train.shape[0], num_in_steps, num_inputs)
    y = y_train
    history = None
    if val_ratio != 0:
      history = model.fit(x, y, epochs=num_epochs, batch_size=batchsize, callbacks=callbacks, validation_split=val_ratio, verbose=verbose, shuffle=shuffle) 
    else:
      history = model.fit(x, y, epochs=num_epochs, batch_size=batchsize, callbacks=callbacks, verbose=verbose, shuffle=shuffle) 
    if plot_results:
      # visualizo el resultado de la ejecución de la celda actual
      plt.subplot(1,2,1)
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.legend(['loss', 'val_loss'], loc='upper right')
      plt.subplot(1,2,2)
      plt.plot(history.history['acc'])
      plt.plot(history.history['val_acc'])
      plt.legend(['acc', 'val_acc'], loc='upper right')
    return history
  except:
    print('Model fit Exception:', sys.exc_info()[0])
    return None
  
        

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def test_evaluation(model, x_test, y_test, num_inputs, num_in_steps):
  """ Evalúa con la partición de test 
      Args:
      model : Modelo a testear
      x_test : Conjunto de entradas
      y_test : Conjunto de salidas objetivo
      num_inputs : Nº de entradas
      num_in_steps : Nº de steps pasados a tener en cuenta
      Returns:
      scores = [loss, acc] o [loss] o [acc], etc...
  """
  scores = model.evaluate(x_test.reshape(x_test.shape[0], num_in_steps, num_inputs), y_test, batch_size=1, verbose=2)
  return scores



#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def build_random_forest_regressor(x_train, y_train, parameter_grid, cv=5, n_jobs=4, verbose=1): 
  """ Crea un RandomForestRegressor a partir de un grid paramétrico con validación cruzada
      Args:
      x_train : Entradas del Conjunto de entrenamiento [num_samples, num_features]
      y_train : Salidas del Conjunto de entrenamiento [num_samples, num_outputs]
      parameter_grid : Dict con las opciones de búsqueda de hyperparámetros. Ej:
                {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [100, 50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                }
      cv : Número de modelos en la evaluación cruzada
      n_jobs : Número de procesos en paralelo para realizar la búsqueda
      verbose : Indicador de logging
  """
  # Configuro la búsqueda de hyperparámetros
  model = GridSearchCV(RandomForestRegressor(), parameter_grid, cv=cv, n_jobs=n_jobs, verbose=verbose)

  # Inicio la búsqueda
  model.fit(x_train, y_train)
  if verbose > 0:
    print('Best score: {}'.format(model.best_score_))
    print('Best parameters: {}'.format(model.best_params_))       
  return model  


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def compute_score(classifier, X, y, scoring='accuracy'):
  """ Computa el score de un clasificador mediante la librería scikit-learn
  """
  xval = cross_val_score(classifier, X, y, cv = 5, scoring=scoring)
  return np.mean(xval)        


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def test_rmse(model, x_test, y_test, num_inputs, num_in_steps, num_outputs, num_out_steps, scaler, cb_out_builder, plot_results=False):
  """Realiza el proceso de test
  Args:
      model : Modelo a testear
      x_test : Conjunto de entradas
      y_test : Conjunto de salidas objetivo
      num_inputs : Nº de entradas
      num_in_steps : Nº de steps pasados a tener en cuenta
      num_outputs : Nº de salidas
      num_out_steps : Nº de steps a predecir
      scaler : scaler para deshacer el scaling
      cb_out_builder : Callback para generar salidas reales y, yhat
      plot_results: Visualiza resultados
  Returns:
      scores = [loss,acc], target, pred
  """
  x = x_test.reshape(x_test.shape[0], num_in_steps, num_inputs)
  y = y_test
  scores = model.evaluate(x, y, batch_size=1, verbose=2)
  print('Model Loss: ', scores[0])  
  print("Model Accuracy: ",scores[1])
  x_data = x
  y_data = y

  rmse = list()
  target,pred=list(),list()
  for sample in range(x_data.shape[0]):
    # Realizo predicción del primer conjunto de datos
    x = x_data[sample].reshape(1, num_in_steps, num_inputs)
    y = y_data[sample].reshape(1, num_out_steps * num_outputs)
    predictions = model.predict(x, batch_size=1,verbose=0)

    # deshago el scaling
    xy_values = np.concatenate((x.reshape(1, x.shape[1]*x.shape[2]), y),axis=1)
    xy_values = scaler.inverse_transform(xy_values)
    xyhat_values = np.concatenate((x.reshape(1, x.shape[1]*x.shape[2]), predictions),axis=1)
    xyhat_values = scaler.inverse_transform(xyhat_values)
    
    # Calculo el error RMSE
    yvalues,yhat_values = cb_out_builder(xy_values[0], xyhat_values[0])
    rmse_val = math.sqrt(sk.mean_squared_error(yvalues, yhat_values, multioutput = 'uniform_average'))
    rmse.append(rmse_val)
    target.append(yvalues)
    pred.append(yhat_values)    
  
  if plot_results:        
    plt.plot(np.asarray(rmse))    
    plt.legend(['RMSE'])    
  return scores, target, pred, rmse


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def visualize_input(df, row, num_inputs, num_in_steps):
  """ Visualiza una entrada de una predicción dada
  """  
  values = df.values[row].ravel()
  plt.figure(figsize=(12,36))
  for i in range(num_inputs):
    legend = df.columns[i]
    data = []
    for j in range(num_in_steps):
      data.append(values[i+(j*num_inputs)])
    plt.subplot(num_inputs, 1, i+1)
    plt.plot(data)
    plt.title(legend.split('(')[0], y=0.5, loc='right')

  

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def plot_heatmap(title, df, wh):
  fig = plt.figure(figsize = wh)
  ax = fig.add_subplot(111)
  cax = ax.matshow(df.corr(), vmin=-1, vmax=1)
  fig.colorbar(cax)
  ticks = np.arange(0,len(df.columns),1)
  ax.set_xticks(ticks)
  ax.set_yticks(ticks)
  ax.set_xticklabels(df.columns)
  ax.set_yticklabels(df.columns)
  for i in range(len(df.columns)):
    for j in range(len(df.columns)):
        text = ax.text(j, i, "{:.4g}".format(df.corr().values[i, j]), ha="center", va="center", color="black")

    
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def plot_feature(df, features, row, n_in):
  """ Visualiza n_in timesteps de un feature en un dataframe y para una muestra n
      Args:
      df : Dataframe que contiene el feature
      row: Muestra que visualizar
      n_in:Número de timesteps que se utiliza en la variable
  """
  array_of_values = []
  for f in features:    
    values = []  
    for i in range(n_in):    
      col = f+'(t'+str(-n_in+i)+')'
      values.append(df.iloc[row][col])
    values.append(df.iloc[row][f+'(t)'])
    array_of_values.append(values)
    plt.plot(values)
  plt.legend(features)
  return array_of_values
    

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def solve_outlier(x, mean, sd):
  res = x
  if x > (mean + (2 * sd)):
    res = mean + (2 * sd)
  elif x < (mean - (2 * sd)):
    res = mean - (2 * sd)
  return res        