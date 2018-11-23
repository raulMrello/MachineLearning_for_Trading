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
tls.set_credentials_file(username='raulMrello', api_key='qX9S30YRQ866mGF9b89u')

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
def load_hist(filename, **kwargs):
  """Carga un archivo csv con datos train-validate-test
    Args:
      filename: Archivo de datos csv
      kwargs: Lista de parámetros aceptados:
        sep: tipo de separador (,)
        remove_weekends: Flag para borrar datos del fin de semana sin cotización
        remove_index: Flag para borrar el índice y obtener el timestamp como una nueva columna
  Returns:
    Dataframe
  """
  sep = ','
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

  df = pd.read_csv(filename, sep=sep) 
  # crea una nueva columna de timestamp uniendo date & time y crea columnas para el día de la semana, el día del mes, el mes y la hora del día actual 
  df['timestamp'] = df['DATE'] + '  ' + df['TIME'] 
  df['timestamp'] = df['timestamp'].map(lambda x: datetime.datetime.strptime(x, '%Y.%m.%d %H:%M:%S'))  
  df['timestamp'] = pd.to_datetime(df['timestamp'])
  df.set_index('timestamp', inplace=True)
  df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="1H"))
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

  df['IND_SAR'] = talib.SAR(df['HIGH'], df['LOW'], acceleration=0, maximum=0)
  cols.append('IND_SAR')  
    
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
  """ Agrega todos los identificadores de patrones de velas de la librería TALIB
  """
  df = df.copy()
  df['CDL2CROWS'] = talib.CDL2CROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDL3INSIDE'] = talib.CDL3INSIDE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(df.OPEN, df.HIGH, df.LOW, df.CLOSE, 0)
  df['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLBELTHOLD'] = talib.CDLBELTHOLD(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
  df['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
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
  df['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(df.OPEN, df.HIGH, df.LOW, df.CLOSE)
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
def normalize_data(df, in_cols_to_scale, out_cols_to_scale, csv_to_save=None, scaler_to_save=None):
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
def solve_outlier(x, mean, sd):
  res = x
  if x > (mean + (2 * sd)):
    res = mean + (2 * sd)
  elif x < (mean - (2 * sd)):
    res = mean - (2 * sd)
  return res        