#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  FuzzyMarketState is a python class that represents the current market state
  conditions by using Fuzzy Logic semanthics.

  In order to achieve the commented output, it uses several technical indicators
  to build sinthetic variables, that afterwards will fuzzify into more abstract
  definition of the market status.

  Technical indicators
  ====================

  - Zigzag
  - Bollinger
  - MACD
  - RSI
  - Fibonacci
  - Moving averages
  - Supports & Resistances
  - Dynamic channels

  Sinthetic Fuzzy variables
  =========================

  - Proximiy of price to:
    - Relevant SMA
    - Specific SMA
    - Relevant support/resistance
    - Relevant fibo level
    - Specific fibo level
    - Current dynamic resistance
    - Current dynamic support
  
  - Force of Patterns:
    - Divergence (regular|hidden, bullish|bearish)
    - Trend (bullish|bearish)
    - Candlestick (bullish|bearish)
"""

####################################################################################
# Librerías de manejo de datos 
import pandas as pd
import numpy as np
import skfuzzy.control as ctrl

####################################################################################
# Librerías de visualización
import plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
from plotly.tools import FigureFactory as FF
import plotly.tools as tls

####################################################################################
# TA-Lib: instalación y carga de la librería
import talib
from ZIGZAG_Signal_Listener import ZIGZAG_Signal_Listener

####################################################################################
# Fuzzy logic libs
from FuzzyLib import Fuzzifier, FuzzyVar



####################################################################################
# Otras utilidades
import datetime
import time
import os
import sys
import math
import logging



####################################################################################
####################################################################################
####################################################################################

class FuzzyMarketState():
  def __init__(self, level=logging.WARN):
    self.__logger = logging.getLogger(__name__)
    self.__logger.setLevel(level)
    self.__logger.info('Created!')
    self.__zigzag = ZIGZAG_Signal_Listener(level)

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def setLoggingLevel(self, level): 
    """ Updates loggging level

      Keyword arguments:
        level -- new logging level
    """   
    self.__logger.setLevel(level)
    self.__logger.debug('Logging level changed to {}'.format(level))

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def loadCSV(self, file, sep=';'):    
    """ Loads a csv file imported from the trading platform with this
        columns: DATE[,TIME],OPEN,HIGH,LOW,CLOSE,TICKVOL,VOL,SPREAD.
        DATE and TIME are converted to datetime as new column TIME, and
        index is reset to start from sample 0.
      
      Keyword arguments:
        file -- csv file
        sep -- character separator (default ';')
      Returns:
        num_rows -- number of rows loaded        
    """
    _df = pd.read_csv(file, sep)
    if 'TIME' in _df.columns:
      _df['TIME'] = _df['DATE'] + '  ' + _df['TIME'] 
    else:
      _df['TIME'] = _df['DATE'] + '  00:00:00'
    _df['TIME'] = _df['TIME'].map(lambda x: datetime.datetime.strptime(x, '%Y.%m.%d %H:%M:%S'))  
    _df['TIME'] = pd.to_datetime(_df['TIME'])
    self.__df = _df.drop(columns=['DATE'])
    self.__logger.debug('loaded {} rows from {} to {}'.format(self.__df.shape[0], _df['TIME'].iloc[0], _df['TIME'].iloc[-1]))
    return self.__df

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def loadDataframe(self, df):    
    """ Loads a dataframe
      
      Keyword arguments:
        df -- dataframe
    """
    self.__df = df.copy()

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def getDataframe(self):    
    """ gets a reference to the internal dataframe
      
      Returns:
        self.__df -- dataframe
    """
    return self.__df

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildIndicators(self, params = dict(), level=None):    
    """ Builds default integrated indicators for all rows in integrated dataframe, such as: 
        - Zigzag (includes BollingerBands and derivatives)
        - MACD
        - RSI
        - MovingAverages (SMA, EMA)
 
        Also builds Fuzzy indicators based on previous ones, such as:
        - Price vs Overbought level:  [Far, Near, In]
        - Price vs Oversell level: [Far, Near, In]
        - Trend: [HardBearish, SoftBearish, NoTrend, SoftBullish, HardBullish]
        - Divergence: [HardBearish, SoftBearish, NoDivergence, SoftBullish, HardBullish]
        - Fibo retracements 23%, 38%, 50%, 61%: [Far, Near, In]
        - Fibo extensions_1 123%, 138%, 150%, 161%: [Far, Near, In]
        - Fibo extensions_2 223%, 238%, 250%, 261%: [Far, Near, In]
        - MovingAverage proximity: [Far, Near, In]
        - SupportResistance proximity: [Far, Near, In]
        - DynamicTrendLine proximity: [Far, Near, In]
        - Candlestick patterns: [NotPresent, Soft, Hard]
      
      Keyword arguments:
        params -- dictionary with configuration parameters for indicators (default None), such us:

      Return:
        self.__df -- Internal dataframe with all indicators
     """
    prev_level = self.__logger.level
    if level is not None:
      self.__logger.setLevel(level)

    # build zigzag indicator (includes BollingerBands and derivatives)   
    minbars   = params['zz_minbars'] if 'zz_minbars' in params.keys() else 12  
    bb_period = params['bb_period'] if 'bb_period' in params.keys() else 20 
    bb_dev    = params['bb_dev'] if 'bb_dev' in params.keys() else 2.0
    nan_value = params['zz_nan_value'] if 'zz_nan_value' in params.keys() else 0.0 
    _df = self.buildZigzag(self.__df, minbars, bb_period, bb_dev, nan_value)

    # build oscillators (includes MACD and RSI)
    macd_applied  = params['macd_applied'] if 'macd_applied' in params.keys() else 'CLOSE'
    macd_fast     = params['macd_fast'] if 'macd_fast' in params.keys() else 12 
    macd_slow     = params['macd_slow'] if 'macd_slow' in params.keys() else 26
    macd_sig      = params['macd_sig'] if 'macd_sig' in params.keys() else 9  
    rsi_applied   = params['rsi_applied'] if 'rsi_applied' in params.keys() else 'CLOSE'
    rsi_period    = params['rsi_period'] if 'rsi_period' in params.keys() else 14     
    self.buildOscillators(_df, macd_applied, macd_fast, macd_slow, macd_sig, rsi_applied, rsi_period)

    # build 3 moving averages (includes SMA50, SMA100, SMA200)
    ma_fast_applied = params['ma_fast_applied'] if 'ma_fast_applied' in params.keys() else 'CLOSE'
    ma_fast_period  = params['ma_fast_period'] if 'ma_fast_period' in params.keys() else 50 
    ma_fast_type    = params['ma_fast_type'] if 'ma_fast_type' in params.keys() else 'SMA'
    ma_mid_applied  = params['ma_mid_applied'] if 'ma_mid_applied' in params.keys() else 'CLOSE'
    ma_mid_period   = params['ma_mid_period'] if 'ma_mid_period' in params.keys() else 100 
    ma_mid_type     = params['ma_mid_type'] if 'ma_mid_type' in params.keys() else 'SMA'
    ma_slow_applied = params['ma_slow_applied'] if 'ma_slow_applied' in params.keys() else 'CLOSE'
    ma_slow_period  = params['ma_slow_period'] if 'ma_slow_period' in params.keys() else 200 
    ma_slow_type    = params['ma_slow_type'] if 'ma_slow_type' in params.keys() else 'SMA'
    ma_trend_filters=params['ma_trend_filters'] if 'ma_trend_filters' in params.keys() else {'price-slow': 0.5, 'price-mid': 0.3, 'price-fast': 0.2}
    self.build3MovingAverages(_df, 
                              ma_fast_applied, ma_fast_period, ma_fast_type,
                              ma_mid_applied, ma_mid_period, ma_mid_type,
                              ma_slow_applied, ma_slow_period, ma_slow_type,
                              ma_trend_filters)


    # build fibonacci retracement and extensions
    nan_value = params['fibo_nan_value'] if 'fibo_nan_value' in params.keys() else 0.0
    fibo_level = params['fibo_level'] if 'fibo_level' in params.keys() else nan_value 
    self.buildCommonFiboLevels(_df, nan_value)    

    # build support and resistances based on previous zigzags
    self.buildSupports(_df)
    self.buildResistances(_df)

    # build dynamic support-resistance of channel based on previous zigzags
    nan_value = params['channel_nan_value'] if 'channel_nan_value' in params.keys() else 'no-channel' 
    self.buildChannel(_df, nan_value)    

    # build trend detector based on different indicators
    nan_value = params['trend_nan_value'] if 'trend_nan_value' in params.keys() else 0.0
    trend_filters=params['trend_filters'] if 'trend_filters' in params.keys() else {'sma-trend':0.75, 'zigzag-trend':0.15, 'fibo-trend':0.1}
    self.buildTrends(_df, trend_filters, nan_value)

    # build divergence detector based on zigzag, macd and rsi
    nan_value = params['div_nan_value'] if 'div_nan_value' in params.keys() else 0.0
    self.buildDivergences(_df, nan_value)    
    
    # remove NaN values and reindex from sample 0
    _df.dropna(inplace=True)
    _df.reset_index(drop=True, inplace=True)

    # restore logging level
    if level is not None:
      self.__logger.setLevel(prev_level)

    # updates and return current dataframe
    self.__df = _df
    return self.__df
  

  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildZigzag(self, df, minbars, bb_period, bb_dev, nan_value, dropna=True):
    """ Builds zigzag indicator

      Keyword arguments:
        df -- dataframe
        zz_minbars -- Min. number of bars per flip to avoid discarding (default 12)
        bb_period -- Bollinger bands period (default 20)
        bb_dev -- Bollinger bands deviation (default 2.0)
        nan_value -- Values for zigzag indicator during search phase (default 0.0)
        zlevel -- logging level (default WARN)
      Return:
        _df -- Copy dataframe from df with zigzag columns
    """

    _df, _=  self.__zigzag.ZIGZAG(df, 
                                  minbars   = minbars,
                                  bb_period = bb_period,
                                  bb_dev    = bb_dev,
                                  nan_value = nan_value,
                                  dropna    = dropna,
                                  level     = self.__logger.level)
    # add columns for trend detection using zigzag
    _df['ZZ_BULLISH_TREND'] = _df.apply(lambda x: 1 if x.P1 > x.P3 and x.P3 > x.P5 and x.P2 > x.P4 else 0, axis= 1)
    _df['ZZ_BEARISH_TREND'] = _df.apply(lambda x: 1 if x.P1 < x.P3 and x.P3 < x.P5 and x.P2 < x.P4 else 0, axis= 1)
    return _df

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildOscillators(self, df, macd_applied, macd_fast, macd_slow, macd_sig, rsi_applied, rsi_period, use_talib=True):
    """ Builds different oscillators like MACD and RSI

      Keyword arguments:
        df -- dataframe
        macd_applied -- Price to apply MACD (default 'CLOSE')
        macd_fast -- MACD fast period (default 12)
        macd_slow -- MACD slow period (default 26)
        macd_sig -- MACD signal period (default 9)
        rsi_applied -- Price to apply RSI (default 'CLOSE')
        rsi_period -- RSI period (default 14)
      Return:
        dict -- Dictionary with Macd and RSI common series (main, signals, ...)
    """
    if not use_talib:
      exp1 = df[macd_applied].ewm(span=macd_fast, adjust=False).mean()
      exp2 = df[macd_applied].ewm(span=macd_slow, adjust=False).mean()
      _macd_main = exp1-exp2
      _macd_sig = macd.ewm(span=macd_sig, adjust=False).mean()
    else:
      _macd_main, _macd_sig, _macd_hist = talib.MACD(df[macd_applied], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_sig)
    # add crossovers between main and zero level
    _macd_cross_zero_up = ((_macd_main > 0) & (_macd_main.shift(1) < 0 | ((_macd_main.shift(1)==0) & (_macd_main.shift(2) < 0))))
    _macd_cross_zero_dn = ((_macd_main < 0) & (_macd_main.shift(1) > 0 | ((_macd_main.shift(1)==0) & (_macd_main.shift(2) > 0))))
    # add crossovers between main and signal lines
    _macd_cross_sig_up = (((_macd_main > _macd_sig) & ((_macd_main.shift(1) < _macd_sig.shift(1)) | ((_macd_main.shift(1)==_macd_sig.shift(1)) & (_macd_main.shift(2) < _macd_sig.shift(2))))) & ((_macd_main < 0) & (_macd_main.shift(1) < 0) & (_macd_main.shift(2) < 0)))
    _macd_cross_sig_dn = (((_macd_main < _macd_sig) & ((_macd_main.shift(1) > _macd_sig.shift(1)) | ((_macd_main.shift(1)==_macd_sig.shift(1)) & (_macd_main.shift(2) > _macd_sig.shift(2))))) & ((_macd_main > 0) & (_macd_main.shift(1) > 0) & (_macd_main.shift(2) > 0)))


    _rsi = talib.RSI(df[rsi_applied], timeperiod=rsi_period)
    # add crossovers between overbought levels
    _rsi_cross_ob_up = ((_rsi > 70) & (_rsi.shift(1) < 70 | ((_rsi.shift(1)==70) & (_rsi.shift(2) < 70))))
    _rsi_cross_ob_dn = ((_rsi < 70) & (_rsi.shift(1) > 70 | ((_rsi.shift(1)==70) & (_rsi.shift(2) > 70))))
    # add crossovers between oversell levels
    _rsi_cross_os_up = ((_rsi > 30) & (_rsi.shift(1) < 30 | ((_rsi.shift(1)==30) & (_rsi.shift(2) < 30))))
    _rsi_cross_os_dn = ((_rsi < 30) & (_rsi.shift(1) > 30 | ((_rsi.shift(1)==30) & (_rsi.shift(2) > 30))))

    df['MACD_main']           = _macd_main
    df['MACD_sig']            = _macd_sig
    df['MACD_hist']           = _macd_hist
    df['MACD_CROSS_ZERO_UP']  = _macd_cross_zero_up
    df['MACD_CROSS_ZERO_DN']  = _macd_cross_zero_dn
    df['MACD_CROSS_SIG_UP']   = _macd_cross_sig_up
    df['MACD_CROSS_SIG_DN']   = _macd_cross_sig_dn
    df['RSI']                = _rsi
    df['RSI_cross_ob_up']    = _rsi_cross_ob_up
    df['RSI_cross_ob_dn']    = _rsi_cross_ob_dn
    df['RSI_cross_os_up']    = _rsi_cross_os_up
    df['RSI_cross_os_dn']    = _rsi_cross_os_dn

    return {'macd': {
              'main': _macd_main, 
              'sig': _macd_sig, 
              'hist': _macd_hist,
              'cross_zero_up' : _macd_cross_zero_up,
              'cross_zero_dn' : _macd_cross_zero_dn,
              'cross_sig_up' : _macd_cross_sig_up,
              'cross_sig_dn' : _macd_cross_sig_dn
            }, 
            'rsi':{
              'main': _rsi,
              'cross_ob_up' : _rsi_cross_ob_up,
              'cross_ob_dn' : _rsi_cross_ob_dn,
              'cross_os_up' : _rsi_cross_os_up,
              'cross_os_dn' : _rsi_cross_os_dn
            }
            }

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def build3MovingAverages( self, df, 
                            ma_fast_applied, ma_fast_period, ma_fast_type, 
                            ma_mid_applied, ma_mid_period, ma_mid_type, 
                            ma_slow_applied, ma_slow_period, ma_slow_type,
                            trend_filters={'price-slow': 0.5, 'price-mid': 0.3, 'price-fast': 0.2}):
    """ Builds different moving averages according with the type and periods

      Keyword arguments:
        df -- dataframe
        ma_x_applied -- Price to apply MA
        ma_x_type -- MA type 
        ma_x_period -- MA period
        trend_filter -- List of Filters to calculate trend strength
          filter: {'price-slow': 0.5} -- This means that price above slow ma gives a strength of +0.5 bullish-trend
                                          and price below slow ma gives a strength of +0.5 bearish-trend
      Return:
        dict -- Dictionary with all SMA series:
          sma_fast
          sma_mid
          sma_slow
          sma_bullish_trend
          sma_bearish_trend
    """
    if ma_fast_type == 'EMA':
      ma_fast = talib.EMA(df[ma_fast_applied], timeperiod=ma_fast_period)
    else:
      ma_fast = talib.SMA(df[ma_fast_applied], timeperiod=ma_fast_period)
    if ma_mid_type == 'EMA':
      ma_mid = talib.EMA(df[ma_mid_applied], timeperiod=ma_mid_period)
    else:
      ma_mid = talib.SMA(df[ma_mid_applied], timeperiod=ma_mid_period)
    if ma_slow_type == 'EMA':
      ma_slow = talib.EMA(df[ma_slow_applied], timeperiod=ma_slow_period)
    else:
      ma_slow = talib.SMA(df[ma_slow_applied], timeperiod=ma_slow_period)
    
    df['SMA_FAST']  = ma_fast
    df['SMA_MID']   = ma_mid
    df['SMA_SLOW']  = ma_slow
    df['SMA_BULLISH_TREND']  = 0.0
    df['SMA_BEARISH_TREND']  = 0.0

    def fn_strength(x, df):
      bull_strength = 0.0
      bear_strength = 0.0
      if x.LOW > x.SMA_SLOW:
        bull_strength = bull_strength + trend_filters['price-slow']
        if x.LOW > x.SMA_MID:
          bull_strength = bull_strength + trend_filters['price-mid']
        if x.LOW > x.SMA_FAST:
          bull_strength = bull_strength + trend_filters['price-fast']
      elif x.HIGH < x.SMA_SLOW:
        bear_strength = bear_strength + trend_filters['price-slow']
        if x.HIGH < x.SMA_MID:
          bear_strength = bear_strength + trend_filters['price-mid']
        if x.HIGH < x.SMA_FAST:
          bear_strength = bear_strength + trend_filters['price-fast']
      df.at[x.name, 'SMA_BULLISH_TREND']  = bull_strength
      df.at[x.name, 'SMA_BEARISH_TREND']  = bear_strength

    df.apply(lambda x: fn_strength(x, df), axis=1)    

    return {'sma_fast': df['SMA_FAST'], 
            'sma_mid': df['SMA_MID'], 
            'sma_slow': df['SMA_SLOW'],
            'sma_bullish_trend': df['SMA_BULLISH_TREND'],
            'sma_bearish_trend': df['SMA_BEARISH_TREND']}

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildCommonFiboLevels(self, df, nan_value):
    self.buildFiboLevel(df, 'FIBO_CURR', nan_value)
    self.buildFiboLevel(df, 'FIBO_023', nan_value, fibo_level=0.236)
    self.buildFiboLevel(df, 'FIBO_038', nan_value, fibo_level=0.382)
    self.buildFiboLevel(df, 'FIBO_050', nan_value, fibo_level=0.500)
    self.buildFiboLevel(df, 'FIBO_061', nan_value, fibo_level=0.618)
    self.buildFiboLevel(df, 'FIBO_078', nan_value, fibo_level=0.786)
    self.buildFiboLevel(df, 'FIBO_123', nan_value, fibo_level=1.236)
    self.buildFiboLevel(df, 'FIBO_138', nan_value, fibo_level=1.382)
    self.buildFiboLevel(df, 'FIBO_150', nan_value, fibo_level=1.500)
    self.buildFiboLevel(df, 'FIBO_161', nan_value, fibo_level=1.618)
    self.buildFiboLevel(df, 'FIBO_178', nan_value, fibo_level=1.786)

  def buildFiboLevel(self, df, name, nan_value, fibo_level=0.0):
    """ Builds fibo level depending on zigzag points

      Keyword arguments:
        df -- dataframe
        name -- name of the column to build
        nan_value -- NaN value for empty results
        fibo_level -- Fibo level to calculate (default 0.0)
      Return:
        fibo -- Fibo level 
    """
    def fibo_retr(x, df, nan_value, fibo_level):
      value = x.ZIGZAG if x.ZIGZAG != nan_value else x.HIGH if x.P1 < x.P2 else x.LOW
      try:
        if x.P1 > x.P2:
          if fibo_level == 0.0:
            return (x.P1 - value)/(x.P1 - x.P2)
          else:
            return (x.P1 - ((x.P1 - x.P2)*fibo_level))
        else:
          if fibo_level == 0.0:
            return (value - x.P1)/(x.P2 - x.P1)
          else:
            return (x.P1 + ((x.P2 - x.P1)*fibo_level))
      except ZeroDivisionError:
        return nan_value
   
    fibo = df.apply(lambda x: fibo_retr(x, df, nan_value, fibo_level), axis=1)  
    df[name] = fibo
    return {'fibo': fibo} 

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildSupports(self, df, nan_value = 0.0):
    """ Builds validated supports through a 4-point zigzag

      Keyword arguments:
        df -- dataframe
      Return:
        supports -- Serie
    """    
    def fn_support(x, df):
      if x.ZIGZAG > x.P2 and x.P2 > x.P1 and  x.P1 > x.P3:
        df.at[x.P1_idx, 'SUPPORTS'] = x.P1
    
    df['SUPPORTS'] = nan_value
    df.apply(lambda x: fn_support(x, df), axis=1)  
    return df['SUPPORTS']   

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildResistances(self, df, nan_value = 0.0):
    """ Builds validated resistances through 4-point zigzag

      Keyword arguments:
        df -- dataframe        
      Return:
        resistances -- Serie
    """
    def fn_resistance(x, df):
      if x.ZIGZAG < x.P2 and x.P2 < x.P1 and  x.P1 < x.P3:
        df.at[x.P1_idx, 'RESISTANCES'] = x.P1      
    
    df['RESISTANCES'] = 0.0
    df.apply(lambda x: fn_resistance(x, df), axis=1)  
    return df['RESISTANCES']   

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildChannel(self, df, nan_value):
    """ Builds 2 main channel limits based on current zigzag trend points

      Keyword arguments:
        df -- dataframe
        nan_value -- NaN value for empty results
      Return:
        ch_up, ch_dwon -- Two main channel lines
    """    
    def fn_channel(x, df, level, nan_value):
      def line(x0,x1,y0,y1,x):
        return (((y1-y0)/(x1-x0))*(x-x0))+y0

      if x.ZZ_BULLISH_TREND == 1 or x.ZZ_BEARISH_TREND == 1:
        if x.P1 > x.P2:
          if level == 1:
            return "P3,P1"
          else:
            return "P4,P2"
        else:
          if level == 1:
            return "P4,P2"
          else:
            return "P3,P1"
      return nan_value      
      
    df['CHANNEL_UPPER_LIMIT'] = df.apply(lambda x: fn_channel(x, df, 1, nan_value), axis=1)  
    df['CHANNEL_LOWER_LIMIT'] = df.apply(lambda x: fn_channel(x, df, 2, nan_value), axis=1)  
    return {'channel_upper_limit': df['CHANNEL_UPPER_LIMIT'], 'channel_lower_limit': df['CHANNEL_LOWER_LIMIT']}   

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildTrends(self, df, filters=[], nan_value=0.0):
    """ Builds a trend detector based on different indicators, as follows:
        ZIGZAG_TREND_DETECTOR: provides a trend feedback according with its last
        zigzag points.
        SMA_TREND_DETECTOR: provides trend feedback according with its fast-mid-slow
        moving averages
        FIBO_TREND_DETECTOR: provides trend feedback according with fibo retracements
        and extensions.

      Keyword arguments:
        df -- dataframe
        filters -- filters for trend strength: sma-trend, zigzag-trend, fibo-trend
        nan_value -- NaN value for empty results
        fibo_tol -- Tolerance % around fibo levels
      Return:
        up_trend, down_trend -- Series containing strength of each trend
    """    
    _fibo_tol = 0.04
    if not 'sma-trend' in filters:
      filters['sma-trend']=0.0
    if not 'zigzag-trend' in filters:
      filters['zigzag-trend']=0.0
    if not 'fibo-trend' in filters:
      filters['fibo-trend']=0.0
    
    def fn_trend(x, df, nan_value, fibo_tol, filters):
      bull_strength = (filters['sma-trend'] * x.SMA_BULLISH_TREND) + (filters['zigzag-trend'] * x.ZZ_BULLISH_TREND)
      bear_strength = (filters['sma-trend'] * x.SMA_BEARISH_TREND) + (filters['zigzag-trend'] * x.ZZ_BEARISH_TREND)
      if x.P1 > x.P2 and x.FIBO_CURR > (0.236-fibo_tol) and (x.FIBO_CURR < 0.618+fibo_tol):
        bull_strength += filters['fibo-trend']
      elif x.P1 < x.P2 and x.FIBO_CURR > (1.236-fibo_tol) and (x.FIBO_CURR < 1.618+fibo_tol):
        bull_strength += filters['fibo-trend']
      if x.P1 < x.P2 and x.FIBO_CURR > (0.236-fibo_tol) and (x.FIBO_CURR < 0.618+fibo_tol):
        bear_strength += filters['fibo-trend']
      elif x.P1 > x.P2 and x.FIBO_CURR > (1.236-fibo_tol) and (x.FIBO_CURR < 1.618+fibo_tol):
        bear_strength += filters['fibo-trend']
      df.at[x.name, 'BULLISH_TREND'] = bull_strength
      df.at[x.name, 'BEARISH_TREND'] = bear_strength
     
    df.apply(lambda x: fn_trend(x, df, nan_value, _fibo_tol, filters), axis=1)  
    return {'bullish': df['BULLISH_TREND'], 'bearish': df['BEARISH_TREND']}   


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildDivergences(self, df, nan_value):    
    """Builds divergences based on zigzag, macd and rsi

      Keyword arguments:
        df -- dataframe
        nan_value -- NaN value for empty results
        fibo_tol -- Tolerance % around fibo levels
      Return:
        regbulldiv, hidbulldiv, regbeardiv, hidbeardiv -- Series containing different divergences
    """
    # add result columns
    df['BULLISH_DIVERGENCE'] = 0.0
    df['BEARISH_DIVERGENCE'] = 0.0

    df['DIV_DOUB_REG_BEAR_MACD'] = 0.0
    df['DIV_REG_BEAR_MACD'] = 0.0
    df['DIV_DOUB_REG_BULL_MACD'] = 0.0
    df['DIV_REG_BULL_MACD'] = 0.0
    df['DIV_DOUB_HID_BEAR_MACD'] = 0.0
    df['DIV_HID_BEAR_MACD'] = 0.0
    df['DIV_DOUB_HID_BULL_MACD'] = 0.0
    df['DIV_HID_BULL_MACD'] = 0.0
    
    df['DIV_DOUB_REG_BEAR_RSI'] = 0.0
    df['DIV_REG_BEAR_RSI'] = 0.0
    df['DIV_DOUB_REG_BULL_RSI'] = 0.0
    df['DIV_REG_BULL_RSI'] = 0.0
    df['DIV_DOUB_HID_BEAR_RSI'] = 0.0
    df['DIV_HID_BEAR_RSI'] = 0.0
    df['DIV_DOUB_HID_BULL_RSI'] = 0.0
    df['DIV_HID_BULL_RSI'] = 0.0
     
    # executes divergence localization process:
    # 1. Set a default trend: requires 3 max and 3 min points
    # 1a.If max increasing or min increasing -> Bullish trend
    # 1b.If max decreasing or min decreasing -> Bearish trend
    # 1c.Else discard.

    def search(row, df, nan_value, logger):
      log = 'row [{}]: '.format(row.name)

      # check p1-p6 are valid
      if row.P6 == nan_value: 
        log += 'error-zzpoints-count'
        logger.debug(log)
        return

      # check if curr sample is max, min or the same as previous
      curr_is = 'unknown'
      # check curr sample is max 
      if row.P1 > row.P2:
        log += 'last is MAX '
        curr_is = 'max'

      #check if is min
      elif row.P1 < row.P2:
        log += 'last is MIN '
        curr_is = 'min'   

      # last 2 samples are equal, then finish
      else:
        log += 'error-no-minmax '
        logger.debug(log)
        return
      
      # at this point, exists a condition to evaluate.
      # Get idx of last 6 points (3 zigzags)
      p0_idx = row.name
      p1_idx = row.P1_idx
      p2_idx = row.P2_idx
      p3_idx = row.P3_idx
      p4_idx = row.P4_idx
      p5_idx = row.P5_idx
      p6_idx = row.P6_idx
      log += 'p0={}, p1={}, p2={}, p3={}, p4={}, p5={}, p6={} '.format(p0_idx, p1_idx, p2_idx, p3_idx, p4_idx, p5_idx, p6_idx)

      # set divergence type case
      class DivType():
        def __init__(self):
          self.enabled = False
          self.ifrom = 0
          self.idoubfrom = 0
          self.to = 0          
      reg_bull_div = DivType()
      reg_bear_div = DivType()
      hid_bull_div = DivType()
      hid_bear_div = DivType()

      if  pd.isna(df.MACD_main.iloc[row.P6_idx]) or pd.isna(df.RSI.iloc[row.P6_idx]):              
        log += 'error-P6-isNaN '
        logger.debug(log)
        return

      # Price check---
      # check regular-bearish-div
      if row.P1 > row.P3 and row.P2 > row.P4 and row.P1 > row.P2:
        reg_bear_div.enabled=True
        reg_bear_div.ifrom = row.P3_idx
        reg_bear_div.idoubfrom = row.P5_idx
        reg_bear_div.to   = row.P1_idx
        _rng1 = int((reg_bear_div.to - reg_bear_div.ifrom)/3)
        _rng2 = int((reg_bear_div.ifrom - reg_bear_div.idoubfrom)/3)
        if reg_bear_div.idoubfrom - _rng2 <= 0:
          reg_bear_div.enabled=False
      # other regular-bearish-div condition
      elif row.P2 > row.P4 and row.P3 > row.P5 and row.P1 < row.P2:
        reg_bear_div.enabled=True
        reg_bear_div.ifrom = row.P4_idx
        reg_bear_div.idoubfrom = row.P6_idx
        reg_bear_div.to   = row.P2_idx
        _rng1 = int((reg_bear_div.to - reg_bear_div.ifrom)/3)
        _rng2 = int((reg_bear_div.ifrom - reg_bear_div.idoubfrom)/3)
        if reg_bear_div.idoubfrom - _rng2 <= 0:
          reg_bear_div.enabled=False
      # check hidden-bullish-div
      if row.P1 > row.P3 and row.P2 > row.P4 and row.P1 < row.P2:
        hid_bull_div.enabled=True
        hid_bull_div.ifrom = row.P3_idx
        hid_bull_div.idoubfrom = row.P5_idx
        hid_bull_div.to   = row.P1_idx
        _rng1 = int((hid_bull_div.to - hid_bull_div.ifrom)/3)
        _rng2 = int((hid_bull_div.ifrom - hid_bull_div.idoubfrom)/3)
        if hid_bull_div.idoubfrom - _rng2 <= 0:
          hid_bull_div.enabled=False
      # other hidden-bullish-div
      elif row.P2 > row.P4 and row.P3 > row.P5 and row.P1 > row.P2:
        hid_bull_div.enabled=True
        hid_bull_div.ifrom = row.P4_idx
        hid_bull_div.idoubfrom = row.P6_idx
        hid_bull_div.to   = row.P2_idx
        _rng1 = int((hid_bull_div.to - hid_bull_div.ifrom)/3)
        _rng2 = int((hid_bull_div.ifrom - hid_bull_div.idoubfrom)/3)
        if hid_bull_div.idoubfrom - _rng2 <= 0:
          hid_bull_div.enabled=False

      # check regular-bullish-div
      if row.P1 < row.P3 and row.P2 < row.P4 and row.P1 < row.P2:
        reg_bull_div.enabled=True
        reg_bull_div.ifrom = row.P3_idx
        reg_bull_div.idoubfrom = row.P5_idx
        reg_bull_div.to   = row.P1_idx
        _rng1 = int((reg_bull_div.to - reg_bull_div.ifrom)/3)
        _rng2 = int((reg_bull_div.ifrom - reg_bull_div.idoubfrom)/3)
        if reg_bull_div.idoubfrom - _rng2 <= 0:
          reg_bull_div.enabled=False
      # other regular-bullish-div condition
      elif row.P2 < row.P4 and row.P3 < row.P5 and row.P1 > row.P2:
        reg_bull_div.enabled=True
        reg_bull_div.ifrom = row.P4_idx
        reg_bull_div.idoubfrom = row.P6_idx
        reg_bull_div.to   = row.P2_idx
        _rng1 = int((reg_bull_div.to - reg_bull_div.ifrom)/3)
        _rng2 = int((reg_bull_div.ifrom - reg_bull_div.idoubfrom)/3)
        if reg_bull_div.idoubfrom - _rng2 <= 0:
          reg_bull_div.enabled=False

      # check hidden-bearish-div
      if row.P1 < row.P3 and row.P2 < row.P4 and row.P1 > row.P2:
        hid_bear_div.enabled=True
        hid_bear_div.ifrom = row.P3_idx
        hid_bear_div.idoubfrom = row.P5_idx
        hid_bear_div.to   = row.P1_idx
        _rng1 = int((hid_bear_div.to - hid_bear_div.ifrom)/3)
        _rng2 = int((hid_bear_div.ifrom - hid_bear_div.idoubfrom)/3)
        if hid_bear_div.idoubfrom - _rng2 <= 0:
          hid_bear_div.enabled=False
      # other hidden-bearish-div
      elif row.P2 < row.P4 and row.P3 < row.P5 and row.P1 < row.P2:
        hid_bear_div.enabled=True
        hid_bear_div.ifrom = row.P4_idx
        hid_bear_div.idoubfrom = row.P6_idx
        hid_bear_div.to   = row.P2_idx
        _rng1 = int((hid_bear_div.to - hid_bear_div.ifrom)/3)
        _rng2 = int((hid_bear_div.ifrom - hid_bear_div.idoubfrom)/3)
        if hid_bear_div.idoubfrom - _rng2 <= 0:
          hid_bear_div.enabled=False


      # MACD check---      
      # checking regular-bearish-div
      # find min or max values around ifrom, ito and idoubfrom to enable more precise divergence detector
 
      # checking regular-bearish-div
      if reg_bear_div.enabled==True:
        _rng1 = int((reg_bear_div.to - reg_bear_div.ifrom)/3)
        _rng2 = int((reg_bear_div.ifrom - reg_bear_div.idoubfrom)/3)
        log += 'reg-bear rng1={} rng2={}'.format(_rng1, _rng2)
        _macd_max_to_v = df.MACD_main[reg_bear_div.to - _rng1 : reg_bear_div.to + _rng1].max()
        _macd_max_from_v = df.MACD_main[reg_bear_div.ifrom - _rng1 : reg_bear_div.ifrom + _rng1].max()
        _macd_max_2from_v = df.MACD_main[reg_bear_div.idoubfrom - _rng2 : reg_bear_div.idoubfrom + _rng2].max()
        _rsi_max_to_v = df.RSI[reg_bear_div.to - _rng1 : reg_bear_div.to + _rng1].max()
        _rsi_max_from_v = df.RSI[reg_bear_div.ifrom - _rng1 : reg_bear_div.ifrom + _rng1].max()
        _rsi_max_2from_v = df.RSI[reg_bear_div.idoubfrom - _rng2 : reg_bear_div.idoubfrom + _rng2].max()

        _macd_max_to_i = min(df.MACD_main[reg_bear_div.to - _rng1 : reg_bear_div.to + _rng1].idxmax(), df.index.values[-1])
        _macd_max_from_i = df.MACD_main[reg_bear_div.ifrom - _rng1 : reg_bear_div.ifrom + _rng1].idxmax()
        logger.debug('reg_bear_div.idoubfrom={}, rng2={}'.format(reg_bear_div.idoubfrom,_rng2))
        _macd_max_2from_i = df.MACD_main[reg_bear_div.idoubfrom - _rng2 : reg_bear_div.idoubfrom + _rng2].idxmax()
        _rsi_max_to_i = min(df.RSI[reg_bear_div.to - _rng1 : reg_bear_div.to + _rng1].idxmax(), df.index.values[-1])
        _rsi_max_from_i = df.RSI[reg_bear_div.ifrom - _rng1 : reg_bear_div.ifrom + _rng1].idxmax()
        _rsi_max_2from_i = df.RSI[reg_bear_div.idoubfrom - _rng2 : reg_bear_div.idoubfrom + _rng2].idxmax()
        
        if _macd_max_from_v > _macd_max_to_v:
          # check double divergence
          if df.ZIGZAG.iloc[reg_bear_div.idoubfrom] < df.ZIGZAG.iloc[reg_bear_div.ifrom] and _macd_max_2from_v > _macd_max_from_v:
            log += 'doub-reg-bear-div on macd ifrom {} to {}'.format(_macd_max_2from_i, _macd_max_from_i)
            df.loc[_macd_max_2from_i:_macd_max_to_i,'DIV_DOUB_REG_BEAR_MACD'] = 1.0
          # else simple divergence
          else:
            log += 'reg-bear-div on macd ifrom {} to {}'.format(_macd_max_from_i, _macd_max_to_i)
            df.loc[_macd_max_from_i:_macd_max_to_i, 'DIV_REG_BEAR_MACD'] = 1.0
        if _rsi_max_from_v > _rsi_max_to_v:
          # check double divergence
          if df.ZIGZAG.iloc[reg_bear_div.idoubfrom] < df.ZIGZAG.iloc[reg_bear_div.ifrom] and _rsi_max_2from_v > _rsi_max_from_v:
            log += 'doub-reg-bear-div on rsi ifrom {} to {}'.format(_rsi_max_2from_i, _rsi_max_to_i)
            df.loc[_rsi_max_2from_i:_rsi_max_to_i, 'DIV_DOUB_REG_BEAR_RSI'] = 1.0
          # else simple divergence
          else:
            log += 'reg-bear-div on rsi ifrom {} to {}'.format(_rsi_max_from_i, _rsi_max_to_i)
            df.loc[_rsi_max_from_i:_rsi_max_to_i, 'DIV_REG_BEAR_RSI'] = 1.0

      # checking hidden-bearish-div 
      if hid_bear_div.enabled==True:
        _rng1 = int((hid_bear_div.to - hid_bear_div.ifrom)/3)
        _rng2 = int((hid_bear_div.ifrom - hid_bear_div.idoubfrom)/3)
        log += 'hid-bear rng1={} rng2={}'.format(_rng1, _rng2)
        _macd_max_to_v = df.MACD_main[hid_bear_div.to - _rng1 : hid_bear_div.to + _rng1].max()
        _macd_max_from_v = df.MACD_main[hid_bear_div.ifrom - _rng1 : hid_bear_div.ifrom + _rng1].max()
        _macd_max_2from_v = df.MACD_main[hid_bear_div.idoubfrom - _rng2 : hid_bear_div.idoubfrom + _rng2].max()
        _rsi_max_to_v = df.RSI[hid_bear_div.to - _rng1 : hid_bear_div.to + _rng1].max()
        _rsi_max_from_v = df.RSI[hid_bear_div.ifrom - _rng1 : hid_bear_div.ifrom + _rng1].max()
        _rsi_max_2from_v = df.RSI[hid_bear_div.idoubfrom - _rng2 : hid_bear_div.idoubfrom + _rng2].max()

        _macd_max_to_i = min(df.MACD_main[hid_bear_div.to - _rng1 : hid_bear_div.to + _rng1].idxmax(), df.index.values[-1])
        _macd_max_from_i = df.MACD_main[hid_bear_div.ifrom - _rng1 : hid_bear_div.ifrom + _rng1].idxmax()
        _macd_max_2from_i = df.MACD_main[hid_bear_div.idoubfrom - _rng2 : hid_bear_div.idoubfrom + _rng2].idxmax()
        _rsi_max_to_i = min(df.RSI[hid_bear_div.to - _rng1 : hid_bear_div.to + _rng1].idxmax(), df.index.values[-1])
        _rsi_max_from_i = df.RSI[hid_bear_div.ifrom - _rng1 : hid_bear_div.ifrom + _rng1].idxmax()
        _rsi_max_2from_i = df.RSI[hid_bear_div.idoubfrom - _rng2 : hid_bear_div.idoubfrom + _rng2].idxmax()

        if _macd_max_from_v < _macd_max_to_v:
          # check double divergence
          if df.ZIGZAG.iloc[hid_bear_div.idoubfrom] > df.ZIGZAG.iloc[hid_bear_div.ifrom] and _macd_max_2from_v < _macd_max_from_v:
            log += 'doub-hid-bear-div on macd ifrom {} to {}'.format(_macd_max_2from_i, _macd_max_to_i)
            df.loc[_macd_max_2from_i:_macd_max_to_i, 'DIV_DOUB_HID_BEAR_MACD'] = 1.0
          # else simple divergence
          else:
            log += 'hid-bear-div on macd ifrom {} to {}'.format(_macd_max_from_i, _macd_max_to_i)
            df.loc[_macd_max_from_i:_macd_max_to_i, 'DIV_HID_BEAR_MACD'] = 1.0
        if _rsi_max_from_v < _rsi_max_to_v:
          # check double divergence
          if df.ZIGZAG.iloc[hid_bear_div.idoubfrom] > df.ZIGZAG.iloc[hid_bear_div.ifrom] and _rsi_max_2from_v < _rsi_max_from_v:
            log += 'doub-hid-bear-div on rsi ifrom {} to {}'.format(_rsi_max_2from_i, _rsi_max_to_i)
            df.loc[_rsi_max_2from_i:_rsi_max_to_i, 'DIV_DOUB_HID_BEAR_RSI'] = 1.0
          # else simple divergence
          else:
            log += 'hid-bear-div on rsi ifrom {} to {}'.format(_rsi_max_from_i, _rsi_max_to_i)
            df.loc[_rsi_max_from_i:_rsi_max_to_i, 'DIV_HID_BEAR_RSI'] = 1.0

      # checking hidden-bullish-div
      if hid_bull_div.enabled==True:
        _rng1 = int((hid_bull_div.to - hid_bull_div.ifrom)/3)
        _rng2 = int((hid_bull_div.ifrom - hid_bull_div.idoubfrom)/3)
        log += 'hid-bull rng1={} rng2={}'.format(_rng1, _rng2)
        _macd_min_to_v = df.MACD_main[hid_bull_div.to - _rng1 : hid_bull_div.to + _rng1].min()
        _macd_min_from_v = df.MACD_main[hid_bull_div.ifrom - _rng1 : hid_bull_div.ifrom + _rng1].min()
        _macd_min_2from_v = df.MACD_main[hid_bull_div.idoubfrom - _rng2 : hid_bull_div.idoubfrom + _rng2].min()
        _rsi_min_to_v = df.RSI[hid_bull_div.to - _rng1 : hid_bull_div.to + _rng1].min()
        _rsi_min_from_v = df.RSI[hid_bull_div.ifrom - _rng1 : hid_bull_div.ifrom + _rng1].min()
        _rsi_min_2from_v = df.RSI[hid_bull_div.idoubfrom - _rng2 : hid_bull_div.idoubfrom + _rng2].min()

        _macd_min_to_i = min(df.MACD_main[hid_bull_div.to - _rng1 : hid_bull_div.to + _rng1].idxmin(), df.index.values[-1])
        _macd_min_from_i = df.MACD_main[hid_bull_div.ifrom - _rng1 : hid_bull_div.ifrom + _rng1].idxmin()
        _macd_min_2from_i = df.MACD_main[hid_bull_div.idoubfrom - _rng2 : hid_bull_div.idoubfrom + _rng2].idxmin()
        _rsi_min_to_i = min(df.RSI[hid_bull_div.to - _rng1 : hid_bull_div.to + _rng1].idxmin(), df.index.values[-1])
        _rsi_min_from_i = df.RSI[hid_bull_div.ifrom - _rng1 : hid_bull_div.ifrom + _rng1].idxmin()
        _rsi_min_2from_i = df.RSI[hid_bull_div.idoubfrom - _rng2 : hid_bull_div.idoubfrom + _rng2].idxmin()

        if _macd_min_from_v > _macd_min_to_v:
          # check double divergence
          if df.ZIGZAG.iloc[hid_bull_div.idoubfrom] < df.ZIGZAG.iloc[hid_bull_div.ifrom] and _macd_min_2from_v > _macd_min_from_v:
            log += 'doub-hid-bull-div on macd ifrom {} to {}'.format(_macd_min_2from_i, _macd_min_to_i)
            df.loc[_macd_min_2from_i:_macd_min_to_i, 'DIV_DOUB_HID_BULL_MACD'] = 1.0
          # else simple divergence
          else:
            log += 'hid-bull-div on macd ifrom {} to {}'.format(_macd_min_from_i, _macd_min_to_i)
            df.loc[_macd_min_from_i:_macd_min_to_i, 'DIV_HID_BULL_MACD'] = 1.0
        if _rsi_min_from_v > _rsi_min_to_v:
          # check double divergence
          if df.ZIGZAG.iloc[hid_bull_div.idoubfrom] < df.ZIGZAG.iloc[hid_bull_div.ifrom] and _rsi_min_2from_v > _rsi_min_from_v:
            log += 'doub-hid-bull-div on rsi ifrom {} to {}'.format(_rsi_min_2from_i, _rsi_min_to_i)
            df.loc[_rsi_min_2from_i:_rsi_min_to_i, 'DIV_DOUB_HID_BULL_RSI'] = 1.0
          # else simple divergence
          else:
            log += 'hid-bull-div on rsi ifrom {} to {}'.format(_rsi_min_from_i, _rsi_min_to_i)
            df.loc[_rsi_min_from_i:_rsi_min_to_i, 'DIV_HID_BULL_RSI'] = 1.0

      # checking regular-bullish-div
      if reg_bull_div.enabled==True:
        _rng1 = int((reg_bull_div.to - reg_bull_div.ifrom)/3)
        _rng2 = int((reg_bull_div.ifrom - reg_bull_div.idoubfrom)/3)
        log += 'reg-bull rng1={} rng2={}'.format(_rng1, _rng2)
        _macd_min_to_v = df.MACD_main[reg_bull_div.to - _rng1 : reg_bull_div.to + _rng1].min()
        _macd_min_from_v = df.MACD_main[reg_bull_div.ifrom - _rng1 : reg_bull_div.ifrom + _rng1].min()
        _macd_min_2from_v = df.MACD_main[reg_bull_div.idoubfrom - _rng2 : reg_bull_div.idoubfrom + _rng2].min()
        _rsi_min_to_v = df.RSI[reg_bull_div.to - _rng1 : reg_bull_div.to + _rng1].min()
        _rsi_min_from_v = df.RSI[reg_bull_div.ifrom - _rng1 : reg_bull_div.ifrom + _rng1].min()
        _rsi_min_2from_v = df.RSI[reg_bull_div.idoubfrom - _rng2 : reg_bull_div.idoubfrom + _rng2].min()

        _macd_min_to_i = min(df.MACD_main[reg_bull_div.to - _rng1 : reg_bull_div.to + _rng1].idxmin(), df.index.values[-1])
        _macd_min_from_i = df.MACD_main[reg_bull_div.ifrom - _rng1 : reg_bull_div.ifrom + _rng1].idxmin()
        _macd_min_2from_i = df.MACD_main[reg_bull_div.idoubfrom - _rng2 : reg_bull_div.idoubfrom + _rng2].idxmin()
        _rsi_min_to_i = min(df.RSI[reg_bull_div.to - _rng1 : reg_bull_div.to + _rng1].idxmin(), df.index.values[-1])
        _rsi_min_from_i = df.RSI[reg_bull_div.ifrom - _rng1 : reg_bull_div.ifrom + _rng1].idxmin()
        _rsi_min_2from_i = df.RSI[reg_bull_div.idoubfrom - _rng2 : reg_bull_div.idoubfrom + _rng2].idxmin()

        if _macd_min_from_v < _macd_min_to_v:
          # check double divergence
          if df.ZIGZAG.iloc[reg_bull_div.idoubfrom] > df.ZIGZAG.iloc[reg_bull_div.ifrom] and _macd_min_2from_v < _macd_min_from_v:
            log += 'doub-reg-bull-div on macd ifrom {} to {}'.format(_macd_min_2from_i, _macd_min_to_i)
            df.loc[_macd_min_2from_i:_macd_min_to_i, 'DIV_DOUB_REG_BULL_MACD'] = 1.0
          # else simple divergence
          else:
            log += 'reg-bull-div on macd ifrom {} to {}'.format(_macd_min_from_i, _macd_min_to_i)
            df.loc[_macd_min_from_i:_macd_min_to_i, 'DIV_REG_BULL_MACD'] = 1.0
        
        # checking regular-bullish-div
        if _rsi_min_from_v < _rsi_min_to_v:
          # check double divergence
          if df.ZIGZAG.iloc[reg_bull_div.idoubfrom] > df.ZIGZAG.iloc[reg_bull_div.ifrom] and _rsi_min_2from_v < _rsi_min_from_v:
            log += 'doub-reg-bull-div on rsi ifrom {} to {}'.format(_rsi_min_2from_i, _rsi_min_to_i)
            df.loc[_rsi_min_2from_i:_rsi_min_to_i, 'DIV_DOUB_REG_BULL_RSI'] = 1.0
          # else simple divergence
          else:
            log += 'reg-bull-div on rsi ifrom {} to {}'.format(_rsi_min_from_i, _rsi_min_to_i)
            df.loc[_rsi_min_from_i:_rsi_min_to_i, 'DIV_REG_BULL_RSI'] = 1.0

      logger.debug(log)
      return

      # is an undefined trend, then discard calculation
      log += 'error-no-trend'      
      logger.debug(log)
      #---end-of-search-function

    # execute search
    df.apply(lambda x: search(x, df, nan_value, self.__logger), axis=1)

    # apply divergence strength
    def bullish_strength(x, df):
      return min(1.0, max((0.8*x.DIV_DOUB_REG_BULL_MACD) + (0.8*x.DIV_DOUB_REG_BULL_RSI) + (0.6*x.DIV_REG_BULL_MACD) + (0.6*x.DIV_REG_BULL_RSI),
                 (0.8*x.DIV_DOUB_HID_BULL_MACD) + (0.8*x.DIV_DOUB_HID_BULL_RSI) + (0.6*x.DIV_HID_BULL_MACD) + (0.6*x.DIV_HID_BULL_RSI)))
    def bearish_strength(x, df):
      return min(1.0, max((0.8*x.DIV_DOUB_REG_BEAR_MACD) + (0.8*x.DIV_DOUB_REG_BEAR_RSI) + (0.6*x.DIV_REG_BEAR_MACD) + (0.6*x.DIV_REG_BEAR_RSI),
                 (0.8*x.DIV_DOUB_HID_BEAR_MACD) + (0.8*x.DIV_DOUB_HID_BEAR_RSI) + (0.6*x.DIV_HID_BEAR_MACD) + (0.6*x.DIV_HID_BEAR_RSI)))

    df['BULLISH_DIVERGENCE'] = df.apply(lambda x: bullish_strength(x, df), axis=1)
    df['BEARISH_DIVERGENCE'] = df.apply(lambda x: bearish_strength(x, df), axis=1)
    return df['BULLISH_DIVERGENCE'], df['BEARISH_DIVERGENCE']


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def plotOHLC(self):
    """ Plot OHLC price
      Returns:
        ohlch_trace
    """
    trace_ohlc = go.Ohlc(x=self.__df.index.values, open=self.__df.OPEN, high=self.__df.HIGH, low=self.__df.LOW, close=self.__df.CLOSE, name='Candlestick')
    return [trace_ohlc]


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def plotIndicators(self):
    """ Plot all indicators
      Returns:
        dict() -- dictionary with all plot traces, annotations and shapes
    """
    # last sample position
    at = self.__df.index.values[-1]

    # OHLC price and Zigzag
    zz_traces = self.plotZigzag('black')
    self.trace_ohlc = zz_traces[0]
    self.trace_zigzag = zz_traces[1]

    # Bollinger Bands
    bb_traces = self.plotBollinger(['black', 'blue', 'red'])
    self.trace_bollinger_up = bb_traces[1]
    self.trace_bollinger_mid = bb_traces[2]
    self.trace_bollinger_down = bb_traces[3]
    self.trace_bollinger_width = bb_traces[4]
    self.trace_bollinger_b = bb_traces[5]
    
    # MACD, RSI oscillators
    osc_traces = self.plotOscillators(color=['blue','red','green'])
    self.trace_macd_main = osc_traces[1]
    self.trace_macd_sig = osc_traces[2]
    self.trace_macd_hist = osc_traces[3]
    self.trace_rsi = osc_traces[4]

    # Moving averages
    ma_traces, ma_shapes = self.plotMovingAverages(color=['blue', 'red', 'green'])
    self.trace_ma_fast = ma_traces[1]
    self.trace_ma_mid = ma_traces[2]
    self.trace_ma_slow = ma_traces[3]
    self.shapes_ma = ma_shapes

    # Fibo levels
    fibo_traces, fibo_annotations, fibo_shapes = self.plotFiboLevels(at=at, width=100, color='black')
    self.fibo_annotations = fibo_annotations
    self.fibo_shapes = fibo_shapes

    # Supports
    s = self.__df['SUPPORTS']
    s = s[s != 0]
    x1 = s.index.values[-1]
    x2 = s.index.values[-2]
    trace_ohlc,s1_shape = self.plotHorizontalLine(x1+1, x1, x1+100, s.iloc[-1], color='brown', width=2, dash='dashdot')
    trace_ohlc,s2_shape = self.plotHorizontalLine(x2+1, x2, x2+100, s.iloc[-2], color='violet', width=2, dash='dashdot')
    self.support1_shape = s1_shape
    self.support2_shape = s2_shape

    # Resistances
    r = self.__df['RESISTANCES']
    r = r[r != 0.0]
    x1 = r.index.values[-1]
    x2 = r.index.values[-2]
    trace_ohlc,r1_shape = self.plotHorizontalLine(x1+1, x1, x1+100, r.iloc[-1], color='brown', width=2, dash='dashdot')
    trace_ohlc,r2_shape = self.plotHorizontalLine(x2+1, x2, x2+100, r.iloc[-2], color='violet', width=2, dash='dashdot')
    self.resistance1_shape = r1_shape
    self.resistance2_shape = r2_shape

    # Channel
    _upperline = self.__df['CHANNEL_UPPER_LIMIT']
    _bottomline = self.__df['CHANNEL_LOWER_LIMIT']
    _upperline = _upperline[_upperline != '']
    _bottomline = _bottomline[_bottomline != '']
    _ux = _upperline.index.values[-1]
    _bx = _bottomline.index.values[-1]
    _ulast = _upperline.iloc[-1]
    _blast = _bottomline.iloc[-1]
    x = _ux+1
    trace_ohlc, ch_shapes = self.plotChannel(x, extended=100, color='black', width=1, dash='dashdot')
    self.channel_shapes = ch_shapes

    # Trend
    trace_ohlc, trend_shapes = self.plotTrends(nan_value=0.0)
    self.trend_shapes = trend_shapes

    # Divergences
    trace_ohlc, trace_macd_main, trace_rsi, div_shapes = self.plotDivergences(color='blue', nan_value = 0.0)
    self.ohlc_macd_rsi_shapes = div_shapes

  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def plotZigzag(self, color='black'):
    """ Plot Zigzag withing OHLC candlesticks
      Arguments:
        color -- Zigzag color (default black)
      Returns:
        [ohlc,zigzag] -- Array of traces to plot with Plot.ly
    """
    trace_ohlc = go.Ohlc(x=self.__df.index.values, open=self.__df.OPEN, high=self.__df.HIGH, low=self.__df.LOW, close=self.__df.CLOSE, name='Candlestick')
    _dfz = self.__df[self.__df.ZIGZAG > 0].copy()
    trace_zigzag = go.Scatter(x=_dfz.reset_index()['index'], y=_dfz.ZIGZAG, name='zigzag', line=scatter.Line(color=color, width=1))
    flip_shapes = []
    def flips_builder(x, flips):
      if x.FLIP == True:
        flips.append({
            'type': 'line',
            'x0': x.name,
            'y0': 0,
            'x1': x.name,
            'y1': x.HIGH,
            'line': {
                'color': 'blue',
                'width': 1,
                'dash': 'dashdot',
            }})
    self.__df.apply(lambda x: flips_builder(x, flip_shapes), axis=1)
    fig = go.Figure(data=[trace_ohlc, trace_zigzag])
    fig['layout'].update(height=600, title='Zigzag')
    fig['layout'].update(shapes=flip_shapes)   
    mx = np.max(self.__df.HIGH)
    mn = np.min(self.__df.LOW)
    r = mx - mn    
    fig['layout'].update(yaxis=dict(range=[mn - (r/2), mx + (r/2)]))

    return fig, [trace_ohlc, trace_zigzag], flip_shapes


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def plotBollinger(self, color=['black','blue','red']):
    """ Plot Bollinger indicators
      Arguments:
        color -- color (default black)
      Returns:
        [ohlc, bb_up, bb_mid, bb_lo, bb_width, bb_b] -- Array of traces to plot with Plot.ly
    """
    trace_ohlc = go.Ohlc(x=self.__df.index.values, open=self.__df.OPEN, high=self.__df.HIGH, low=self.__df.LOW, close=self.__df.CLOSE, name='Candlestick')
    trace_bollinger_up = go.Scatter(x=self.__df.index.values, y=self.__df.BOLLINGER_HI, name='BB_up', line=scatter.Line(color=color[0], width=1))
    trace_bollinger_mid = go.Scatter(x=self.__df.index.values, y=self.__df.BOLLINGER_MA, name='BB_ma', line=scatter.Line(color=color[0], width=1))
    trace_bollinger_down = go.Scatter(x=self.__df.index.values, y=self.__df.BOLLINGER_LO, name='BB_lo', line=scatter.Line(color=color[0], width=1))
    trace_bollinger_width = go.Scatter(x=self.__df.index.values, y=self.__df.BOLLINGER_WIDTH, name='BB_width', line=scatter.Line(color=color[1], width=1))
    trace_bollinger_b = go.Scatter(x=self.__df.index.values, y=self.__df.BOLLINGER_b, name='BB_%b', line=scatter.Line(color=color[2], width=1))
    fig = plotly.tools.make_subplots(rows=3, cols=1, subplot_titles=('BBands', 'BBands_W', 'BBands_%b'), shared_xaxes=True, vertical_spacing=0.1)
    fig.append_trace(trace_ohlc, 1, 1)
    fig.append_trace(trace_bollinger_up, 1, 1)
    fig.append_trace(trace_bollinger_mid, 1, 1)
    fig.append_trace(trace_bollinger_down, 1, 1)
    fig.append_trace(trace_bollinger_width, 2, 1)
    fig.append_trace(trace_bollinger_b, 3, 1)
    fig['layout'].update(height=800, title='BBands')    
    return fig,[trace_ohlc, trace_bollinger_up, trace_bollinger_mid, trace_bollinger_down, trace_bollinger_width, trace_bollinger_b]


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def plotOscillators(self, color=['blue','red','green']):
    """ Plot oscillators indicators
      Arguments:
        color -- colors
      Returns:
        [ohlc, macd_main, macd_sig, macd_hist, rsi] -- Array of traces to plot with Plot.ly
    """
    trace_ohlc = go.Ohlc(x=self.__df.index.values, open=self.__df.OPEN, high=self.__df.HIGH, low=self.__df.LOW, close=self.__df.CLOSE, name='Candlestick')
    trace_macd_main = go.Scatter(x=self.__df.index.values, y=self.__df.MACD_main, name='MACD_main', line=scatter.Line(color=color[0], width=1))
    trace_macd_sig = go.Scatter(x=self.__df.index.values, y=self.__df.MACD_sig, name='MACD_sig', line=scatter.Line(color=color[1], width=1))
    trace_macd_hist = go.Scatter(x=self.__df.index.values, y=self.__df.MACD_hist, name='MACD_hist', line=scatter.Line(color=color[2], width=1))
    trace_rsi = go.Scatter(x=self.__df.index.values, y=self.__df.RSI, name='RSI', line=scatter.Line(color=color[0], width=1))
    fig = plotly.tools.make_subplots(rows=3, cols=1, subplot_titles=('Price', 'MACD', 'RSI'), shared_xaxes=True, vertical_spacing=0.1)
    fig.append_trace(trace_ohlc, 1, 1)
    fig.append_trace(trace_macd_main, 2, 1)
    fig.append_trace(trace_macd_sig, 2, 1)
    fig.append_trace(trace_rsi, 3, 1)
    fig['layout'].update(height=800, title='Oscillators') 
    return fig, [trace_ohlc, trace_macd_main, trace_macd_sig, trace_macd_hist, trace_rsi]


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def plotMovingAverages(self, color=['blue','red','green']):
    """ Plot Moving averages and trends signals
      Arguments:
        color -- color 
      Returns:
        [ohlc, ma_fast, ma_mid, ma_slow, bull_trend, bear_trend] -- Array of traces to plot with Plot.ly
    """
    trace_ohlc = go.Ohlc(x=self.__df.index.values, open=self.__df.OPEN, high=self.__df.HIGH, low=self.__df.LOW, close=self.__df.CLOSE, name='Candlestick')
    trace_fast = go.Scatter(x=self.__df.index.values, y=self.__df.SMA_FAST, name='SMA_fast', line=scatter.Line(color=color[0], width=1))
    trace_mid = go.Scatter(x=self.__df.index.values, y=self.__df.SMA_MID, name='SMA_mid', line=scatter.Line(color=color[1], width=1))
    trace_slow = go.Scatter(x=self.__df.index.values, y=self.__df.SMA_SLOW, name='SMA_slow', line=scatter.Line(color=color[2], width=1))
    class ShapeBuilder():
      def __init__(self, logger, nan_value=0.0):
        self.__logger = logger
        self.shapes = []
        self.bullval = nan_value
        self.bearval = nan_value
        self.bullx0 = 0
        self.bullx1 = 0
        self.bearx0 = 0
        self.bearx1 = 0
      def build_shapes(self, x, nan_value=0.0):
        if x.SMA_BULLISH_TREND != nan_value and self.bullx0 == 0:        
          self.bullx0 = x.name
          self.bullx1 = x.name
          self.bullval = x.SMA_BULLISH_TREND
        elif x.SMA_BULLISH_TREND == self.bullval and self.bullx0 != 0:        
          self.bullx1 = x.name
        elif x.SMA_BULLISH_TREND != self.bullval and self.bullx0 != 0:
          self.shapes.append({
            'type': 'rect', 'xref': 'x', 'yref': 'paper',
            'x0': self.bullx0,'y0': 0,'x1': self.bullx1,'y1': 1,
            'fillcolor': 'green', 'opacity': self.bullval * 0.5, 'line':{'width':0,}}
          )
          if x.SMA_BULLISH_TREND != nan_value:
            self.bullx0 = x.name
            self.bullx1 = x.name
            self.bullval = x.SMA_BULLISH_TREND
          else:
            self.bullx0 = 0
            self.bullx1 = 0
            self.bullval = nan_value
        if x.SMA_BEARISH_TREND != nan_value and self.bearx0 == 0:
          self.bearx0 = x.name
          self.bearx1 = x.name
          self.bearval = x.SMA_BEARISH_TREND
        elif x.SMA_BEARISH_TREND == self.bearval and self.bearx0 != 0:
          self.bearx1 = x.name
        elif x.SMA_BEARISH_TREND != self.bearval and self.bearx0 != 0:
          self.shapes.append({
            'type': 'rect', 'xref': 'x', 'yref': 'paper',
            'x0': self.bearx0,'y0': 0,'x1': self.bearx1,'y1': 1,
            'fillcolor': 'red', 'opacity': self.bearval * 0.5, 'line':{'width':0,}}
          )
          if x.SMA_BEARISH_TREND != nan_value:
            self.bearx0 = x.name
            self.bearx1 = x.name
            self.bearval = x.SMA_BEARISH_TREND
          else:
            self.bearx0 = 0
            self.bearx1 = 0
            self.bearval = nan_value
    sb = ShapeBuilder(self.__logger)  
    self.__df.apply(lambda x: sb.build_shapes(x), axis=1) 
    fig = go.Figure(data=[trace_ohlc,trace_fast, trace_mid, trace_slow])
    fig['layout'].update(height=600, title='SMA Trends')
    fig['layout'].update(shapes=sb.shapes)   
    return fig, [trace_ohlc, trace_fast, trace_mid, trace_slow], sb.shapes


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def plotFiboLevels(self, at=-1, width = 100, color='black'):
    """ Plot fibolevels for sample at index 'at'
      Arguments:
        at -- sample to plot
        width -- line size
        color -- color 
      Returns:
        fibo_trace, fibo_anotations -- Trace and anotations for sample 
    """
    fibo_df = self.__df[:at].copy()
    _x0 = fibo_df.index.values[-1]
    trace_ohlc = go.Ohlc(x=fibo_df.index.values, open=fibo_df.OPEN, high=fibo_df.HIGH, low=fibo_df.LOW, close=fibo_df.CLOSE, name='Candlestick')
    fibo_anotations =[
      dict(x=_x0, y=fibo_df.CLOSE.iloc[-1], xref='x', yref='y', text='{}'.format(fibo_df.FIBO_CURR.iloc[-1]), showarrow=False, arrowhead=0, ax=0, ay=0, xanchor = "left", yanchor = "bottom"),
      dict(x=_x0, y=fibo_df.FIBO_023.iloc[-1], xref='x', yref='y', text=' 23%:{}'.format(fibo_df.FIBO_023.iloc[-1]), showarrow=False, arrowhead=0, ax=0, ay=0, xanchor = "left", yanchor = "bottom"),
      dict(x=_x0, y=fibo_df.FIBO_038.iloc[-1], xref='x', yref='y', text=' 38%:{}'.format(fibo_df.FIBO_038.iloc[-1]), showarrow=False, arrowhead=0, ax=0, ay=0, xanchor = "left", yanchor = "bottom"),
      dict(x=_x0, y=fibo_df.FIBO_050.iloc[-1], xref='x', yref='y', text=' 50%:{}'.format(fibo_df.FIBO_050.iloc[-1]), showarrow=False, arrowhead=0, ax=0, ay=0, xanchor = "left", yanchor = "bottom"),
      dict(x=_x0, y=fibo_df.FIBO_061.iloc[-1], xref='x', yref='y', text=' 61%:{}'.format(fibo_df.FIBO_061.iloc[-1]), showarrow=False, arrowhead=0, ax=0, ay=0, xanchor = "left", yanchor = "bottom"),
      dict(x=_x0, y=fibo_df.FIBO_078.iloc[-1], xref='x', yref='y', text=' 78%:{}'.format(fibo_df.FIBO_078.iloc[-1]), showarrow=False, arrowhead=0, ax=0, ay=0, xanchor = "left", yanchor = "bottom"),
      dict(x=_x0, y=fibo_df.FIBO_123.iloc[-1], xref='x', yref='y', text='123%:{}'.format(fibo_df.FIBO_123.iloc[-1]), showarrow=False, arrowhead=0, ax=0, ay=0, xanchor = "left", yanchor = "bottom"),
      dict(x=_x0, y=fibo_df.FIBO_138.iloc[-1], xref='x', yref='y', text='138%:{}'.format(fibo_df.FIBO_138.iloc[-1]), showarrow=False, arrowhead=0, ax=0, ay=0, xanchor = "left", yanchor = "bottom"),
      dict(x=_x0, y=fibo_df.FIBO_150.iloc[-1], xref='x', yref='y', text='150%:{}'.format(fibo_df.FIBO_150.iloc[-1]), showarrow=False, arrowhead=0, ax=0, ay=0, xanchor = "left", yanchor = "bottom"),
      dict(x=_x0, y=fibo_df.FIBO_161.iloc[-1], xref='x', yref='y', text='161%:{}'.format(fibo_df.FIBO_161.iloc[-1]), showarrow=False, arrowhead=0, ax=0, ay=0, xanchor = "left", yanchor = "bottom"),
      dict(x=_x0, y=fibo_df.FIBO_178.iloc[-1], xref='x', yref='y', text='178%:{}'.format(fibo_df.FIBO_178.iloc[-1]), showarrow=False, arrowhead=0, ax=0, ay=0, xanchor = "left", yanchor = "bottom")
    ]
    fibo_shapes = [
      {'type': 'line', 'x0': _x0, 'y0': fibo_df.FIBO_CURR.iloc[-1], 'x1': _x0+width, 'y1': fibo_df.FIBO_CURR.iloc[-1], 'line': {'color': color, 'width': 1, 'dash': 'dashdot'}},
      {'type': 'line', 'x0': _x0, 'y0': fibo_df.FIBO_023.iloc[-1], 'x1': _x0+width, 'y1': fibo_df.FIBO_023.iloc[-1], 'line': {'color': color, 'width': 1, 'dash': 'dashdot'}},     
      {'type': 'line', 'x0': _x0, 'y0': fibo_df.FIBO_038.iloc[-1], 'x1': _x0+width, 'y1': fibo_df.FIBO_038.iloc[-1], 'line': {'color': color, 'width': 1, 'dash': 'dashdot'}},     
      {'type': 'line', 'x0': _x0, 'y0': fibo_df.FIBO_050.iloc[-1], 'x1': _x0+width, 'y1': fibo_df.FIBO_050.iloc[-1], 'line': {'color': color, 'width': 1, 'dash': 'dashdot'}},     
      {'type': 'line', 'x0': _x0, 'y0': fibo_df.FIBO_061.iloc[-1], 'x1': _x0+width, 'y1': fibo_df.FIBO_061.iloc[-1], 'line': {'color': color, 'width': 1, 'dash': 'dashdot'}},     
      {'type': 'line', 'x0': _x0, 'y0': fibo_df.FIBO_078.iloc[-1], 'x1': _x0+width, 'y1': fibo_df.FIBO_078.iloc[-1], 'line': {'color': color, 'width': 1, 'dash': 'dashdot'}},     
      {'type': 'line', 'x0': _x0, 'y0': fibo_df.FIBO_123.iloc[-1], 'x1': _x0+width, 'y1': fibo_df.FIBO_123.iloc[-1], 'line': {'color': color, 'width': 1, 'dash': 'dashdot'}},     
      {'type': 'line', 'x0': _x0, 'y0': fibo_df.FIBO_138.iloc[-1], 'x1': _x0+width, 'y1': fibo_df.FIBO_138.iloc[-1], 'line': {'color': color, 'width': 1, 'dash': 'dashdot'}},     
      {'type': 'line', 'x0': _x0, 'y0': fibo_df.FIBO_150.iloc[-1], 'x1': _x0+width, 'y1': fibo_df.FIBO_150.iloc[-1], 'line': {'color': color, 'width': 1, 'dash': 'dashdot'}},     
      {'type': 'line', 'x0': _x0, 'y0': fibo_df.FIBO_161.iloc[-1], 'x1': _x0+width, 'y1': fibo_df.FIBO_161.iloc[-1], 'line': {'color': color, 'width': 1, 'dash': 'dashdot'}},     
      {'type': 'line', 'x0': _x0, 'y0': fibo_df.FIBO_178.iloc[-1], 'x1': _x0+width, 'y1': fibo_df.FIBO_178.iloc[-1], 'line': {'color': color, 'width': 1, 'dash': 'dashdot'}},     
    ]
    return trace_ohlc, fibo_anotations, fibo_shapes
  

  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def plotHorizontalLine(self, at, x0, x1, value, color='black', width=2, dash='dashdot'):
    """ Plot Horizontal line (Support & Resistance) at any given position
      Arguments:
        at -- sample to plot
        x0 -- idx from line start
        x1 -- idx to line end
        value -- value of the horizontal line
        color -- color 
        width -- line width
        dash -- line dash
      Returns:
        h_trace, h_shape -- Trace and shape for sample 
    """
    h_df = self.__df[:at].copy()
    trace_ohlc = go.Ohlc(x=h_df.index.values, open=h_df.OPEN, high=h_df.HIGH, low=h_df.LOW, close=h_df.CLOSE, name='Candlestick')
    h_shape = {'type': 'line', 'x0': x0, 'y0': value, 'x1': x1, 'y1': value, 'line': {'color': color, 'width': width, 'dash': dash}}
    return trace_ohlc, h_shape


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def plotChannel(self, at, extended=100, color='black', width=1, dash='dashdot'):
    """ Plot channel lines at any given position
      Arguments:
        at -- sample to plot
        color -- color 
        width -- line width
        dash -- line dash
      Returns:
        ch_trace, ch_shape -- Trace and shape for sample 
    """
    ch_df = self.__df[:at].copy()
    ux0,ux1 = 0, 0
    uy0,uy1 = 0.0, 0.0
    if ch_df.CHANNEL_UPPER_LIMIT.iloc[-1] == 'P3,P1':
      ux0 = ch_df.P3_idx.iloc[-1]
      ux1 = ch_df.P1_idx.iloc[-1]
      uy0 = ch_df.P3.iloc[-1]
      uy1 = ch_df.P1.iloc[-1]
    else:
      ux0 = ch_df.P4_idx.iloc[-1]
      ux1 = ch_df.P2_idx.iloc[-1]
      uy0 = ch_df.P4.iloc[-1]
      uy1 = ch_df.P2.iloc[-1]
    if extended > 0:      
      uy1 = (((uy1-uy0)/(ux1-ux0))*((ux1+extended)-ux0))+uy0
      ux1 = ux1 + extended
    bx0,bx1 = 0, 0
    by0,by1 = 0.0, 0.0
    if ch_df.CHANNEL_LOWER_LIMIT.iloc[-1] == 'P3,P1':
      bx0 = ch_df.P3_idx.iloc[-1]
      bx1 = ch_df.P1_idx.iloc[-1]
      by0 = ch_df.P3.iloc[-1]
      by1 = ch_df.P1.iloc[-1]
    else:
      bx0 = ch_df.P4_idx.iloc[-1]
      bx1 = ch_df.P2_idx.iloc[-1]
      by0 = ch_df.P4.iloc[-1]
      by1 = ch_df.P2.iloc[-1]
    if extended > 0:      
      by1 = (((by1-by0)/(bx1-bx0))*((bx1+extended)-bx0))+by0
      bx1 = bx1 + extended
    trace_ohlc = go.Ohlc(x=ch_df.index.values, open=ch_df.OPEN, high=ch_df.HIGH, low=ch_df.LOW, close=ch_df.CLOSE, name='Candlestick')
    ch_shapes = [ 
      {'type': 'line', 'x0': ux0, 'y0': uy0, 'x1': ux1, 'y1': uy1, 'line': {'color': color, 'width': width, 'dash': dash}},
      {'type': 'line', 'x0': bx0, 'y0': by0, 'x1': bx1, 'y1': by1, 'line': {'color': color, 'width': width, 'dash': dash}}
    ]
    return trace_ohlc, ch_shapes


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def plotTrends(self, nan_value = 0.0):
    """ Plot trends areas
      Arguments:
      Returns:
        h_trace, h_shape -- Trace and shape for sample 
    """
    class ShapeBuilder():
      def __init__(self, logger, nan_value=0.0):
        self.__logger = logger
        self.shapes = []
        self.bullval = nan_value
        self.bearval = nan_value
        self.bullx0 = 0
        self.bullx1 = 0
        self.bearx0 = 0
        self.bearx1 = 0
      def build_shapes(self, x, nan_value=0.0):
        if x.BULLISH_TREND != nan_value and self.bullx0 == 0:        
          self.bullx0 = x.name
          self.bullx1 = x.name
          self.bullval = x.BULLISH_TREND
        elif x.BULLISH_TREND == self.bullval and self.bullx0 != 0:        
          self.bullx1 = x.name
        elif x.BULLISH_TREND != self.bullval and self.bullx0 != 0:
          self.shapes.append({
            'type': 'rect', 'xref': 'x', 'yref': 'paper',
            'x0': self.bullx0,'y0': 0,'x1': self.bullx1,'y1': 1,
            'fillcolor': 'green', 'opacity': self.bullval * 0.5, 'line':{'width':0,}}
          )
          if x.BULLISH_TREND != nan_value:
            self.bullx0 = x.name
            self.bullx1 = x.name
            self.bullval = x.BULLISH_TREND
          else:
            self.bullx0 = 0
            self.bullx1 = 0
            self.bullval = nan_value
        if x.BEARISH_TREND != nan_value and self.bearx0 == 0:
          self.bearx0 = x.name
          self.bearx1 = x.name
          self.bearval = x.BEARISH_TREND
        elif x.BEARISH_TREND == self.bearval and self.bearx0 != 0:
          self.bearx1 = x.name
        elif x.BEARISH_TREND != self.bearval and self.bearx0 != 0:
          self.shapes.append({
            'type': 'rect', 'xref': 'x', 'yref': 'paper',
            'x0': self.bearx0,'y0': 0,'x1': self.bearx1,'y1': 1,
            'fillcolor': 'red', 'opacity': self.bearval * 0.5, 'line':{'width':0,}}
          )
          if x.BEARISH_TREND != nan_value:
            self.bearx0 = x.name
            self.bearx1 = x.name
            self.bearval = x.BEARISH_TREND
          else:
            self.bearx0 = 0
            self.bearx1 = 0
            self.bearval = nan_value
    sb = ShapeBuilder(self.__logger)  
    self.__df.apply(lambda x: sb.build_shapes(x), axis=1)
    trace_ohlc = go.Ohlc(x=self.__df.index.values, open=self.__df.OPEN, high=self.__df.HIGH, low=self.__df.LOW, close=self.__df.CLOSE, name='Candlestick')
    fig = go.Figure(data=[trace_ohlc])
    fig['layout'].update(height=600, title='Trends')
    fig['layout'].update(shapes=sb.shapes)
    return fig, trace_ohlc, sb.shapes


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def plotDivergences(self, color='blue', nan_value = 0.0):
    """ Plot divergences
      Arguments:
        color -- List of colors for oscillators and divergence markers
        nan_value -- NaN value (default 0.0)
      Returns:
        ohlc_trace, macd_trace, rsi_trace, ohlc_shape, macd_shape, rsi_shape -- Trace and shape for sample 
    """
    class ShapeBuilder():
      def __init__(self, logger, nan_value=0.0):
        self.__logger = logger
        self.shapes = []
        self.ohlc_shapes = []
        self.macd_shapes = []
        self.rsi_shapes = []
        self.bullval = nan_value
        self.bearval = nan_value
        self.regbullmacdx0 = 0
        self.hidbullmacdx0 = 0
        self.regbullmacdx1 = 0
        self.hidbullmacdx1 = 0
        self.regbearmacdx0 = 0
        self.hidbearmacdx0 = 0
        self.regbearmacdx1 = 0
        self.hidbearmacdx1 = 0
        self.regbullrsix0 = 0
        self.hidbullrsix0 = 0
        self.regbullrsix1 = 0
        self.hidbullrsix1 = 0
        self.regbearrsix0 = 0
        self.hidbearrsix0 = 0
        self.regbearrsix1 = 0
        self.hidbearrsix1 = 0
      def build_shapes(self, x, df, nan_value=0.0):
        # process div-markers
        log = ''
        if x.DIV_DOUB_REG_BEAR_MACD > 0.0 or x.DIV_REG_BEAR_MACD > 0.0:
          if self.regbearmacdx0 == 0:
            self.regbearmacdx0 = x.name
            self.regbearmacdx1 = x.name
          else:
            self.regbearmacdx1 = x.name
        elif self.regbearmacdx0 > 0:
          _x0 = self.regbearmacdx0
          _x1 = self.regbearmacdx1
          self.regbearmacdx0 = 0
          self.regbearmacdx1 = 0
          log += ' doub-reg-bear-macd from {} to {}'.format(_x0, _x1)
          self.shapes.append({ 'type': 'line', 'x0': _x0, 'y0': df.MACD_main.iloc[_x0], 'x1': _x1, 'y1': df.MACD_main.iloc[_x1], 
                                    'line': {'color': 'black', 'width': 3, 'dash': 'dashdot'}, 'xref':'x1', 'yref':'y2'})   
          self.shapes.append({ 'type': 'line', 'x0': _x0, 'y0': df.HIGH.iloc[_x0], 'x1': _x1, 'y1': df.HIGH.iloc[_x1], 
                                    'line': {'color': 'black', 'width': 3, 'dash': 'dashdot'}, 'xref':'x1', 'yref':'y1'})   

        if x.DIV_DOUB_REG_BEAR_RSI > 0.0 or x.DIV_REG_BEAR_RSI > 0.0:
          if self.regbearrsix0 == 0:
            self.regbearrsix0 = x.name
            self.regbearrsix1 = x.name
          else:
            self.regbearrsix1 = x.name
        elif self.regbearrsix0 > 0:
          _x0 = self.regbearrsix0
          _x1 = self.regbearrsix1
          self.regbearrsix0 = 0
          self.regbearrsix1 = 0
          log += ' doub-reg-bear-rsi from {} to {}'.format(_x0, _x1)
          self.shapes.append({ 'type': 'line', 'x0': _x0, 'y0': df.RSI.iloc[_x0], 'x1': _x1, 'y1': df.RSI.iloc[_x1], 
                                    'line': {'color': 'black', 'width': 3, 'dash': 'dashdot'}, 'xref':'x1', 'yref':'y3'})   
          self.shapes.append({ 'type': 'line', 'x0': _x0, 'y0': df.HIGH.iloc[_x0], 'x1': _x1, 'y1': df.HIGH.iloc[_x1], 
                                    'line': {'color': 'black', 'width': 3, 'dash': 'dashdot'}, 'xref':'x1', 'yref':'y1'})   

        if x.DIV_DOUB_REG_BULL_MACD > 0.0 or x.DIV_REG_BULL_MACD > 0.0:
          if self.regbullmacdx0 == 0:
            self.regbullmacdx0 = x.name
            self.regbullmacdx1 = x.name
          else:
            self.regbullmacdx1 = x.name
        elif self.regbullmacdx0 > 0:
          _x0 = self.regbullmacdx0
          _x1 = self.regbullmacdx1
          self.regbullmacdx0 = 0
          self.regbullmacdx1 = 0
          log += ' doub-reg-bear-macd from {} to {}'.format(_x0, _x1)
          self.shapes.append({ 'type': 'line', 'x0': _x0, 'y0': df.MACD_main.iloc[_x0], 'x1': _x1, 'y1': df.MACD_main.iloc[_x1], 
                                    'line': {'color': 'black', 'width': 3, 'dash': 'dashdot'}, 'xref':'x1', 'yref':'y2'})   
          self.shapes.append({ 'type': 'line', 'x0': _x0, 'y0': df.HIGH.iloc[_x0], 'x1': _x1, 'y1': df.HIGH.iloc[_x1], 
                                    'line': {'color': 'black', 'width': 3, 'dash': 'dashdot'}, 'xref':'x1', 'yref':'y1'})   

        if x.DIV_DOUB_REG_BULL_RSI > 0.0 or x.DIV_REG_BULL_RSI > 0.0:
          if self.regbullrsix0 == 0:
            self.regbullrsix0 = x.name
            self.regbullrsix1 = x.name
          else:
            self.regbullrsix1 = x.name
        elif self.regbullrsix0 > 0:
          _x0 = self.regbullrsix0
          _x1 = self.regbullrsix1
          self.regbullrsix0 = 0
          self.regbullrsix1 = 0
          log += ' doub-reg-bear-rsi from {} to {}'.format(_x0, _x1)
          self.shapes.append({ 'type': 'line', 'x0': _x0, 'y0': df.RSI.iloc[_x0], 'x1': _x1, 'y1': df.RSI.iloc[_x1], 
                                    'line': {'color': 'black', 'width': 3, 'dash': 'dashdot'}, 'xref':'x1', 'yref':'y3'})   
          self.shapes.append({ 'type': 'line', 'x0': _x0, 'y0': df.HIGH.iloc[_x0], 'x1': _x1, 'y1': df.HIGH.iloc[_x1], 
                                    'line': {'color': 'black', 'width': 3, 'dash': 'dashdot'}, 'xref':'x1', 'yref':'y1'})   

        if x.DIV_DOUB_HID_BEAR_MACD > 0.0 or x.DIV_HID_BEAR_MACD > 0.0:
          if self.hidbearmacdx0 == 0:
            self.hidbearmacdx0 = x.name
            self.hidbearmacdx1 = x.name
          else:
            self.hidbearmacdx1 = x.name
        elif self.hidbearmacdx0 > 0:
          _x0 = self.hidbearmacdx0
          _x1 = self.hidbearmacdx1
          self.hidbearmacdx0 = 0
          self.hidbearmacdx1 = 0
          log += ' doub-reg-bear-macd from {} to {}'.format(_x0, _x1)
          self.shapes.append({ 'type': 'line', 'x0': _x0, 'y0': df.MACD_main.iloc[_x0], 'x1': _x1, 'y1': df.MACD_main.iloc[_x1], 
                                    'line': {'color': 'black', 'width': 3, 'dash': 'dashdot'}, 'xref':'x1', 'yref':'y2'})   
          self.shapes.append({ 'type': 'line', 'x0': _x0, 'y0': df.HIGH.iloc[_x0], 'x1': _x1, 'y1': df.HIGH.iloc[_x1], 
                                    'line': {'color': 'black', 'width': 3, 'dash': 'dashdot'}, 'xref':'x1', 'yref':'y1'})   

        if x.DIV_DOUB_HID_BEAR_RSI > 0.0 or x.DIV_DOUB_HID_BEAR_RSI > 0.0:
          if self.hidbearrsix0 == 0:
            self.hidbearrsix0 = x.name
            self.hidbearrsix1 = x.name
          else:
            self.hidbearrsix1 = x.name
        elif self.hidbearrsix0 > 0:
          _x0 = self.hidbearrsix0
          _x1 = self.hidbearrsix1
          self.hidbearrsix0 = 0
          self.hidbearrsix1 = 0
          log += ' doub-reg-bear-rsi from {} to {}'.format(_x0, _x1)
          self.shapes.append({ 'type': 'line', 'x0': _x0, 'y0': df.RSI.iloc[_x0], 'x1': _x1, 'y1': df.RSI.iloc[_x1], 
                                    'line': {'color': 'black', 'width': 3, 'dash': 'dashdot'}, 'xref':'x1', 'yref':'y3'})   
          self.shapes.append({ 'type': 'line', 'x0': _x0, 'y0': df.HIGH.iloc[_x0], 'x1': _x1, 'y1': df.HIGH.iloc[_x1], 
                                    'line': {'color': 'black', 'width': 3, 'dash': 'dashdot'}, 'xref':'x1', 'yref':'y1'})   

        if x.DIV_DOUB_HID_BULL_MACD > 0.0 or x.DIV_HID_BULL_MACD > 0.0:
          if self.hidbullmacdx0 == 0:
            self.hidbullmacdx0 = x.name
            self.hidbullmacdx1 = x.name
          else:
            self.hidbullmacdx1 = x.name
        elif self.hidbullmacdx0 > 0:
          _x0 = self.hidbullmacdx0
          _x1 = self.hidbullmacdx1
          self.hidbullmacdx0 = 0
          self.hidbullmacdx1 = 0
          log += ' doub-reg-bear-macd from {} to {}'.format(_x0, _x1)
          self.shapes.append({ 'type': 'line', 'x0': _x0, 'y0': df.MACD_main.iloc[_x0], 'x1': _x1, 'y1': df.MACD_main.iloc[_x1], 
                                    'line': {'color': 'black', 'width': 3, 'dash': 'dashdot'}, 'xref':'x1', 'yref':'y2'})   
          self.shapes.append({ 'type': 'line', 'x0': _x0, 'y0': df.HIGH.iloc[_x0], 'x1': _x1, 'y1': df.HIGH.iloc[_x1], 
                                    'line': {'color': 'black', 'width': 3, 'dash': 'dashdot'}, 'xref':'x1', 'yref':'y1'})   

        if x.DIV_DOUB_HID_BULL_RSI > 0.0 or x.DIV_DOUB_HID_BULL_RSI > 0.0:
          if self.hidbullrsix0 == 0:
            self.hidbullrsix0 = x.name
            self.hidbullrsix1 = x.name
          else:
            self.hidbullrsix1 = x.name
        elif self.hidbullrsix0 > 0:
          _x0 = self.hidbullrsix0
          _x1 = self.hidbullrsix1
          self.hidbullrsix0 = 0
          self.hidbullrsix1 = 0
          log += ' doub-reg-bear-rsi from {} to {}'.format(_x0, _x1)
          self.shapes.append({ 'type': 'line', 'x0': _x0, 'y0': df.RSI.iloc[_x0], 'x1': _x1, 'y1': df.RSI.iloc[_x1], 
                                    'line': {'color': 'black', 'width': 3, 'dash': 'dashdot'}, 'xref':'x1', 'yref':'y3'})   
          self.shapes.append({ 'type': 'line', 'x0': _x0, 'y0': df.HIGH.iloc[_x0], 'x1': _x1, 'y1': df.HIGH.iloc[_x1], 
                                    'line': {'color': 'black', 'width': 3, 'dash': 'dashdot'}, 'xref':'x1', 'yref':'y1'})   
        if log != '':
          self.__logger.debug(log)
      #<--- end of ShapeBuilder.build_shapes

    sb = ShapeBuilder(self.__logger)  
    self.__df.apply(lambda x: sb.build_shapes(x, self.__df), axis=1)
    trace_ohlc = go.Ohlc(x=self.__df.index.values, open=self.__df.OPEN, high=self.__df.HIGH, low=self.__df.LOW, close=self.__df.CLOSE, name='Candlestick')
    trace_macd_main = go.Scatter(x=self.__df.index.values, y=self.__df.MACD_main, name='MACD_main', line=scatter.Line(color=color, width=1))
    trace_rsi = go.Scatter(x=self.__df.index.values, y=self.__df.RSI, name='RSI', line=scatter.Line(color=color, width=1))

    ohlc_shapes = [sb.shapes]
    fig = plotly.tools.make_subplots(rows=3, cols=1, subplot_titles=('OHLC', 'MACD', 'RSI'), shared_xaxes=True, vertical_spacing=0.1)
    fig.append_trace(trace_ohlc, 1, 1)
    fig.append_trace(trace_macd_main, 2, 1)
    fig.append_trace(trace_rsi, 3, 1)
    fig['layout'].update(height=600, title='Divergences')
    fig['layout'].update(shapes=sb.shapes)

    return fig, [trace_ohlc, trace_macd_main, trace_rsi], sb.shapes


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def updateIndicators(self, new_bar):    
    """ Updates indicator dataframe with a new OHLC bar
      Keyword Argumnets:
        new_bar -- New OHLC bar with time,open,high,low,close columns
    """
    
    # use last row as a template for new data
    new_row = self.__df.iloc[-1].copy()

    # set new_bar values to template row
    new_row['TIME'] = new_bar['TIME']
    new_row['OPEN']=new_bar['OPEN']
    new_row['HIGH']=new_bar['HIGH']
    new_row['LOW']=new_bar['LOW']
    new_row['CLOSE']=new_bar['CLOSE']
    new_row['VOL']=new_bar['VOL']
    new_row['TICKVOL']=new_bar['TICKVOL']
    new_row['SPREAD']=new_bar['SPREAD']

    _df_backup = self.__df.copy()
    self.__logger.debug('Backup from idx={} to idx={}'.format(_df_backup.index.values[0], _df_backup.index.values[-1]))

    # append row to dataframe
    self.__df = self.__df.append(new_row, ignore_index=True)

    # rebuild indicators
    _idx_for_update = max(500 , (self.__df.index.values[-1] - self.__df.P6_idx.iloc[-1]))
    self.__logger.debug('Updating from last {} rows'.format(_idx_for_update))
    self.__df = self.__df[-_idx_for_update:].copy()
    self.__df.reset_index(drop=True, inplace=True)
    self.__logger.debug('Reindex idx[0]={} to idx={}'.format(self.__df.index.values[0], self.__df.index.values[-1]))
    self.buildIndicators()
    self.__df = _df_backup.append(self.__df.iloc[-1], ignore_index=True)
    return self.__df     


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def fuzzifyZigzag(self, timeperiod=50):
    """ Fuzzifies zigzag indicators into next fuzzy-variables:
      - ZZ_DURATION_1 variable gives duration between consecutive flips
        - ZZDur1VerySmall
        - ZZDur1SighthlySmall
        - ZZDur1Similar
        - ZZDur1SlightlyLarge
        - ZZDur1VeryLarge 
      - ZZ_DURATION_2 variable gives duration between flips in the same direction (pos, neg)
        - ZZDur2VerySmall
        - ZZDur2SighthlySmall
        - ZZDur2Similar
        - ZZDur2SlightlyLarge
        - ZZDur2VeryLarge 
      - ZZ_RANGE variable gives range against its SMA
        - ZZRangeVerySmall
        - ZZRangeSighthlySmall
        - ZZRangeSimilar
        - ZZRangeSlightlyLarge
        - ZZRangeVeryLarge 

      - flip range      
      Keyword arguments:
        timeperiod -- period to build the sinthetic fuzzy indicator
      Return:
        self.__df -- Updated dataframe      
    """
    # builds ZZ_FUZ_ZZ_DURATION_1 and ZZ_FUZ_ZZ_DURATION_2 columns
    self.__logger.debug('Fuzzifying Zigzag indicators...')
    # Get zigzags
    df_zz = self.__df[(self.__df.ZIGZAG > 0.0) & (self.__df.ACTION.str.contains('-in-progress')==False)]
    df_zz = df_zz[['ZIGZAG','ACTION']].copy()
    self.__logger.debug('processing {} rows'.format(df_zz.shape[0]))

    # Use index as a sample count difference between zigzags
    _df1 = df_zz.reset_index().copy()
    _df2 = df_zz.reset_index().shift(1).copy()
    _df3 = df_zz.reset_index().shift(2).copy()    

    # Create DURATION columns with the duration of each flip and with the previous of same direction
    self.__logger.debug('Building fuzzy set points based on BBANDS')

    _df_result = df_zz.reset_index(drop=True)
    _df_result['ZZ_IDX'] = _df1['index']
    _df_result['ZZ_DURATION_1'] = _df1['index'] - _df2['index']
    _df_result['ZZ_DURATION_2'] = _df1['index'] - _df3['index']  
    _df_result['d1bbup1'], _df_result['d1bbma1'], _df_result['d1bblo1'] = talib.BBANDS(_df_result.ZZ_DURATION_1, timeperiod=timeperiod, nbdevup=1.0, nbdevdn=1.0, matype=0)
    _df_result['d1bbup2'], _df_result['d1bbma2'], _df_result['d1bblo2'] = talib.BBANDS(_df_result.ZZ_DURATION_1, timeperiod=timeperiod, nbdevup=2.0, nbdevdn=2.0, matype=0)    

    _df_result['d2bbup1'], _df_result['d2bbma1'], _df_result['d2bblo1'] = talib.BBANDS(_df_result.ZZ_DURATION_2, timeperiod=timeperiod, nbdevup=1.0, nbdevdn=1.0, matype=0)
    _df_result['d2bbup2'], _df_result['d2bbma2'], _df_result['d2bblo2'] = talib.BBANDS(_df_result.ZZ_DURATION_2, timeperiod=timeperiod, nbdevup=2.0, nbdevdn=2.0, matype=0)

    def fn_fuzzify_duration(x, df, logger):
      logger.debug('fuzzifying row[{}]=> crisp={}'.format(x.name, x.ZZ_DURATION_1))
      f_sets = [{'type':'left-edge',    'name':'ZZDur1VerySmall',     'p0': x.d1bblo2, 'p1': x.d1bblo1},
                {'type':'internal-3pt', 'name':'ZZDur1SighthlySmall', 'p0': x.d1bblo2, 'p1': x.d1bblo1, 'p2': x.d1bbma1},
                {'type':'internal-3pt', 'name':'ZZDur1Similar',       'p0': x.d1bblo1, 'p1': x.d1bbma1, 'p2': x.d1bbup1},
                {'type':'internal-3pt', 'name':'ZZDur1SlightlyLarge', 'p0': x.d1bbma1, 'p1': x.d1bbup1, 'p2': x.d1bbup2},
                {'type':'right-edge'  , 'name':'ZZDur1VeryLarge',     'p0': x.d1bbup1, 'p1': x.d1bbup2}]
      fz1 = Fuzzifier.fuzzify(x.ZZ_DURATION_1, f_sets)
      
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_1'] = x.ZZ_DURATION_1
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_1_G0'] = fz1[0]
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_1_G1'] = fz1[1]
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_1_G2'] = fz1[2]
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_1_G3'] = fz1[3]
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_1_G4'] = fz1[4]
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_1_S-2'] = x.d1bblo2
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_1_S-1'] = x.d1bblo1
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_1_S0'] = x.d1bbma1
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_1_S+1'] = x.d1bbup1
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_1_S+2'] = x.d1bbup2
      f_sets = [{'type':'left-edge',    'name':'ZZDur2VerySmall',     'p0': x.d2bblo2, 'p1': x.d2bblo1},
                {'type':'internal-3pt', 'name':'ZZDur2SighthlySmall', 'p0': x.d2bblo2, 'p1': x.d2bblo1, 'p2': x.d2bbma1},
                {'type':'internal-3pt', 'name':'ZZDur2Similar',       'p0': x.d2bblo1, 'p1': x.d2bbma1, 'p2': x.d2bbup1},
                {'type':'internal-3pt', 'name':'ZZDur2SlightlyLarge', 'p0': x.d2bbma1, 'p1': x.d2bbup1, 'p2': x.d2bbup2},
                {'type':'right-edge'  , 'name':'ZZDur2VeryLarge',     'p0': x.d2bbup1, 'p1': x.d2bbup2}]
      fz2 = Fuzzifier.fuzzify(x.ZZ_DURATION_2, f_sets)
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_2'] = x.ZZ_DURATION_2
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_2_G0'] = fz2[0]
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_2_G1'] = fz2[1]
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_2_G2'] = fz2[2]
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_2_G3'] = fz2[3]
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_2_G4'] = fz2[4]
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_2_S-2'] = x.d2bblo2
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_2_S-1'] = x.d2bblo1
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_2_S0'] = x.d2bbma1
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_2_S+1'] = x.d2bbup1
      df.at[x.ZZ_IDX, 'FUZ_ZZ_DURATION_2_S+2'] = x.d2bbup2
    _df_result.apply(lambda x: fn_fuzzify_duration(x, self.__df, self.__logger), axis=1)


    # Create RANGE columns with the range of each flip and previous 
    _df_result = df_zz.reset_index(drop=True)
    _df_result['ZZ_IDX'] = _df1['index']
    _df_result['ZZ_RANGE_ABS'] = _df1.ZIGZAG - _df2.ZIGZAG
    _df_result['r1bbup1'], _df_result['r1bbma1'], _df_result['r1bblo1'] = talib.BBANDS(_df_result['ZZ_RANGE_ABS'], timeperiod=timeperiod, nbdevup=1.0, nbdevdn=1.0, matype=0)
    _df_result['r1bbup2'], _df_result['r1bbma2'], _df_result['r1bblo2'] = talib.BBANDS(_df_result['ZZ_RANGE_ABS'], timeperiod=timeperiod, nbdevup=2.0, nbdevdn=2.0, matype=0)    
    def fn_fuzzify_range(x, df, logger):
      logger.debug('fuzzifying_range row[{}]=> crisp={}'.format(x.name, x.ZZ_RANGE_ABS))
      f_sets = [{'type':'left-edge',    'name':'ZZRangeVerySmall',      'p0': x.r1bblo2, 'p1': x.r1bblo1},
                {'type':'internal-3pt', 'name':'ZZRangeSighthlySmall',  'p0': x.r1bblo2, 'p1': x.r1bblo1, 'p2': x.r1bbma1},
                {'type':'internal-3pt', 'name':'ZZRangeSimilar',        'p0': x.r1bblo1, 'p1': x.r1bbma1, 'p2': x.r1bbup1},
                {'type':'internal-3pt', 'name':'ZZRangeSlightlyLarge',  'p0': x.r1bbma1, 'p1': x.r1bbup1, 'p2': x.r1bbup2},
                {'type':'right-edge'  , 'name':'ZZRangeVeryLarge',      'p0': x.r1bbup1, 'p1': x.r1bbup2}]
      fz1 = Fuzzifier.fuzzify(x.ZZ_RANGE_ABS, f_sets)
      df.at[x.ZZ_IDX, 'FUZ_ZZ_RANGE'] = x.ZZ_RANGE_ABS
      df.at[x.ZZ_IDX, 'FUZ_ZZ_RANGE_G0'] = fz1[0]
      df.at[x.ZZ_IDX, 'FUZ_ZZ_RANGE_G1'] = fz1[1]
      df.at[x.ZZ_IDX, 'FUZ_ZZ_RANGE_G2'] = fz1[2]
      df.at[x.ZZ_IDX, 'FUZ_ZZ_RANGE_G3'] = fz1[3]
      df.at[x.ZZ_IDX, 'FUZ_ZZ_RANGE_G4'] = fz1[4]
      df.at[x.ZZ_IDX, 'FUZ_ZZ_RANGE_S-2'] = x.r1bblo2
      df.at[x.ZZ_IDX, 'FUZ_ZZ_RANGE_S-1'] = x.r1bblo1
      df.at[x.ZZ_IDX, 'FUZ_ZZ_RANGE_S0'] = x.r1bbma1
      df.at[x.ZZ_IDX, 'FUZ_ZZ_RANGE_S+1'] = x.r1bbup1
      df.at[x.ZZ_IDX, 'FUZ_ZZ_RANGE_S+2'] = x.r1bbup2
      
    _df_result.apply(lambda x: fn_fuzzify_range(x, self.__df, self.__logger), axis=1)

    return self.__df


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def fuzzifyBollinger(self, timeperiod=50):
    """ Fuzzifies BBands indicator in 2 fuzzy-variables:
      - BOLLINGER_WIDTH variable classifies Bands-width against its SMA into these fuzzy-sets:
        - WidthVerySmall
        - WidthSmall
        - WidthSighthlySmall
        - WidthSimilar
        - WidthSlightlyLarge
        - WidthLarge
        - WidthVeryLarge 
      - BOLLINGER_b variables classifies BBands %b against its SMA into these fuzzy-sets:
        - InOverSell
        - NearOverSell
        - FarFromOverSell
        - FarFromAnySignal
        - FarFromOverBought
        - NearToOverBought
        - InOverBought
      Return:
        self.__df -- Updated dataframe      
    """
    _df_result = self.__df[['BOLLINGER_WIDTH', 'BOLLINGER_b']].copy()
    _df_result['wbbup1'], _df_result['wbbma1'], _df_result['wbblo1'] = talib.BBANDS(_df_result.BOLLINGER_WIDTH, timeperiod=timeperiod, nbdevup=1.0, nbdevdn=1.0, matype=0)
    _df_result['wbbup2'], _df_result['wbbma2'], _df_result['wbblo2'] = talib.BBANDS(_df_result.BOLLINGER_WIDTH, timeperiod=timeperiod, nbdevup=2.0, nbdevdn=2.0, matype=0)    
    _df_result['wbbup3'], _df_result['wbbma3'], _df_result['wbblo3'] = talib.BBANDS(_df_result.BOLLINGER_WIDTH, timeperiod=timeperiod, nbdevup=3.0, nbdevdn=3.0, matype=0)    
    def fn_fuzzify_bollinger(x, df, logger):
      logger.debug('fuzzifying row[{}]=> crisp={}'.format(x.name, x.BOLLINGER_WIDTH))
      f_sets = [{'type':'left-edge',    'name':'WidthVerySmall',      'p0': x.wbblo3, 'p1': x.wbblo2},
                {'type':'internal-3pt', 'name':'WidthSmall',          'p0': x.wbblo3, 'p1': x.wbblo2, 'p2': x.wbblo1},
                {'type':'internal-3pt', 'name':'WidthSighthlySmall',  'p0': x.wbblo2, 'p1': x.wbblo1, 'p2': x.wbbma1},
                {'type':'internal-3pt', 'name':'WidthSimilar',        'p0': x.wbblo1, 'p1': x.wbbma1, 'p2': x.wbbup1},
                {'type':'internal-3pt', 'name':'WidthSlightlyLarge',  'p0': x.wbbma1, 'p1': x.wbbup1, 'p2': x.wbbup2},
                {'type':'internal-3pt', 'name':'WidthLarge',          'p0': x.wbbup1, 'p1': x.wbbup2, 'p2': x.wbbup3},
                {'type':'right-edge'  , 'name':'WidthVeryLarge',      'p0': x.wbbup2, 'p1': x.wbbup3}]
      fz1 = Fuzzifier.fuzzify(x.BOLLINGER_WIDTH, f_sets)
      df.at[x.name, 'FUZ_BOLLINGER_WIDTH'] = x.BOLLINGER_WIDTH
      df.at[x.name, 'FUZ_BOLLINGER_WIDTH_G0'] = fz1[0]
      df.at[x.name, 'FUZ_BOLLINGER_WIDTH_G1'] = fz1[1]
      df.at[x.name, 'FUZ_BOLLINGER_WIDTH_G2'] = fz1[2]
      df.at[x.name, 'FUZ_BOLLINGER_WIDTH_G3'] = fz1[3]
      df.at[x.name, 'FUZ_BOLLINGER_WIDTH_G4'] = fz1[4]
      df.at[x.name, 'FUZ_BOLLINGER_WIDTH_G5'] = fz1[5]
      df.at[x.name, 'FUZ_BOLLINGER_WIDTH_G6'] = fz1[6]
      df.at[x.name, 'FUZ_BOLLINGER_WIDTH_S-3'] = x.wbblo3
      df.at[x.name, 'FUZ_BOLLINGER_WIDTH_S-2'] = x.wbblo2
      df.at[x.name, 'FUZ_BOLLINGER_WIDTH_S-1'] = x.wbblo1
      df.at[x.name, 'FUZ_BOLLINGER_WIDTH_S0'] = x.wbbma1
      df.at[x.name, 'FUZ_BOLLINGER_WIDTH_S+1'] = x.wbbup1
      df.at[x.name, 'FUZ_BOLLINGER_WIDTH_S+2'] = x.wbbup2
      df.at[x.name, 'FUZ_BOLLINGER_WIDTH_S+3'] = x.wbbup3
      logger.debug('fuzzifying row[{}]=> crisp={}'.format(x.name, x.BOLLINGER_b))
      f_sets = [{'type':'left-edge',    'name':'InOverSell',        'p0': 0.0, 'p1': 0.15},
                {'type':'internal-3pt', 'name':'NearOverSell',      'p0': 0.0, 'p1': 0.15, 'p2': 0.25},
                {'type':'internal-3pt', 'name':'FarFromOverSell',   'p0': 0.25, 'p1': 0.35, 'p2': 0.5},
                {'type':'internal-3pt', 'name':'FarFromAnySignal',  'p0': 0.35, 'p1': 0.5, 'p2': 0.65},
                {'type':'internal-3pt', 'name':'FarFromOverBought', 'p0': 0.5, 'p1': 0.65, 'p2': 0.75},
                {'type':'internal-3pt', 'name':'NearToOverBought',  'p0': 0.65, 'p1': 0.75, 'p2': 0.85},
                {'type':'right-edge'  , 'name':'InOverBought',      'p0': 0.85, 'p1': 1.0}]
      fz1 = Fuzzifier.fuzzify(x.BOLLINGER_b, f_sets)
      df.at[x.name, 'FUZ_BOLLINGER_b'] = x.BOLLINGER_b
      df.at[x.name, 'FUZ_BOLLINGER_b_G0'] = fz1[0]
      df.at[x.name, 'FUZ_BOLLINGER_b_G1'] = fz1[1]
      df.at[x.name, 'FUZ_BOLLINGER_b_G2'] = fz1[2]
      df.at[x.name, 'FUZ_BOLLINGER_b_G3'] = fz1[3]
      df.at[x.name, 'FUZ_BOLLINGER_b_G4'] = fz1[4]
      df.at[x.name, 'FUZ_BOLLINGER_b_G5'] = fz1[5]
      df.at[x.name, 'FUZ_BOLLINGER_b_G6'] = fz1[6]
      df.at[x.name, 'FUZ_BOLLINGER_b_S-3'] = 0.0
      df.at[x.name, 'FUZ_BOLLINGER_b_S-2'] = 0.15
      df.at[x.name, 'FUZ_BOLLINGER_b_S-1'] = 0.35
      df.at[x.name, 'FUZ_BOLLINGER_b_S0'] = 0.5
      df.at[x.name, 'FUZ_BOLLINGER_b_S+1'] = 0.65
      df.at[x.name, 'FUZ_BOLLINGER_b_S+2'] = 0.75
      df.at[x.name, 'FUZ_BOLLINGER_b_S+3'] = 1.0
    _df_result.apply(lambda x: fn_fuzzify_bollinger(x, self.__df, self.__logger), axis=1)
    return self.__df


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def fuzzifyMACD(self, timeperiod=500):
    """ Fuzzifies MACD into 2 variables and their associated fuzzy-sets:
        - MACD_CROSS_ZERO variable indicates approaching state from MACD_main to ZEROLEVEL:
          - MacdFarBelowZero
          - MacdMidDistBelowZero
          - MacdNearBelowZero
          - MacdJustCrossedZero 
          - MacdNearAboveZero
          - MacdMidDistAboveZero
          - MacdFarAboveZero
        - MACD_CROSS_SIG variable indicates approaching state from MACD_main to MACD_sig:
          - MacdFarBelowSig
          - MacdMidDistBelowSig
          - MacdNearBelowSig
          - MacdNearAboveSig
          - MacdMidDistAboveSig
          - MacdFarAboveSig
          - MacdJustCrossedSig 
       Return:
        self.__df -- Updated dataframe      
    """
    _df_result = self.__df[['MACD_main','MACD_sig','MACD_CROSS_ZERO_UP','MACD_CROSS_ZERO_DN','MACD_CROSS_SIG_UP','MACD_CROSS_SIG_DN']].copy()
    _df_result['macd_main_sig'] = _df_result.MACD_main - _df_result.MACD_sig
    _df_z_pos = _df_result.apply(lambda x: max(x.MACD_main, 0.0), axis=1)
    _df_z_neg = _df_result.apply(lambda x: min(x.MACD_main, 0.0), axis=1)
    _df_result['zbbup1'], _, _ = talib.BBANDS(_df_z_pos, timeperiod=timeperiod, nbdevup=0.5, nbdevdn=0.5, matype=0)
    _df_result['zbbup2'], _, _ = talib.BBANDS(_df_z_pos, timeperiod=timeperiod, nbdevup=1.5, nbdevdn=1.5, matype=0)    
    _df_result['zbbup3'], _, _ = talib.BBANDS(_df_z_pos, timeperiod=timeperiod, nbdevup=2.5, nbdevdn=2.5, matype=0)    
    _, _, _df_result['zbblo1'] = talib.BBANDS(_df_z_neg, timeperiod=timeperiod, nbdevup=0.5, nbdevdn=0.5, matype=0)
    _, _, _df_result['zbblo2'] = talib.BBANDS(_df_z_neg, timeperiod=timeperiod, nbdevup=1.5, nbdevdn=1.5, matype=0)    
    _, _, _df_result['zbblo3'] = talib.BBANDS(_df_z_neg, timeperiod=timeperiod, nbdevup=2.5, nbdevdn=2.5, matype=0)    
    _df_s_pos = _df_result.apply(lambda x: max(x.macd_main_sig, 0.0), axis=1)
    _df_s_neg = _df_result.apply(lambda x: min(x.macd_main_sig, 0.0), axis=1)
    _df_result['sbbup1'], _, _ = talib.BBANDS(_df_s_pos, timeperiod=timeperiod, nbdevup=0.5, nbdevdn=0.5, matype=0)
    _df_result['sbbup2'], _, _ = talib.BBANDS(_df_s_pos, timeperiod=timeperiod, nbdevup=1.5, nbdevdn=1.5, matype=0)    
    _df_result['sbbup3'], _, _ = talib.BBANDS(_df_s_pos, timeperiod=timeperiod, nbdevup=2.5, nbdevdn=2.5, matype=0)    
    _, _, _df_result['sbblo1'] = talib.BBANDS(_df_s_neg, timeperiod=timeperiod, nbdevup=0.5, nbdevdn=0.5, matype=0)
    _, _, _df_result['sbblo2'] = talib.BBANDS(_df_s_neg, timeperiod=timeperiod, nbdevup=1.5, nbdevdn=1.5, matype=0)    
    _, _, _df_result['sbblo3'] = talib.BBANDS(_df_s_neg, timeperiod=timeperiod, nbdevup=2.5, nbdevdn=2.5, matype=0)    
    _df_result['ZERO']=0.0
    def fn_fuzzify_macd(x, df, logger):
      logger.debug('fuzzifying row[{}]=> crisp_macd={}'.format(x.name, x.MACD_main))
      f_sets = [{'type':'left-edge',      'name':'MacdFarBelowZero',    'p0': x.zbblo3, 'p1': x.zbblo2},
                {'type':'internal-3pt',   'name':'MacdMidDistBelowZero','p0': x.zbblo3, 'p1': x.zbblo2, 'p2': x.zbblo1},
                {'type':'trapezoid-left', 'name':'MacdNearBelowZero',   'p0': x.zbblo2, 'p1': x.zbblo1, 'p2': x.ZERO},
                {'type':'singleton',      'name':'MacdJustCrossedZero', 'p0': x.ZERO},
                {'type':'trapezoid-right','name':'MacdNearAboveZero',   'p0': x.ZERO,   'p1': x.zbbup1, 'p2': x.zbbup2},
                {'type':'internal-3pt',   'name':'MacdMidDistAboveZero','p0': x.zbbup1, 'p1': x.zbbup2, 'p2': x.zbbup3},
                {'type':'right-edge'  ,   'name':'MacdFarAboveZero',    'p0': x.zbbup2, 'p1': x.zbbup3}]
      fz1 = Fuzzifier.fuzzify(x.MACD_main, f_sets)
      df.at[x.name, 'FUZ_MACD_CROSS_ZERO'] = x.MACD_main
      if x.MACD_CROSS_ZERO_UP==True or x.MACD_CROSS_ZERO_DN==True:
        fz1[3] = 1.0
      else:
        fz1[3] = 0.0
      df.at[x.name, 'FUZ_MACD_CROSS_ZERO_G0'] = fz1[0]
      df.at[x.name, 'FUZ_MACD_CROSS_ZERO_G1'] = fz1[1]
      df.at[x.name, 'FUZ_MACD_CROSS_ZERO_G2'] = fz1[2]
      df.at[x.name, 'FUZ_MACD_CROSS_ZERO_G3'] = fz1[3]
      df.at[x.name, 'FUZ_MACD_CROSS_ZERO_G4'] = fz1[4]       
      df.at[x.name, 'FUZ_MACD_CROSS_ZERO_G5'] = fz1[5]       
      df.at[x.name, 'FUZ_MACD_CROSS_ZERO_G6'] = fz1[6]       
      df.at[x.name, 'FUZ_MACD_CROSS_ZERO_S-3'] = x.zbblo3
      df.at[x.name, 'FUZ_MACD_CROSS_ZERO_S-2'] = x.zbblo2
      df.at[x.name, 'FUZ_MACD_CROSS_ZERO_S-1'] = x.zbblo1
      df.at[x.name, 'FUZ_MACD_CROSS_ZERO_S0'] = x.ZERO
      df.at[x.name, 'FUZ_MACD_CROSS_ZERO_S+1'] = x.zbbup1
      df.at[x.name, 'FUZ_MACD_CROSS_ZERO_S+2'] = x.zbbup2
      df.at[x.name, 'FUZ_MACD_CROSS_ZERO_S+3'] = x.zbbup3
      logger.debug('fuzzifying row[{}]=> crisp_sig={}'.format(x.name, x.macd_main_sig))
      f_sets = [{'type':'left-edge',      'name':'MacdFarBelowSig',    'p0': x.sbblo3, 'p1': x.sbblo2},
                {'type':'internal-3pt',   'name':'MacdMidDistBelowSig','p0': x.sbblo3, 'p1': x.sbblo2, 'p2': x.sbblo1},
                {'type':'trapezoid-left', 'name':'MacdNearBelowSig',   'p0': x.sbblo2, 'p1': x.sbblo1, 'p2': x.ZERO},
                {'type':'singleton',      'name':'MacdJustCrossedSig', 'p0': x.ZERO},
                {'type':'trapezoid-right','name':'MacdNearAboveSig',   'p0': x.ZERO,   'p1': x.sbbup1, 'p2': x.sbbup2},
                {'type':'internal-3pt',   'name':'MacdMidDistAboveSig','p0': x.sbbup1, 'p1': x.sbbup2, 'p2': x.sbbup3},
                {'type':'right-edge'  ,   'name':'MacdFarAboveSig',    'p0': x.sbbup2, 'p1': x.sbbup3}]
      fz1 = Fuzzifier.fuzzify(x.macd_main_sig, f_sets)
      df.at[x.name, 'FUZ_MACD_CROSS_SIG'] = x.macd_main_sig
      if x.MACD_CROSS_SIG_UP==True or x.MACD_CROSS_SIG_DN==True:
        fz1[3] = 1.0
      else:
        fz1[3] = 0.0
      df.at[x.name, 'FUZ_MACD_CROSS_SIG_G0'] = fz1[0]
      df.at[x.name, 'FUZ_MACD_CROSS_SIG_G1'] = fz1[1]
      df.at[x.name, 'FUZ_MACD_CROSS_SIG_G2'] = fz1[2]
      df.at[x.name, 'FUZ_MACD_CROSS_SIG_G3'] = fz1[3]
      df.at[x.name, 'FUZ_MACD_CROSS_SIG_G4'] = fz1[4]
      df.at[x.name, 'FUZ_MACD_CROSS_SIG_G5'] = fz1[5]
      df.at[x.name, 'FUZ_MACD_CROSS_SIG_G6'] = fz1[6]
      df.at[x.name, 'FUZ_MACD_CROSS_SIG_S-3'] = x.sbblo3
      df.at[x.name, 'FUZ_MACD_CROSS_SIG_S-2'] = x.sbblo2
      df.at[x.name, 'FUZ_MACD_CROSS_SIG_S-1'] = x.sbblo1
      df.at[x.name, 'FUZ_MACD_CROSS_SIG_S0'] = x.ZERO
      df.at[x.name, 'FUZ_MACD_CROSS_SIG_S+1'] = x.sbbup1
      df.at[x.name, 'FUZ_MACD_CROSS_SIG_S+2'] = x.sbbup2
      df.at[x.name, 'FUZ_MACD_CROSS_SIG_S+3'] = x.sbbup3
     
    _df_result.apply(lambda x: fn_fuzzify_macd(x, self.__df, self.__logger), axis=1)
    return self.__df


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def fuzzifyRSI(self, timeperiod=500):
    """ Fuzzifies RSI into these fuzzy sets:
        - InOverSell
        - NearOverSell
        - FarFromLevels
        - NearOverBought
        - InOverBought
       Return:
        self.__df -- Updated dataframe      
    """
    _df_result = self.__df[['RSI']].copy()
    def fn_fuzzify_rsi(x, df, logger):
      logger.debug('fuzzifying row[{}]=> crisp_macd={}'.format(x.name, x.RSI))
      f_sets = [{'type':'left-edge',    'name':'RsiInOverSell',     'p0': 30.0, 'p1': 30.1},
                {'type':'internal-3pt', 'name':'RsiNearOverSell',   'p0': 30.0, 'p1': 30.1, 'p2': 50.0},
                {'type':'internal-3pt', 'name':'RsiFarFromLevels',  'p0': 30.0, 'p1': 50.0, 'p2': 70.0},
                {'type':'internal-3pt', 'name':'RsiNearOverBought', 'p0': 50.0, 'p1': 69.9, 'p2': 70.0},
                {'type':'right-edge'  , 'name':'RsiInOverBought',   'p0': 69.9, 'p1': 70.0}]
      fz1 = Fuzzifier.fuzzify(x.RSI, f_sets)
      df.at[x.name, 'FUZ_RSI'] = x.RSI
      df.at[x.name, 'FUZ_RSI_G0'] = fz1[0]
      df.at[x.name, 'FUZ_RSI_G1'] = fz1[1]
      df.at[x.name, 'FUZ_RSI_G2'] = fz1[2]
      df.at[x.name, 'FUZ_RSI_G3'] = fz1[3]
      df.at[x.name, 'FUZ_RSI_G4'] = fz1[4]       
      df.at[x.name, 'FUZ_RSI_S-2'] = 30.0
      df.at[x.name, 'FUZ_RSI_S-1'] = 30.1
      df.at[x.name, 'FUZ_RSI_S0'] = 50.0
      df.at[x.name, 'FUZ_RSI_S+1'] = 69.9
      df.at[x.name, 'FUZ_RSI_S+2'] = 70.0      
     
    _df_result.apply(lambda x: fn_fuzzify_rsi(x, self.__df, self.__logger), axis=1)
    return self.__df


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def fuzzifyMovingAverages(self, timeperiod=4, loopback_bb_mult = 25):
    """ Fuzzifies SMA_BULLISH_TREND and SMA_BEARISH_TREND into these fuzzy sets:
        - NoTrend
        - SmallTrend
        - MediumTrend
        - HighTrend
        Also builds one column for each SMA showing if price is near to each one.
        These variables are named: SMA_SLOW_DISTANCE, SMA_MID_DISTANCE and SMA_FAST_DISTANCE, with
        fuzzy sets:
        - FarAbove
        - NearAbove
        - Touching
        - NearBelow
        - FarBelow

       Return:
        self.__df -- Updated dataframe      
    """
    _df_result = self.__df[['SMA_BULLISH_TREND', 'SMA_BEARISH_TREND', 'SMA_SLOW', 'SMA_MID', 'SMA_FAST', 'CLOSE']].copy()
    _df_result['SMA_SLOW_DISTANCE'] = _df_result['CLOSE'] - _df_result['SMA_SLOW']
    _df_result['SMA_MID_DISTANCE'] = _df_result['CLOSE'] - _df_result['SMA_MID']
    _df_result['SMA_FAST_DISTANCE'] = _df_result['CLOSE'] - _df_result['SMA_FAST']
    _df_result['SMA_BULLISH_TREND'] = talib.SMA(_df_result['SMA_BULLISH_TREND'], timeperiod=timeperiod)
    _df_result['SMA_BEARISH_TREND'] = talib.SMA(_df_result['SMA_BEARISH_TREND'], timeperiod=timeperiod)
    _max_s = 0.25 * np.mean(np.clip(_df_result.SMA_SLOW_DISTANCE, 0.0, None))
    _max_m = 0.25 * np.mean(np.clip(_df_result.SMA_MID_DISTANCE, 0.0, None))
    _max_f = 0.25 * np.mean(np.clip(_df_result.SMA_FAST_DISTANCE, 0.0, None))

    _df_result['sma_slow_up2'], _, _ = talib.BBANDS(np.clip(_df_result.SMA_SLOW_DISTANCE, _max_s, None), timeperiod=timeperiod*loopback_bb_mult, nbdevup=0.61, nbdevdn=0.61, matype=0)    
    _df_result['sma_mid_up2'], _, _ = talib.BBANDS(np.clip(_df_result.SMA_MID_DISTANCE, _max_m, None), timeperiod=timeperiod*loopback_bb_mult, nbdevup=0.61, nbdevdn=0.61, matype=0)    
    _df_result['sma_fast_up2'], _, _ = talib.BBANDS(np.clip(_df_result.SMA_FAST_DISTANCE, _max_f, None), timeperiod=timeperiod*loopback_bb_mult, nbdevup=0.61, nbdevdn=0.61, matype=0)    
    _df_result['sma_slow_up1'] = 0.1 * _df_result['sma_slow_up2']
    _df_result['sma_mid_up1'] = 0.1 * _df_result['sma_mid_up2']
    _df_result['sma_fast_up1'] = 0.1 * _df_result['sma_fast_up2']

    _min_s = 0.25 * np.mean(np.clip(_df_result.SMA_SLOW_DISTANCE, None, 0.0))
    _min_m = 0.25 * np.mean(np.clip(_df_result.SMA_MID_DISTANCE, None, 0.0))
    _min_f = 0.25 * np.mean(np.clip(_df_result.SMA_FAST_DISTANCE, None, 0.0))
    _, _, _df_result['sma_slow_dn2'] = talib.BBANDS(np.clip(_df_result.SMA_SLOW_DISTANCE, None, _min_s), timeperiod=timeperiod*loopback_bb_mult, nbdevup=0.61, nbdevdn=0.61, matype=0)    
    _, _, _df_result['sma_mid_dn2'] = talib.BBANDS(np.clip(_df_result.SMA_MID_DISTANCE, None, _min_m), timeperiod=timeperiod*loopback_bb_mult, nbdevup=0.61, nbdevdn=0.61, matype=0)    
    _, _, _df_result['sma_fast_dn2'] = talib.BBANDS(np.clip(_df_result.SMA_FAST_DISTANCE, None, _min_f), timeperiod=timeperiod*loopback_bb_mult, nbdevup=0.61, nbdevdn=0.61, matype=0)    
    _df_result['sma_slow_dn1'] = 0.1 * _df_result['sma_slow_dn2']
    _df_result['sma_mid_dn1'] = 0.1 * _df_result['sma_mid_dn2']
    _df_result['sma_fast_dn1'] = 0.1 * _df_result['sma_fast_dn2']

    def fn_fuzzify_sma(x, df, logger):
      logger.debug('fuzzifying row[{}]=> crisp_bull={} crips_bear={}'.format(x.name, x.SMA_BULLISH_TREND, x.SMA_BEARISH_TREND))
      f_sets = [{'type':'left-edge',    'name':'NoTrend',     'p0': 0.0, 'p1': 0.2},
                {'type':'internal-3pt', 'name':'SmallTrend',  'p0': 0.0, 'p1': 0.2, 'p2': 0.5},
                {'type':'internal-3pt', 'name':'MediumTrend', 'p0': 0.2, 'p1': 0.5, 'p2': 0.8},
                {'type':'right-edge'  , 'name':'HighTrend',   'p0': 0.8, 'p1': 1.0}]
      fz1 = Fuzzifier.fuzzify(x.SMA_BULLISH_TREND, f_sets)
      fz2 = Fuzzifier.fuzzify(x.SMA_BEARISH_TREND, f_sets)
      df.at[x.name, 'FUZ_SMA_BULLISH_TREND'] = x.SMA_BULLISH_TREND
      df.at[x.name, 'FUZ_SMA_BULLISH_TREND_G0'] = fz1[0]
      df.at[x.name, 'FUZ_SMA_BULLISH_TREND_G1'] = fz1[1]
      df.at[x.name, 'FUZ_SMA_BULLISH_TREND_G2'] = fz1[2]
      df.at[x.name, 'FUZ_SMA_BULLISH_TREND_G3'] = fz1[3]
      df.at[x.name, 'FUZ_SMA_BULLISH_TREND_S0'] = 0.0
      df.at[x.name, 'FUZ_SMA_BULLISH_TREND_S+1'] = 0.2
      df.at[x.name, 'FUZ_SMA_BULLISH_TREND_S+2'] = 0.5      
      df.at[x.name, 'FUZ_SMA_BULLISH_TREND_S+3'] = 1.0      
      df.at[x.name, 'FUZ_SMA_BEARISH_TREND'] = x.SMA_BEARISH_TREND
      df.at[x.name, 'FUZ_SMA_BEARISH_TREND_G0'] = fz2[0]
      df.at[x.name, 'FUZ_SMA_BEARISH_TREND_G1'] = fz2[1]
      df.at[x.name, 'FUZ_SMA_BEARISH_TREND_G2'] = fz2[2]
      df.at[x.name, 'FUZ_SMA_BEARISH_TREND_G3'] = fz2[3]
      df.at[x.name, 'FUZ_SMA_BEARISH_TREND_S0'] = 0.0
      df.at[x.name, 'FUZ_SMA_BEARISH_TREND_S+1'] = 0.2
      df.at[x.name, 'FUZ_SMA_BEARISH_TREND_S+2'] = 0.5      
      df.at[x.name, 'FUZ_SMA_BEARISH_TREND_S+3'] = 1.0   

      f_sets = [{'type':'left-edge',      'name':'FarBelow',   'p0': x.sma_slow_dn2, 'p1': x.sma_slow_dn1},
                {'type':'internal-3pt',   'name':'NearBelow',  'p0': x.sma_slow_dn2, 'p1': x.sma_slow_dn1,  'p2': 0.0},
                {'type':'internal-3pt',   'name':'Touching',   'p0': x.sma_slow_dn1, 'p1': 0.0,             'p2': x.sma_slow_up1},
                {'type':'internal-3pt',   'name':'NearAbove',  'p0': 0.0,            'p1': x.sma_slow_up1,  'p2': x.sma_slow_up2},
                {'type':'right-edge'  ,   'name':'FarAbove',   'p0': x.sma_slow_up1, 'p1': x.sma_slow_up2}]
      fz1 = Fuzzifier.fuzzify(x.SMA_SLOW_DISTANCE, f_sets)
      df.at[x.name, 'FUZ_SMA_SLOW_DISTANCE'] = x.SMA_SLOW_DISTANCE
      df.at[x.name, 'FUZ_SMA_SLOW_DISTANCE_G0'] = fz1[0]
      df.at[x.name, 'FUZ_SMA_SLOW_DISTANCE_G1'] = fz1[1]
      df.at[x.name, 'FUZ_SMA_SLOW_DISTANCE_G2'] = fz1[2]
      df.at[x.name, 'FUZ_SMA_SLOW_DISTANCE_G3'] = fz1[3]
      df.at[x.name, 'FUZ_SMA_SLOW_DISTANCE_G4'] = fz1[4]       
      df.at[x.name, 'FUZ_SMA_SLOW_DISTANCE_S-2'] = x.sma_slow_dn2
      df.at[x.name, 'FUZ_SMA_SLOW_DISTANCE_S-1'] = x.sma_slow_dn1
      df.at[x.name, 'FUZ_SMA_SLOW_DISTANCE_S0'] = 0.0
      df.at[x.name, 'FUZ_SMA_SLOW_DISTANCE_S+1'] = x.sma_slow_up1
      df.at[x.name, 'FUZ_SMA_SLOW_DISTANCE_S+2'] = x.sma_slow_up2   

      f_sets = [{'type':'left-edge',      'name':'FarBelow',   'p0': x.sma_mid_dn2, 'p1': x.sma_mid_dn1},
                {'type':'internal-3pt',   'name':'NearBelow',  'p0': x.sma_mid_dn2, 'p1': x.sma_mid_dn1,  'p2': 0.0},
                {'type':'internal-3pt',   'name':'Touching',   'p0': x.sma_mid_dn1, 'p1': 0.0,             'p2': x.sma_mid_up1},
                {'type':'internal-3pt',   'name':'NearAbove',  'p0': 0.0,            'p1': x.sma_mid_up1,  'p2': x.sma_mid_up2},
                {'type':'right-edge'  ,   'name':'FarAbove',   'p0': x.sma_mid_up1, 'p1': x.sma_mid_up2}]
      fz1 = Fuzzifier.fuzzify(x.SMA_SLOW_DISTANCE, f_sets)
      df.at[x.name, 'FUZ_SMA_MID_DISTANCE'] = x.SMA_MID_DISTANCE
      df.at[x.name, 'FUZ_SMA_MID_DISTANCE_G0'] = fz1[0]
      df.at[x.name, 'FUZ_SMA_MID_DISTANCE_G1'] = fz1[1]
      df.at[x.name, 'FUZ_SMA_MID_DISTANCE_G2'] = fz1[2]
      df.at[x.name, 'FUZ_SMA_MID_DISTANCE_G3'] = fz1[3]
      df.at[x.name, 'FUZ_SMA_MID_DISTANCE_G4'] = fz1[4]       
      df.at[x.name, 'FUZ_SMA_MID_DISTANCE_S-2'] = x.sma_mid_dn2
      df.at[x.name, 'FUZ_SMA_MID_DISTANCE_S-1'] = x.sma_mid_dn1
      df.at[x.name, 'FUZ_SMA_MID_DISTANCE_S0'] = 0.0
      df.at[x.name, 'FUZ_SMA_MID_DISTANCE_S+1'] = x.sma_mid_up1
      df.at[x.name, 'FUZ_SMA_MID_DISTANCE_S+2'] = x.sma_mid_up2   

      f_sets = [{'type':'left-edge',      'name':'FarBelow',   'p0': x.sma_fast_dn2, 'p1': x.sma_fast_dn1},
                {'type':'internal-3pt',   'name':'NearBelow',  'p0': x.sma_fast_dn2, 'p1': x.sma_fast_dn1,  'p2': 0.0},
                {'type':'internal-3pt',   'name':'Touching',   'p0': x.sma_fast_dn1, 'p1': 0.0,             'p2': x.sma_fast_up1},
                {'type':'internal-3pt',   'name':'NearAbove',  'p0': 0.0,            'p1': x.sma_fast_up1,  'p2': x.sma_fast_up2},
                {'type':'right-edge'  ,   'name':'FarAbove',   'p0': x.sma_fast_up1, 'p1': x.sma_fast_up2}]
      fz1 = Fuzzifier.fuzzify(x.SMA_FAST_DISTANCE, f_sets)
      df.at[x.name, 'FUZ_SMA_FAST_DISTANCE'] = x.SMA_FAST_DISTANCE
      df.at[x.name, 'FUZ_SMA_FAST_DISTANCE_G0'] = fz1[0]
      df.at[x.name, 'FUZ_SMA_FAST_DISTANCE_G1'] = fz1[1]
      df.at[x.name, 'FUZ_SMA_FAST_DISTANCE_G2'] = fz1[2]
      df.at[x.name, 'FUZ_SMA_FAST_DISTANCE_G3'] = fz1[3]
      df.at[x.name, 'FUZ_SMA_FAST_DISTANCE_G4'] = fz1[4]       
      df.at[x.name, 'FUZ_SMA_FAST_DISTANCE_S-2'] = x.sma_fast_dn2
      df.at[x.name, 'FUZ_SMA_FAST_DISTANCE_S-1'] = x.sma_fast_dn1
      df.at[x.name, 'FUZ_SMA_FAST_DISTANCE_S0'] = 0.0
      df.at[x.name, 'FUZ_SMA_FAST_DISTANCE_S+1'] = x.sma_fast_up1
      df.at[x.name, 'FUZ_SMA_FAST_DISTANCE_S+2'] = x.sma_fast_up2   
           
    _df_result.apply(lambda x: fn_fuzzify_sma(x, self.__df, self.__logger), axis=1)
    return self.__df


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def fuzzifyFibo(self):
    """ Fuzzifies FIBO_x variables into these fuzzy sets:
        - InFiboLevel
        - NearFiboLevel
        - FarFromFiboLevel
       Return:
        self.__df -- Updated dataframe      
    """
    cols = [c for c in self.__df.columns if 'FIBO_' in c]
    _df_result = self.__df[cols].copy()
    def fn_fuzzify_fibo(x, df, logger):
      logger.debug('fuzzifying row[{}]=> crisp={}'.format(x.name, x.FIBO_CURR))
      def get_sets(value_inf, value, value_sup):
        si = (value - value_inf)/3
        ss = (value_sup - value)/3
        return  [{'type':'left-edge',    'name':'FarFromFiboLevel','p0': value_inf+si, 'p1': value-si},
                {'type':'internal-3pt', 'name':'NearFiboLevel',   'p0': value_inf+si, 'p1': value-si, 'p2': value},
                {'type':'internal-3pt', 'name':'InFiboLevel',     'p0': value-si, 'p1': value, 'p2': value+ss},
                {'type':'internal-3pt', 'name':'NearFiboLevel',   'p0': value, 'p1': value+ss, 'p2': value_sup-ss},
                {'type':'right-edge'  , 'name':'FarFromFiboLevel','p0': value+ss, 'p1': value_sup-ss}]
      df.at[x.name, 'FUZ_FIBO_RETR'] = x.FIBO_CURR
      df.at[x.name, 'FUZ_FIBO_EXTN'] = x.FIBO_CURR
      f_sets = get_sets(0.14, 0.23, 0.38)
      fz1 = Fuzzifier.fuzzify(x.FIBO_CURR, f_sets)
      df.at[x.name, 'FUZ_FIBO_023_G0'] = fz1[0]
      df.at[x.name, 'FUZ_FIBO_023_G1'] = fz1[1]
      df.at[x.name, 'FUZ_FIBO_023_G2'] = fz1[2]
      df.at[x.name, 'FUZ_FIBO_023_G3'] = fz1[3]
      df.at[x.name, 'FUZ_FIBO_023_G4'] = fz1[4]
      df.at[x.name, 'FUZ_FIBO_023_S-2'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_FIBO_023_S-1'] = f_sets[1]['p1']      
      df.at[x.name, 'FUZ_FIBO_023_S0'] = f_sets[2]['p1']
      df.at[x.name, 'FUZ_FIBO_023_S+1'] = f_sets[3]['p1']
      df.at[x.name, 'FUZ_FIBO_023_S+2'] = f_sets[4]['p1']   
      nearest_level = 'FUZ_FIBO_023'   
      nearest_value = abs(x.FIBO_CURR - 0.23)
      f_sets = get_sets(0.23, 0.38, 0.50)
      fz1 = Fuzzifier.fuzzify(x.FIBO_CURR, f_sets)
      df.at[x.name, 'FUZ_FIBO_038_G0'] = fz1[0]
      df.at[x.name, 'FUZ_FIBO_038_G1'] = fz1[1]
      df.at[x.name, 'FUZ_FIBO_038_G2'] = fz1[2]
      df.at[x.name, 'FUZ_FIBO_038_G3'] = fz1[3]
      df.at[x.name, 'FUZ_FIBO_038_G4'] = fz1[4]
      df.at[x.name, 'FUZ_FIBO_038_S-2'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_FIBO_038_S-1'] = f_sets[1]['p1']      
      df.at[x.name, 'FUZ_FIBO_038_S0'] = f_sets[2]['p1']
      df.at[x.name, 'FUZ_FIBO_038_S+1'] = f_sets[3]['p1']
      df.at[x.name, 'FUZ_FIBO_038_S+2'] = f_sets[4]['p1']      
      if abs(x.FIBO_CURR - 0.38) < nearest_value:
        nearest_value = abs(x.FIBO_CURR - 0.38)
        nearest_level = 'FUZ_FIBO_038'   
      f_sets = get_sets(0.38, 0.50, 0.61)
      fz1 = Fuzzifier.fuzzify(x.FIBO_CURR, f_sets)
      df.at[x.name, 'FUZ_FIBO_050_G0'] = fz1[0]
      df.at[x.name, 'FUZ_FIBO_050_G1'] = fz1[1]
      df.at[x.name, 'FUZ_FIBO_050_G2'] = fz1[2]
      df.at[x.name, 'FUZ_FIBO_050_G3'] = fz1[3]
      df.at[x.name, 'FUZ_FIBO_050_G4'] = fz1[4]
      df.at[x.name, 'FUZ_FIBO_050_S-2'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_FIBO_050_S-1'] = f_sets[1]['p1']      
      df.at[x.name, 'FUZ_FIBO_050_S0'] = f_sets[2]['p1']
      df.at[x.name, 'FUZ_FIBO_050_S+1'] = f_sets[3]['p1']
      df.at[x.name, 'FUZ_FIBO_050_S+2'] = f_sets[4]['p1']      
      if abs(x.FIBO_CURR - 0.50) < nearest_value:
        nearest_value = abs(x.FIBO_CURR - 0.50)
        nearest_level = 'FUZ_FIBO_050'   
      f_sets = get_sets(0.50, 0.61, 0.78)
      fz1 = Fuzzifier.fuzzify(x.FIBO_CURR, f_sets)
      df.at[x.name, 'FUZ_FIBO_061_G0'] = fz1[0]
      df.at[x.name, 'FUZ_FIBO_061_G1'] = fz1[1]
      df.at[x.name, 'FUZ_FIBO_061_G2'] = fz1[2]
      df.at[x.name, 'FUZ_FIBO_061_G3'] = fz1[3]
      df.at[x.name, 'FUZ_FIBO_061_G4'] = fz1[4]
      df.at[x.name, 'FUZ_FIBO_061_S-2'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_FIBO_061_S-1'] = f_sets[1]['p1']      
      df.at[x.name, 'FUZ_FIBO_061_S0'] = f_sets[2]['p1']
      df.at[x.name, 'FUZ_FIBO_061_S+1'] = f_sets[3]['p1']
      df.at[x.name, 'FUZ_FIBO_061_S+2'] = f_sets[4]['p1']      
      if abs(x.FIBO_CURR - 0.61) < nearest_value:
        nearest_value = abs(x.FIBO_CURR - 0.61)
        nearest_level = 'FUZ_FIBO_061'   
      f_sets = get_sets(0.61, 0.78, 0.86)
      fz1 = Fuzzifier.fuzzify(x.FIBO_CURR, f_sets)
      df.at[x.name, 'FUZ_FIBO_078_G0'] = fz1[0]
      df.at[x.name, 'FUZ_FIBO_078_G1'] = fz1[1]
      df.at[x.name, 'FUZ_FIBO_078_G2'] = fz1[2]
      df.at[x.name, 'FUZ_FIBO_078_G3'] = fz1[3]
      df.at[x.name, 'FUZ_FIBO_078_G4'] = fz1[4]
      df.at[x.name, 'FUZ_FIBO_078_S-2'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_FIBO_078_S-1'] = f_sets[1]['p1']      
      df.at[x.name, 'FUZ_FIBO_078_S0'] = f_sets[2]['p1']
      df.at[x.name, 'FUZ_FIBO_078_S+1'] = f_sets[3]['p1']
      df.at[x.name, 'FUZ_FIBO_078_S+2'] = f_sets[4]['p1']      
      if abs(x.FIBO_CURR - 0.78) < nearest_value:
        nearest_value = abs(x.FIBO_CURR - 0.78)
        nearest_level = 'FUZ_FIBO_078' 

      df.at[x.name, 'FUZ_FIBO_RETR_G0'] = df['{}_G0'.format(nearest_level)].iloc[x.name]
      df.at[x.name, 'FUZ_FIBO_RETR_G1'] = df['{}_G1'.format(nearest_level)].iloc[x.name]
      df.at[x.name, 'FUZ_FIBO_RETR_G2'] = df['{}_G2'.format(nearest_level)].iloc[x.name]
      df.at[x.name, 'FUZ_FIBO_RETR_G3'] = df['{}_G3'.format(nearest_level)].iloc[x.name]
      df.at[x.name, 'FUZ_FIBO_RETR_G4'] = df['{}_G4'.format(nearest_level)].iloc[x.name]
      df.at[x.name, 'FUZ_FIBO_RETR_S-2'] = df['{}_S-2'.format(nearest_level)].iloc[x.name]
      df.at[x.name, 'FUZ_FIBO_RETR_S-1'] = df['{}_S-1'.format(nearest_level)].iloc[x.name]    
      df.at[x.name, 'FUZ_FIBO_RETR_S0'] = df['{}_S0'.format(nearest_level)].iloc[x.name]
      df.at[x.name, 'FUZ_FIBO_RETR_S+1'] = df['{}_S+1'.format(nearest_level)].iloc[x.name]
      df.at[x.name, 'FUZ_FIBO_RETR_S+2'] = df['{}_S+2'.format(nearest_level)].iloc[x.name]     
      f_sets = get_sets(1.14, 1.23, 1.38)
      fz1 = Fuzzifier.fuzzify(x.FIBO_CURR, f_sets)
      df.at[x.name, 'FUZ_FIBO_123_G0'] = fz1[0]
      df.at[x.name, 'FUZ_FIBO_123_G1'] = fz1[1]
      df.at[x.name, 'FUZ_FIBO_123_G2'] = fz1[2]
      df.at[x.name, 'FUZ_FIBO_123_G3'] = fz1[3]
      df.at[x.name, 'FUZ_FIBO_123_G4'] = fz1[4]
      df.at[x.name, 'FUZ_FIBO_123_S-2'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_FIBO_123_S-1'] = f_sets[1]['p1']      
      df.at[x.name, 'FUZ_FIBO_123_S0'] = f_sets[2]['p1']
      df.at[x.name, 'FUZ_FIBO_123_S+1'] = f_sets[3]['p1']
      df.at[x.name, 'FUZ_FIBO_123_S+2'] = f_sets[4]['p1']   
      nearest_level = 'FUZ_FIBO_123'   
      nearest_value = abs(x.FIBO_CURR - 1.23)
      f_sets = get_sets(1.23, 1.38, 1.50)
      fz1 = Fuzzifier.fuzzify(x.FIBO_CURR, f_sets)
      df.at[x.name, 'FUZ_FIBO_138_G0'] = fz1[0]
      df.at[x.name, 'FUZ_FIBO_138_G1'] = fz1[1]
      df.at[x.name, 'FUZ_FIBO_138_G2'] = fz1[2]
      df.at[x.name, 'FUZ_FIBO_138_G3'] = fz1[3]
      df.at[x.name, 'FUZ_FIBO_138_G4'] = fz1[4]
      df.at[x.name, 'FUZ_FIBO_138_S-2'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_FIBO_138_S-1'] = f_sets[1]['p1']      
      df.at[x.name, 'FUZ_FIBO_138_S0'] = f_sets[2]['p1']
      df.at[x.name, 'FUZ_FIBO_138_S+1'] = f_sets[3]['p1']
      df.at[x.name, 'FUZ_FIBO_138_S+2'] = f_sets[4]['p1']      
      if abs(x.FIBO_CURR - 1.38) < nearest_value:
        nearest_value = abs(x.FIBO_CURR - 1.38)
        nearest_level = 'FUZ_FIBO_138'   
      f_sets = get_sets(1.38, 1.50, 1.61)
      fz1 = Fuzzifier.fuzzify(x.FIBO_CURR, f_sets)
      df.at[x.name, 'FUZ_FIBO_150_G0'] = fz1[0]
      df.at[x.name, 'FUZ_FIBO_150_G1'] = fz1[1]
      df.at[x.name, 'FUZ_FIBO_150_G2'] = fz1[2]
      df.at[x.name, 'FUZ_FIBO_150_G3'] = fz1[3]
      df.at[x.name, 'FUZ_FIBO_150_G4'] = fz1[4]
      df.at[x.name, 'FUZ_FIBO_150_S-2'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_FIBO_150_S-1'] = f_sets[1]['p1']      
      df.at[x.name, 'FUZ_FIBO_150_S0'] = f_sets[2]['p1']
      df.at[x.name, 'FUZ_FIBO_150_S+1'] = f_sets[3]['p1']
      df.at[x.name, 'FUZ_FIBO_150_S+2'] = f_sets[4]['p1']      
      if abs(x.FIBO_CURR - 1.50) < nearest_value:
        nearest_value = abs(x.FIBO_CURR - 1.50)
        nearest_level = 'FUZ_FIBO_150'   
      f_sets = get_sets(1.50, 1.61, 1.78)
      fz1 = Fuzzifier.fuzzify(x.FIBO_CURR, f_sets)
      df.at[x.name, 'FUZ_FIBO_161_G0'] = fz1[0]
      df.at[x.name, 'FUZ_FIBO_161_G1'] = fz1[1]
      df.at[x.name, 'FUZ_FIBO_161_G2'] = fz1[2]
      df.at[x.name, 'FUZ_FIBO_161_G3'] = fz1[3]
      df.at[x.name, 'FUZ_FIBO_161_G4'] = fz1[4]
      df.at[x.name, 'FUZ_FIBO_161_S-2'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_FIBO_161_S-1'] = f_sets[1]['p1']      
      df.at[x.name, 'FUZ_FIBO_161_S0'] = f_sets[2]['p1']
      df.at[x.name, 'FUZ_FIBO_161_S+1'] = f_sets[3]['p1']
      df.at[x.name, 'FUZ_FIBO_161_S+2'] = f_sets[4]['p1']      
      if abs(x.FIBO_CURR - 1.61) < nearest_value:
        nearest_value = abs(x.FIBO_CURR - 1.61)
        nearest_level = 'FUZ_FIBO_161'   
      f_sets = get_sets(1.61, 1.78, 1.86)
      fz1 = Fuzzifier.fuzzify(x.FIBO_CURR, f_sets)
      df.at[x.name, 'FUZ_FIBO_178_G0'] = fz1[0]
      df.at[x.name, 'FUZ_FIBO_178_G1'] = fz1[1]
      df.at[x.name, 'FUZ_FIBO_178_G2'] = fz1[2]
      df.at[x.name, 'FUZ_FIBO_178_G3'] = fz1[3]
      df.at[x.name, 'FUZ_FIBO_178_G4'] = fz1[4]
      df.at[x.name, 'FUZ_FIBO_178_S-2'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_FIBO_178_S-1'] = f_sets[1]['p1']      
      df.at[x.name, 'FUZ_FIBO_178_S0'] = f_sets[2]['p1']
      df.at[x.name, 'FUZ_FIBO_178_S+1'] = f_sets[3]['p1']
      df.at[x.name, 'FUZ_FIBO_178_S+2'] = f_sets[4]['p1']      
      if abs(x.FIBO_CURR - 1.78) < nearest_value:
        nearest_value = abs(x.FIBO_CURR - 1.78)
        nearest_level = 'FUZ_FIBO_178'        
      df.at[x.name, 'FUZ_FIBO_EXTN_G0'] = df['{}_G0'.format(nearest_level)].iloc[x.name]
      df.at[x.name, 'FUZ_FIBO_EXTN_G1'] = df['{}_G1'.format(nearest_level)].iloc[x.name]
      df.at[x.name, 'FUZ_FIBO_EXTN_G2'] = df['{}_G2'.format(nearest_level)].iloc[x.name]
      df.at[x.name, 'FUZ_FIBO_EXTN_G3'] = df['{}_G3'.format(nearest_level)].iloc[x.name]
      df.at[x.name, 'FUZ_FIBO_EXTN_G4'] = df['{}_G4'.format(nearest_level)].iloc[x.name]
      df.at[x.name, 'FUZ_FIBO_EXTN_S-2'] = df['{}_S-2'.format(nearest_level)].iloc[x.name]
      df.at[x.name, 'FUZ_FIBO_EXTN_S-1'] = df['{}_S-1'.format(nearest_level)].iloc[x.name]    
      df.at[x.name, 'FUZ_FIBO_EXTN_S0'] = df['{}_S0'.format(nearest_level)].iloc[x.name]
      df.at[x.name, 'FUZ_FIBO_EXTN_S+1'] = df['{}_S+1'.format(nearest_level)].iloc[x.name]
      df.at[x.name, 'FUZ_FIBO_EXTN_S+2'] = df['{}_S+2'.format(nearest_level)].iloc[x.name]          

    _df_result.apply(lambda x: fn_fuzzify_fibo(x, self.__df, self.__logger), axis=1)
    return self.__df


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def fuzzifySRLevels(self, timeperiod=2000):
    """ Build SR_DISTANCE indicator based on f_points displacements from zero-point.
        Requires previous execution of fuzzifyZIGZAG due to column FUZ_ZZ_RANGE
    """
    _df_result = self.__df[['CLOSE','SUPPORTS','RESISTANCES', 'FUZ_BOLLINGER_WIDTH_S-2']].copy()
    _df_result['filt_bbw'] = _df_result['FUZ_BOLLINGER_WIDTH_S-2'].apply(lambda x: max(x, 0.0))    
    _df_result['s3'], _, _df_result['s1'] = talib.BBANDS(_df_result['filt_bbw'], timeperiod=timeperiod, nbdevup=0.5, nbdevdn=0.5, matype=0)
    def fn_fuzzify_sr(x, df, logger):
      logger.debug('fuzzifying row[{}]=> crisp={}'.format(x.name, x.CLOSE))
      _subdf = df[0:x.name]
      _supports = _subdf[(_subdf['SUPPORTS'] > 0.0)]
      _resist = _subdf[(_subdf['RESISTANCES'] > 0.0)]
      if _supports.shape[0] < 2 or _resist.shape[0] < 2:
        return
      _sr = [_supports.SUPPORTS.iloc[-2], _supports.SUPPORTS.iloc[-1], _resist.RESISTANCES.iloc[-2], _resist.RESISTANCES.iloc[-2]]
      f_sets = [{'type':'left-edge',    'name':'FarFromFiboLevel','p0': -x.s3, 'p1': -x.s1},
                {'type':'internal-3pt', 'name':'NearFiboLevel',   'p0': -x.s3, 'p1': -x.s1, 'p2': 0.0},
                {'type':'internal-3pt', 'name':'InFiboLevel',     'p0': -x.s1, 'p1': 0.0, 'p2': x.s1},
                {'type':'internal-3pt', 'name':'NearFiboLevel',   'p0': 0.0, 'p1': x.s1, 'p2': x.s3},
                {'type':'right-edge'  , 'name':'FarFromFiboLevel','p0': x.s1, 'p1': x.s3}]
      nearest = np.inf
      for i in _sr:
        fz1 = Fuzzifier.fuzzify(x.CLOSE - i, f_sets)
        if abs(x.CLOSE - i) < nearest:
          nearest = abs(x.CLOSE - i)
          df.at[x.name, 'FUZ_SR_DISTANCE'] = x.CLOSE - i
          df.at[x.name, 'FUZ_SR_DISTANCE_G0'] = fz1[0]
          df.at[x.name, 'FUZ_SR_DISTANCE_G1'] = fz1[1]
          df.at[x.name, 'FUZ_SR_DISTANCE_G2'] = fz1[2]
          df.at[x.name, 'FUZ_SR_DISTANCE_G3'] = fz1[3]
          df.at[x.name, 'FUZ_SR_DISTANCE_G4'] = fz1[4]
          df.at[x.name, 'FUZ_SR_DISTANCE_S-2'] = f_sets[0]['p0']
          df.at[x.name, 'FUZ_SR_DISTANCE_S-1'] = f_sets[1]['p1']      
          df.at[x.name, 'FUZ_SR_DISTANCE_S0'] = f_sets[2]['p1']
          df.at[x.name, 'FUZ_SR_DISTANCE_S+1'] = f_sets[3]['p1']
          df.at[x.name, 'FUZ_SR_DISTANCE_S+2'] = f_sets[4]['p1']      
    _df_result.apply(lambda x: fn_fuzzify_sr(x, self.__df, self.__logger), axis=1)
    return self.__df


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def fuzzifyChannel(self):
    """ Build CHHI_DISTANCE and CHLO_DISTANCE fuzzy-variables within these fuzzy sets:
      - FarBelow
      - NearBelow
      - In
      - NearAbove
      - FarAbove
    """
    _df_result = self.__df[['CLOSE','CHANNEL_UPPER_LIMIT','CHANNEL_LOWER_LIMIT','P1','P2','P3','P4','P1_idx','P2_idx','P3_idx','P4_idx']].copy()
    def fn_fuzzify_channel(x, df, logger):
      logger.debug('fuzzifying row[{}]=> crisp={}'.format(x.name, x.CLOSE))
      if x.CHANNEL_UPPER_LIMIT == 'P3,P1':
        ux0 = x.P3_idx
        ux1 = x.P1_idx
        uy0 = x.P3
        uy1 = x.P1
      elif x.CHANNEL_UPPER_LIMIT == 'P4,P2':
        ux0 = x.P4_idx
        ux1 = x.P2_idx
        uy0 = x.P4
        uy1 = x.P2
      else:
        return
      if x.CHANNEL_LOWER_LIMIT == 'P3,P1':
        lx0 = x.P3_idx
        lx1 = x.P1_idx
        ly0 = x.P3
        ly1 = x.P1
      elif x.CHANNEL_LOWER_LIMIT == 'P4,P2':
        lx0 = x.P4_idx
        lx1 = x.P2_idx
        ly0 = x.P4
        ly1 = x.P2
      else:
        return

      uy_at = (((uy1-uy0)/(ux1-ux0))*((x.name)-ux0))+uy0
      ly_at = (((ly1-ly0)/(lx1-lx0))*((x.name)-lx0))+ly0
      _ch_step = abs((uy_at - ly_at)/4)
      f_sets = [{'type':'left-edge',    'name':'FarBelow',  'p0': -(2*_ch_step), 'p1': -_ch_step},
                {'type':'internal-3pt', 'name':'NearBelow', 'p0': -(2*_ch_step), 'p1': -_ch_step, 'p2': 0.0},
                {'type':'internal-3pt', 'name':'In',        'p0': -_ch_step, 'p1': 0.0, 'p2': _ch_step},
                {'type':'internal-3pt', 'name':'NearAbove', 'p0': 0.0, 'p1': _ch_step, 'p2': (2*_ch_step)},
                {'type':'right-edge'  , 'name':'FarAbove',  'p0': _ch_step, 'p1': (2*_ch_step)}]
      fz1 = Fuzzifier.fuzzify(x.CLOSE - uy_at, f_sets)
      df.at[x.name, 'FUZ_CHHI_DISTANCE'] = x.CLOSE - uy_at
      df.at[x.name, 'FUZ_CHHI_DISTANCE_G0'] = fz1[0]
      df.at[x.name, 'FUZ_CHHI_DISTANCE_G1'] = fz1[1]
      df.at[x.name, 'FUZ_CHHI_DISTANCE_G2'] = fz1[2]
      df.at[x.name, 'FUZ_CHHI_DISTANCE_G3'] = fz1[3]
      df.at[x.name, 'FUZ_CHHI_DISTANCE_G4'] = fz1[4]
      df.at[x.name, 'FUZ_CHHI_DISTANCE_S-2'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_CHHI_DISTANCE_S-1'] = f_sets[1]['p1']      
      df.at[x.name, 'FUZ_CHHI_DISTANCE_S0'] = f_sets[2]['p1']
      df.at[x.name, 'FUZ_CHHI_DISTANCE_S+1'] = f_sets[3]['p1']
      df.at[x.name, 'FUZ_CHHI_DISTANCE_S+2'] = f_sets[4]['p1']      
      fz1 = Fuzzifier.fuzzify(x.CLOSE - ly_at, f_sets)
      df.at[x.name, 'FUZ_CHLO_DISTANCE'] = x.CLOSE - ly_at
      df.at[x.name, 'FUZ_CHLO_DISTANCE_G0'] = fz1[0]
      df.at[x.name, 'FUZ_CHLO_DISTANCE_G1'] = fz1[1]
      df.at[x.name, 'FUZ_CHLO_DISTANCE_G2'] = fz1[2]
      df.at[x.name, 'FUZ_CHLO_DISTANCE_G3'] = fz1[3]
      df.at[x.name, 'FUZ_CHLO_DISTANCE_G4'] = fz1[4]
      df.at[x.name, 'FUZ_CHLO_DISTANCE_S-2'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_CHLO_DISTANCE_S-1'] = f_sets[1]['p1']      
      df.at[x.name, 'FUZ_CHLO_DISTANCE_S0'] = f_sets[2]['p1']
      df.at[x.name, 'FUZ_CHLO_DISTANCE_S+1'] = f_sets[3]['p1']
      df.at[x.name, 'FUZ_CHLO_DISTANCE_S+2'] = f_sets[4]['p1']      
    _df_result.apply(lambda x: fn_fuzzify_channel(x, self.__df, self.__logger), axis=1)
    return self.__df


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def fuzzifyTrend(self):
    """ Build TREND_STRENGTH fuzzy-variable within these fuzzy sets:
      - StrongBearish
      - MidBearish
      - WeakBearish
      - Undefined
      - WeakBullish
      - MidBullish
      - StrongBullish
    """
    _df_result = self.__df[['BULLISH_TREND','BEARISH_TREND']].copy()
    def fn_fuzzify_trend(x, df, logger):
      _trend = x.BULLISH_TREND - x.BEARISH_TREND
      logger.debug('fuzzifying row[{}]=> crisp={}'.format(x.name, _trend))
      f_sets = [{'type':'left-edge',    'name':'StrongBearish', 'p0': -1.0, 'p1': -0.7},
                {'type':'internal-3pt', 'name':'MidBearish',    'p0': -1.0, 'p1': -0.7, 'p2': -0.3},
                {'type':'internal-3pt', 'name':'WeakBearish',   'p0': -0.7, 'p1': -0.3, 'p2': 0.0},
                {'type':'internal-3pt', 'name':'Undefined',     'p0': -0.3, 'p1': 0.0, 'p2': 0.3},
                {'type':'internal-3pt', 'name':'WeakBullish',   'p0': 0.0, 'p1': 0.3, 'p2': 0.7},
                {'type':'internal-3pt', 'name':'MidBullish',    'p0': 0.3, 'p1': 0.7, 'p2': 1.0},
                {'type':'right-edge'  , 'name':'StrongBullish', 'p0': 0.7, 'p1': 1.0}]
      fz1 = Fuzzifier.fuzzify(_trend, f_sets)
      df.at[x.name, 'FUZ_TREND_STRENGTH'] = _trend
      df.at[x.name, 'FUZ_TREND_STRENGTH_G0'] = fz1[0]
      df.at[x.name, 'FUZ_TREND_STRENGTH_G1'] = fz1[1]
      df.at[x.name, 'FUZ_TREND_STRENGTH_G2'] = fz1[2]
      df.at[x.name, 'FUZ_TREND_STRENGTH_G3'] = fz1[3]
      df.at[x.name, 'FUZ_TREND_STRENGTH_G4'] = fz1[4]
      df.at[x.name, 'FUZ_TREND_STRENGTH_G5'] = fz1[5]
      df.at[x.name, 'FUZ_TREND_STRENGTH_G6'] = fz1[6]
      df.at[x.name, 'FUZ_TREND_STRENGTH_S-3'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_TREND_STRENGTH_S-2'] = f_sets[1]['p1']
      df.at[x.name, 'FUZ_TREND_STRENGTH_S-1'] = f_sets[2]['p1']      
      df.at[x.name, 'FUZ_TREND_STRENGTH_S0']  = f_sets[3]['p1']
      df.at[x.name, 'FUZ_TREND_STRENGTH_S+1'] = f_sets[4]['p1']
      df.at[x.name, 'FUZ_TREND_STRENGTH_S+2'] = f_sets[5]['p1']      
      df.at[x.name, 'FUZ_TREND_STRENGTH_S+3'] = f_sets[6]['p1']   
    _df_result.apply(lambda x: fn_fuzzify_trend(x, self.__df, self.__logger), axis=1)
    return self.__df


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def fuzzifyDivergence(self):
    """ Build BULL_DIV_STRENGTH and BEAR_DIV_STRENGTH fuzzy-variables within these fuzzy sets:
      - NoDiv
      - WeakDiv
      - StrongDiv
    """
    _df_result = self.__df[['BULLISH_DIVERGENCE', 'BEARISH_DIVERGENCE']].copy()
    def fn_fuzzify_divergence(x, df, logger):
      logger.debug('fuzzifying row[{}]=> crisp_bull={} crips_bear={}'.format(x.name, x.BULLISH_DIVERGENCE, x.BEARISH_DIVERGENCE))
      f_sets = [{'type':'left-edge',    'name':'NoDiv',     'p0': 0.0, 'p1': 0.3},
                {'type':'internal-3pt', 'name':'WeakDiv',   'p0': 0.0, 'p1': 0.5, 'p2': 1.0},
                {'type':'right-edge'  , 'name':'StrongDiv', 'p0': 0.7, 'p1': 1.0}]
      fz1 = Fuzzifier.fuzzify(x.BULLISH_DIVERGENCE, f_sets)
      df.at[x.name, 'FUZ_BULL_DIV_STRENGTH'] = x.BULLISH_DIVERGENCE
      df.at[x.name, 'FUZ_BULL_DIV_STRENGTH_G0'] = fz1[0]
      df.at[x.name, 'FUZ_BULL_DIV_STRENGTH_G1'] = fz1[1]
      df.at[x.name, 'FUZ_BULL_DIV_STRENGTH_G2'] = fz1[2]
      df.at[x.name, 'FUZ_BULL_DIV_STRENGTH_S0']  = f_sets[0]['p0']
      df.at[x.name, 'FUZ_BULL_DIV_STRENGTH_S+1'] = f_sets[1]['p1']
      df.at[x.name, 'FUZ_BULL_DIV_STRENGTH_S+2'] = f_sets[2]['p1']      
      fz1 = Fuzzifier.fuzzify(x.BEARISH_DIVERGENCE, f_sets)
      df.at[x.name, 'FUZ_BEAR_DIV_STRENGTH'] = x.BEARISH_DIVERGENCE
      df.at[x.name, 'FUZ_BEAR_DIV_STRENGTH_G0'] = fz1[0]
      df.at[x.name, 'FUZ_BEAR_DIV_STRENGTH_G1'] = fz1[1]
      df.at[x.name, 'FUZ_BEAR_DIV_STRENGTH_G2'] = fz1[2]
      df.at[x.name, 'FUZ_BEAR_DIV_STRENGTH_S0']  = f_sets[0]['p0']
      df.at[x.name, 'FUZ_BEAR_DIV_STRENGTH_S+1'] = f_sets[1]['p1']
      df.at[x.name, 'FUZ_BEAR_DIV_STRENGTH_S+2'] = f_sets[2]['p1']      
      
    _df_result.apply(lambda x: fn_fuzzify_divergence(x, self.__df, self.__logger), axis=1)
    return self.__df


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def fuzzifySessionWdows(self):
    """ Build TKLND_WDOW, LNDNY_WDOW, MKTOPEN_WDOW, MKTCLOSE_WDOW fuzzy-variables within these fuzzy sets:
      - FarAbove
      - NearAbove
      - In
      - NearBelow
      - FarBelow

      Each window has different timetables:
      Tokio from 22h to 08h GMT (MT4: 00h to 10h)
      Londres from 07h to 16h GMT (MT4: 09h to 18h)
      NY   from 12h to 23h GMT (MT4: 14h to 01h)
      OPEN on Sunday at 22h GMT (MT4: monday 00h)
      CLOSE on Friday at 21h GMT (MT4: friday 23:59)
    """
    _df_result = self.__df[['TIME']].copy()
    _df_result['TIME_OF_DAY'] = _df_result.apply(lambda x: (x.TIME.hour * 60.0 + x.TIME.minute)/1440, axis=1)
    _df_result['TIME_OF_WEEK'] = _df_result.apply(lambda x: (x.TIME.dayofweek*1440.0 + x.TIME.hour * 60.0 + x.TIME.minute)/7200, axis=1)
    def fn_fuzzify_sessions(x, df, logger):
      logger.debug('fuzzifying row[{}]=> time_of_day={} time_of_week={}'.format(x.name, x.TIME_OF_DAY, x.TIME_OF_WEEK))
      f_sets = [{'type':'left-edge',    'name':'FarAbove',  'p0': 450.0/1440, 'p1': 510.0/1440},
                {'type':'internal-3pt', 'name':'NearAbove', 'p0': 450.0/1440, 'p1': 510.0/1440, 'p2': 570.0/1440},
                {'type':'internal-3pt', 'name':'InRange',   'p0': 510.0/1440, 'p1': 570.0/1440, 'p2': 630.0/1440},
                {'type':'internal-3pt', 'name':'NearBelow', 'p0': 570.0/1440, 'p1': 630.0/1440, 'p2': 690.0/1440},
                {'type':'right-edge'  , 'name':'FarBelow',  'p0': 630.0/1440, 'p1': 690.0/1440}]
      fz1 = Fuzzifier.fuzzify(x.TIME_OF_DAY, f_sets)
      df.at[x.name, 'FUZ_TKLND_WDOW'] = x.TIME_OF_DAY
      df.at[x.name, 'FUZ_TKLND_WDOW_G0'] = fz1[0]
      df.at[x.name, 'FUZ_TKLND_WDOW_G1'] = fz1[1]
      df.at[x.name, 'FUZ_TKLND_WDOW_G2'] = fz1[2]
      df.at[x.name, 'FUZ_TKLND_WDOW_G3'] = fz1[3]
      df.at[x.name, 'FUZ_TKLND_WDOW_G4'] = fz1[4]       
      df.at[x.name, 'FUZ_TKLND_WDOW_S-2'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_TKLND_WDOW_S-1'] = f_sets[1]['p1']
      df.at[x.name, 'FUZ_TKLND_WDOW_S0'] = f_sets[2]['p1']
      df.at[x.name, 'FUZ_TKLND_WDOW_S+1'] = f_sets[3]['p1']
      df.at[x.name, 'FUZ_TKLND_WDOW_S+2'] = f_sets[4]['p1']    
      f_sets = [{'type':'left-edge',    'name':'FarAbove',  'p0': 750.0/1440, 'p1': 810.0/1440},
                {'type':'internal-3pt', 'name':'NearAbove', 'p0': 750.0/1440, 'p1': 810.0/1440, 'p2': 960.0/1440},
                {'type':'internal-3pt', 'name':'InRange',   'p0': 810.0/1440, 'p1': 960.0/1440, 'p2': 1110.0/1440},
                {'type':'internal-3pt', 'name':'NearBelow', 'p0': 960.0/1440, 'p1': 1110.0/1440, 'p2': 1170.0/1440},
                {'type':'right-edge'  , 'name':'FarBelow',  'p0': 1110.0/1440, 'p1': 1170.0/1440}]
      fz1 = Fuzzifier.fuzzify(x.TIME_OF_DAY, f_sets)
      df.at[x.name, 'FUZ_LNDNY_WDOW'] = x.TIME_OF_DAY
      df.at[x.name, 'FUZ_LNDNY_WDOW_G0'] = fz1[0]
      df.at[x.name, 'FUZ_LNDNY_WDOW_G1'] = fz1[1]
      df.at[x.name, 'FUZ_LNDNY_WDOW_G2'] = fz1[2]
      df.at[x.name, 'FUZ_LNDNY_WDOW_G3'] = fz1[3]
      df.at[x.name, 'FUZ_LNDNY_WDOW_G4'] = fz1[4]       
      df.at[x.name, 'FUZ_LNDNY_WDOW_S-2'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_LNDNY_WDOW_S-1'] = f_sets[1]['p1']
      df.at[x.name, 'FUZ_LNDNY_WDOW_S0'] = f_sets[2]['p1']
      df.at[x.name, 'FUZ_LNDNY_WDOW_S+1'] = f_sets[3]['p1']
      df.at[x.name, 'FUZ_LNDNY_WDOW_S+2'] = f_sets[4]['p1']    
      f_sets = [{'type':'left-edge',    'name':'InRange',  'p0': 0, 'p1': 120.0/7200},
                {'type':'internal-3pt', 'name':'Near', 'p0': 0, 'p1': 120.0/7200, 'p2': 240.0/7200},
                {'type':'right-edge'  , 'name':'Far',  'p0': 120.0/7200, 'p1': 240.0/7200}]
      fz1 = Fuzzifier.fuzzify(x.TIME_OF_WEEK, f_sets)
      df.at[x.name, 'FUZ_MKTOPEN_WDOW'] = x.TIME_OF_WEEK
      df.at[x.name, 'FUZ_MKTOPEN_WDOW_G0'] = fz1[0]
      df.at[x.name, 'FUZ_MKTOPEN_WDOW_G1'] = fz1[1]
      df.at[x.name, 'FUZ_MKTOPEN_WDOW_G2'] = fz1[2]
      df.at[x.name, 'FUZ_MKTOPEN_WDOW_S-2'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_MKTOPEN_WDOW_S-1'] = f_sets[1]['p1']
      df.at[x.name, 'FUZ_MKTOPEN_WDOW_S0'] = f_sets[2]['p1']
      f_sets = [{'type':'left-edge',    'name':'Far',     'p0': 6960.0/7200, 'p1': 7080.0/7200},
                {'type':'internal-3pt', 'name':'Near',    'p0': 6960.0/7200, 'p1': 7080.0/7200, 'p2': 7200.0/7200},
                {'type':'right-edge'  , 'name':'InRange', 'p0': 7080.0/7200, 'p1': 7200.0/7200}]
      fz1 = Fuzzifier.fuzzify(x.TIME_OF_WEEK, f_sets)
      df.at[x.name, 'FUZ_MKTCLOSE_WDOW'] = x.TIME_OF_WEEK
      df.at[x.name, 'FUZ_MKTCLOSE_WDOW_G0'] = fz1[0]
      df.at[x.name, 'FUZ_MKTCLOSE_WDOW_G1'] = fz1[1]
      df.at[x.name, 'FUZ_MKTCLOSE_WDOW_G2'] = fz1[2]
      df.at[x.name, 'FUZ_MKTCLOSE_WDOW_S-2'] = f_sets[0]['p0']
      df.at[x.name, 'FUZ_MKTCLOSE_WDOW_S-1'] = f_sets[1]['p1']
      df.at[x.name, 'FUZ_MKTCLOSE_WDOW_S0'] = f_sets[2]['p1']
      
    _df_result.apply(lambda x: fn_fuzzify_sessions(x, self.__df, self.__logger), axis=1)
    return self.__df


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildStateChangeEvent(self, level=0.5, threshold=0.1):
    """ Builds column MKT_STAT_CHANGE_EVT in which a boolean value establishes is a market state change
        is present, due to a change in fuzzy variable memberships functions:
        Key arguments:
          level -- Base Level of state change in membership value
          threshold -- Threshold to confirm state change
    """
    fuzzy_vars = self.listFuzzyVariables()
    _df_result = self.__df[fuzzy_vars].copy()
    prev_state = []
    def fn_state_change(x, df, fuzzy_vars, prev_state, logger):
      logger.debug('------------------------------------------')
      logger.debug('scanning row[{}]'.format(x.name))
      # check those vars in prev_state which becomes inactive
      removed_count = 0
      for fv in prev_state:
        if x[fv] < level - threshold:
          prev_state.remove(fv)
          removed_count += 1
          logger.debug('removed {} =>{}'.format(removed_count, fv))

      # selects winning fuzzy_vars of current_state
      curr_state = prev_state
      added_count  = 0
      for fv in fuzzy_vars:
        if fv not in curr_state:
          if x[fv] > level + threshold:
            curr_state.append(fv)
            added_count += 1
            logger.debug('added {} =>{}'.format(added_count, fv))

      # check differences from previous state
      if removed_count + added_count > 0:
        prev_state = curr_state
        df.at[x.name, 'MKT_STAT_CHANGE_EVT'] = True
        logger.debug('STATE_CHANGED')
      else:
        df.at[x.name, 'MKT_STAT_CHANGE_EVT'] = False
        logger.debug('STATE_KEEP')

    _df_result.apply(lambda x: fn_state_change(x, self.__df, fuzzy_vars, prev_state, self.__logger), axis=1)
    return self.__df

  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def fuzzifyIndicators(self):
    """ Build fuzzy-indicators
    """
    self.fuzzifyZigzag(timeperiod=50)
    self.fuzzifyBollinger(timeperiod=50)
    self.fuzzifyMACD(timeperiod=500)
    self.fuzzifyRSI(timeperiod=500)
    self.fuzzifyMovingAverages(timeperiod=14, loopback_bb_mult=7)
    self.fuzzifyFibo()
    self.fuzzifySRLevels(timeperiod=2000)
    self.fuzzifyChannel()
    self.fuzzifyTrend()
    self.fuzzifyDivergence()
    self.fuzzifySessionWdows()
    return self.__df

  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def plotFuzzyZigzagVariable(self, var, colors=['rgb(195, 243, 195)','rgb(245, 230, 180)','rgb(252, 180, 197)']):
    """ Plot stacked areas as fuzzy sets evolution agains range variable
    """
    df_zz = self.__df[(self.__df.ZIGZAG > 0.0) & (self.__df.ACTION.str.contains('-in-progress')==False)]
    trace_lo2 = go.Scatter(x=df_zz.index.values, y=df_zz['FUZ_{}_S-2'.format(var)], 
                      fill='none', mode='lines', name='-2std', line=dict(width=4, color=colors[2]))
    trace_lo1 = go.Scatter(x=df_zz.index.values, y=df_zz['FUZ_{}_S-1'.format(var)], 
                      fill='none', mode='lines', name='-1std', line=dict(width=4, color=colors[1]))
    trace_ma = go.Scatter(x=df_zz.index.values, y=df_zz['FUZ_{}_S0'.format(var)], 
                      fill='none', mode='lines', name=' 0std', line=dict(width=4, color=colors[0]))
    trace_up1 = go.Scatter(x=df_zz.index.values, y=df_zz['FUZ_{}_S+1'.format(var)], 
                      fill='none', mode='lines', name='+1std', line=dict(width=4, color=colors[1]))
    trace_up2 = go.Scatter(x=df_zz.index.values, y=df_zz['FUZ_{}_S+2'.format(var)], 
                      fill='none', mode='lines', name='+2std', line=dict(width=4, color=colors[2]))
    trace_crisp = go.Scatter(x=df_zz.index.values, y=df_zz['FUZ_{}'.format(var)], name='fuz_{}'.format(var), line=scatter.Line(color='black', width=1))
    return [trace_up2, trace_up1, trace_ma, trace_lo1, trace_lo2, trace_crisp]

  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def plotFuzzyVariable(self, var, colors=['rgb(195, 243, 195)','rgb(245, 230, 180)','rgb(252, 180, 197)','rgb(150, 27, 200)']):
    """ Plot stacked areas as fuzzy sets evolution agains range variable
    """
    traces = []
    if 'FUZ_{}_S-3'.format(var) in self.__df.columns:
      traces.append(go.Scatter( x=self.__df.index.values, y=self.__df['FUZ_{}_S-3'.format(var)], 
                      fill='none', mode='lines', name='-3std', line=dict(width=4, color=colors[3])))
    if 'FUZ_{}_S-2'.format(var) in self.__df.columns:
      traces.append(go.Scatter(x=self.__df.index.values, y=self.__df['FUZ_{}_S-2'.format(var)], 
                      fill='none', mode='lines', name='-2std', line=dict(width=4, color=colors[2])))
    if 'FUZ_{}_S-1'.format(var) in self.__df.columns:
      traces.append(go.Scatter(x=self.__df.index.values, y=self.__df['FUZ_{}_S-1'.format(var)], 
                      fill='none', mode='lines', name='-1std', line=dict(width=4, color=colors[1])))
    if 'FUZ_{}_S0'.format(var) in self.__df.columns:
      traces.append(go.Scatter(x=self.__df.index.values, y=self.__df['FUZ_{}_S0'.format(var)], 
                      fill='none', mode='lines', name=' 0std', line=dict(width=4, color=colors[0])))
    if 'FUZ_{}_S+1'.format(var) in self.__df.columns:
      traces.append(go.Scatter(x=self.__df.index.values, y=self.__df['FUZ_{}_S+1'.format(var)], 
                      fill='none', mode='lines', name='+1std', line=dict(width=4, color=colors[1])))
    if 'FUZ_{}_S+2'.format(var) in self.__df.columns:
      traces.append(go.Scatter(x=self.__df.index.values, y=self.__df['FUZ_{}_S+2'.format(var)], 
                      fill='none', mode='lines', name='+2std', line=dict(width=4, color=colors[2])))
    if 'FUZ_{}_S+3'.format(var) in self.__df.columns:
      traces.append(go.Scatter( x=self.__df.index.values, y=self.__df['FUZ_{}_S+3'.format(var)], 
                      fill='none', mode='lines', name='+3std', line=dict(width=4, color=colors[3])))
    traces.append(go.Scatter(x=self.__df.index.values, y=self.__df['FUZ_{}'.format(var)], 
                  name='fuz_{}'.format(var), line=scatter.Line(color='black', width=1)))
    return traces

  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def listFuzzyVariables(self):
    """ List all fuzzy variables in self.__df 
      Return: list of fuzzy variables
    """
    total_fuzzy_vars = []
    for c in self.__df.columns:
      for g in range(0,7):
        if 'FUZ_' and '_G{}'.format(g) in c:
          total_fuzzy_vars.append(c)
          self.__logger.debug('Added=> {}'.format(c))
    self.__logger.debug('Total num of fuzz_vars={}'.format(len(total_fuzzy_vars)))                   
    return total_fuzzy_vars