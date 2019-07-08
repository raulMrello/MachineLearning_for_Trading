#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

####################################################################################
# Data handling 
import pandas as pd
from pandas import concat
from pandas.plotting import scatter_matrix
import numpy as np

####################################################################################
# Visualization
import matplotlib.pyplot as plt
from matplotlib import dates, ticker
from matplotlib.dates import (MONDAY, DateFormatter, MonthLocator, WeekdayLocator, date2num)
import matplotlib as mpl
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
from plotly.tools import FigureFactory as FF
import plotly.tools as tls

####################################################################################
# TA-Lib: 
import talib

####################################################################################
# Other utilities
import datetime
import time
import os
import sys
import math
from enum import Enum

####################################################################################
####################################################################################
####################################################################################

class MACD_Events():
  def __init__(self):
    self.clear()

  def clear(self):
    self.MACD_Bullish_Divergence = False
    self.MACD_Bearish_Divergence = False
    self.MACD_Bullish_Main_Zero_Crossover = False
    self.MACD_Bearish_Main_Zero_Crossover = False
    self.MACD_Bullish_Main_Signal_Crossover = False
    self.MACD_Bearish_Main_Signal_Crossover = False

  def any(self):
    if self.MACD_Bullish_Divergence or self.MACD_Bearish_Divergence or self.MACD_Bullish_Main_Zero_Crossover or self.MACD_Bearish_Main_Zero_Crossover or self.MACD_Bullish_Main_Signal_Crossover or self.MACD_Bearish_Main_Signal_Crossover:
      return True
    return False

  def info(self):
    result =''
    if self.MACD_Bullish_Divergence:
      result += 'BullishDivergence '
    if self.MACD_Bearish_Divergence:
      result += 'BearishDivergence '
    if self.MACD_Bullish_Main_Zero_Crossover:
      result += 'BullishMainZeroCrossover '
    if self.MACD_Bearish_Main_Zero_Crossover:
      result += 'BearishMainZeroCrossover '
    if self.MACD_Bullish_Main_Signal_Crossover:
      result += 'BullishMainSignalCrossover '
    if self.MACD_Bearish_Main_Signal_Crossover:
      result += 'BearishMainSignalCrossover '
    return result


####################################################################################
####################################################################################
####################################################################################

class MACD_Signal_Listener():
  #-------------------------------------------------
  def __init__( self, 
                param_num_bars_per_swing=5,
                param_num_bars_per_minmax_wdow=2):
    """Build a MACD Signal Listener object

    Keyword arguments:
      param_num_bars_per_swing -- Minimum number of bars in a swing (from crossover with zero to crossover with signal). Those swings
        with smaller number of bars will be discarded.
      param_num_bars_per_minmax_wdow -- Number of bars around a min|max value in MACD main, to look for a min|max value in OHLC prices,
        in order to verify if a divergence is present or not.
    """
    self.__CONFIG_MIN_BARS_ON_SWING = param_num_bars_per_swing
    self.__CONFIG_BARS = param_num_bars_per_minmax_wdow
    self.__df = None
    self.__events = MACD_Events()

  #-------------------------------------------------
  def MACD( self, 
            df, 
            applied = 'CLOSE', 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9,
            common_event_listeners=[],
            bullish_divergence_listeners=[],
            bearish_divergence_listeners=[],
            bullish_main_zero_cross_listeners=[],
            bearish_main_zero_cross_listeners=[],
            bullish_main_signal_cross_listeners=[],
            bearish_main_signal_cross_listeners=[]):
    """Builds MACD indicator with default configuration on df dataframe

      Keyword arguments:
      df -- DataFeed (requires column names: OPEN,HIGH,LOW,CLOSE)
      applied -- Column name to apply MACD indicator
      fastperiod -- Fast EMA (default 12)
      slowperiod -- Slow EMA (default 26)
      signalperiod -- Signal EMA (default 9)
      common_event_listeners -- List of listeners to process any kind of event or their combination
      bullish_divergence_listeners -- List of listeners to be notified on signal detection
      bearish_divergence_listeners -- List of listeners to be notified on signal detection 
      bullish_main_zero_cross_listeners -- List of listeners to be notified on signal detection
      bearish_main_zero_cross_listeners -- List of listeners to be notified on signal detection
      bullish_main_signal_cross_listeners -- List of listeners to be notified on signal detection
      bearish_main_signal_cross_listeners -- List of listeners to be notified on signal detection
    """
    if not 'OPEN' or not 'HIGH' or not 'LOW' or not 'CLOSE' in df.columns:
      printf('ERROR: some column is missing. Required columns are: OPEN,HIGH,LOW,CLOSE')
      return 0,None

    if not applied in df.columns:
      print('Column {} is missing in df'.format(applied))
      return 0,None

    # copy df to apply changes
    self.__df = df.copy()      
    self.__df.reset_index(drop=True, inplace=True)

    # add MACD main, signal and histogram
    self.__df['MACD'], self.__df['MACDS'], self.__df['MACDH'] = talib.MACD(self.__df[applied], fastperiod, slowperiod, signalperiod)

    # remove NaNs on moving averages calculation
    self.__df.dropna(inplace=True)
    self.__df.reset_index(drop=True, inplace=True)

    # add crossovers between main and zero level
    self.__df['CROSS_ZERO_UP'] = ((self.__df.MACD > 0) & (self.__df.MACD.shift(1) < 0 | ((self.__df.MACD.shift(1)==0) & (self.__df.MACD.shift(2) < 0))))
    self.__df['CROSS_ZERO_DN'] = ((self.__df.MACD < 0) & (self.__df.MACD.shift(1) > 0 | ((self.__df.MACD.shift(1)==0) & (self.__df.MACD.shift(2) > 0))))

    # add crossovers between main and signal lines
    self.__df['CROSS_SIG_UP'] = (((self.__df.MACD > self.__df.MACDS) & ((self.__df.MACD.shift(1) < self.__df.MACDS.shift(1)) | ((self.__df.MACD.shift(1)==self.__df.MACDS.shift(1)) & (self.__df.MACD.shift(2) < self.__df.MACDS.shift(2))))) & ((self.__df.MACD < 0) & (self.__df.MACD.shift(1) < 0) & (self.__df.MACD.shift(2) < 0)))
    self.__df['CROSS_SIG_DN'] = (((self.__df.MACD < self.__df.MACDS) & ((self.__df.MACD.shift(1) > self.__df.MACDS.shift(1)) | ((self.__df.MACD.shift(1)==self.__df.MACDS.shift(1)) & (self.__df.MACD.shift(2) > self.__df.MACDS.shift(2))))) & ((self.__df.MACD > 0) & (self.__df.MACD.shift(1) > 0) & (self.__df.MACD.shift(2) > 0)))

    # bars with upwards crossover
    swing_start = self.__df['CROSS_ZERO_UP'] 

    # bars with downwards crossover above zero
    swing_stop = self.__df['CROSS_ZERO_DN'] 

    # indexes where crossovers occurs
    swing_start_idx = swing_start[swing_start == True]
    swing_stop_idx = swing_stop[swing_stop == True]

    # ensures first valid swing
    swing_stop_idx  = swing_stop_idx[swing_stop_idx.index > swing_start_idx.index[0]]

    # now, builds column with macd_max points  
    self.__df['MACD_MAX'] = False

    # for each downwards crossover, searches previous upwards crossover
    discarded_bearish_swings = 0
    for dn_cross in swing_stop_idx.index:
      prev_up_cross = swing_start_idx[swing_start_idx.index < dn_cross].index[-1]
      if dn_cross-prev_up_cross < self.__CONFIG_MIN_BARS_ON_SWING:
        discarded_bearish_swings += 1
      else:
        idx_max_val = self.__df.MACD[prev_up_cross:dn_cross].idxmax()
        self.__df['MACD_MAX'].at[idx_max_val] = True  

    # bars with upwards crossover
    bull_swing_start = self.__df['CROSS_ZERO_DN'] 

    # bars with upwards crossover below zero
    bull_swing_stop = self.__df['CROSS_ZERO_UP'] 

    # indexes where crossovers occurs
    bull_swing_start_idx = bull_swing_start[bull_swing_start == True]
    bull_swing_stop_idx = bull_swing_stop[bull_swing_stop == True]

    # ensures first valid swing
    bull_swing_stop_idx  = bull_swing_stop_idx[bull_swing_stop_idx.index > bull_swing_start_idx.index[0]]

    # now, builds column with macd_min points  
    self.__df['MACD_MIN'] = False

    # for each upwards crossover, searches previous downwards crossover
    discarded_bullish_swings = 0
    for up_cross in bull_swing_stop_idx.index:
      prev_dn_cross = bull_swing_start_idx[bull_swing_start_idx.index < up_cross].index[-1]
      if up_cross-prev_dn_cross < self.__CONFIG_MIN_BARS_ON_SWING:
        discarded_bullish_swings += 1
      else:
        idx_min_val = self.__df.MACD[prev_dn_cross:up_cross].idxmin()
        self.__df['MACD_MIN'].at[idx_min_val] = True  

    # Now, select decreasing-max pairs in MACD main line
    macd_max = self.__df.MACD[self.__df['MACD_MAX'] == True]
    self.__df['DECR_MAX'] = False
    prev_ix = macd_max.index[0]
    bear_div_candidates = []
    for x in macd_max[1:].index:
      if macd_max[x] < macd_max[prev_ix]:
        self.__df.DECR_MAX.at[x] = True
        bear_div_candidates.append((prev_ix, x))
      prev_ix = x

    # Now, select increasing-min pairs in MACD main line
    macd_min = self.__df.MACD[self.__df['MACD_MIN'] == True]
    self.__df['DECR_MIN'] = False
    bull_prev_ix = macd_min.index[0]
    bull_div_candidates = []
    for x in macd_min[1:].index:
      if macd_min[x] > macd_min[bull_prev_ix]:
        self.__df.DECR_MIN.at[x] = True
        bull_div_candidates.append((bull_prev_ix, x))
      bull_prev_ix = x

    # setup a window of +- N bars around both MACD max values, check if a max value exists and if their are increasing
    self.__df['BEAR_DIVERGENCE'] = False
    bear_div_zones = []
    for c in bear_div_candidates:  
      idx_1st_max = self.__df.HIGH[c[0]-self.__CONFIG_BARS:c[0]+self.__CONFIG_BARS].idxmax()
      max_1 = self.__df.HIGH[c[0]-self.__CONFIG_BARS:c[0]+self.__CONFIG_BARS].max()
      idx_2nd_max = self.__df.HIGH[c[1]-self.__CONFIG_BARS:c[1]+self.__CONFIG_BARS].idxmax()  
      max_2 = self.__df.HIGH[c[1]-self.__CONFIG_BARS:c[1]+self.__CONFIG_BARS].max() 
      if max_1 < max_2 and idx_1st_max in range(c[0]-self.__CONFIG_BARS+1, c[0]+self.__CONFIG_BARS) and idx_2nd_max in range(c[1]-self.__CONFIG_BARS+1, c[1]+self.__CONFIG_BARS):
        bear_div_zones.append([idx_1st_max, idx_2nd_max])
        self.__df.loc[(self.__df.index >= idx_1st_max) & (self.__df.index <= idx_2nd_max), 'BEAR_DIVERGENCE'] = True

    # setup a window of +- N bars around both MACD min values, check if a min value exists and if their are decreasing
    self.__df['BULL_DIVERGENCE'] = False
    bull_div_zones = []
    for c in bull_div_candidates:  
      idx_1st_min = self.__df.LOW[c[0]-self.__CONFIG_BARS:c[0]+self.__CONFIG_BARS].idxmin()
      min_1 = self.__df.LOW[c[0]-self.__CONFIG_BARS:c[0]+self.__CONFIG_BARS].min()
      idx_2nd_min = self.__df.LOW[c[1]-self.__CONFIG_BARS:c[1]+self.__CONFIG_BARS].idxmin()  
      min_2 = self.__df.LOW[c[1]-self.__CONFIG_BARS:c[1]+self.__CONFIG_BARS].min() 
      if min_1 > min_2 and idx_1st_min in range(c[0]-self.__CONFIG_BARS+1, c[0]+self.__CONFIG_BARS) and idx_2nd_min in range(c[1]-self.__CONFIG_BARS+1, c[1]+self.__CONFIG_BARS):
        bull_div_zones.append([idx_1st_min, idx_2nd_min])
        self.__df.loc[(self.__df.index >= idx_1st_min) & (self.__df.index <= idx_2nd_min), 'BULL_DIVERGENCE'] = True

    # copy calculated divergences
    self.__bear_div_zones = bear_div_zones
    self.__bull_div_zones = bull_div_zones

    # signal notification to listeners
    # clear events
    self.__events.clear()
    # notify bullish divergences if required
    if self.__df.BULL_DIVERGENCE.iloc[-1] == True:
      self.__events.MACD_Bullish_Divergence = True
      for listener in bullish_divergence_listeners:
        listener(self.__df)
    # notify bearish divergences if required
    if self.__df.BEAR_DIVERGENCE.iloc[-1] == True:
      self.__events.MACD_Bearish_Divergence = True
      for listener in bearish_divergence_listeners:
        listener(self.__df)
    # notify bullish main-zero-cross if required
    if self.__df.CROSS_ZERO_UP.iloc[-1] == True:
      self.__events.MACD_Bullish_Main_Zero_Crossover = True
      for listener in bullish_main_zero_cross_listeners:
        listener(self.__df)
    # notify bearish main-zero-cross if required
    if self.__df.CROSS_ZERO_DN.iloc[-1] == True:
      self.__events.MACD_Bearish_Main_Zero_Crossover = True
      for listener in bearish_main_zero_cross_listeners:
        listener(self.__df)
    # notify bullish main-sig-cross if required
    if self.__df.CROSS_SIG_UP.iloc[-1] == True:
      self.__events.MACD_Bullish_Main_Signal_Crossover = True
      for listener in bullish_main_signal_cross_listeners:
        listener(self.__df)
    # notify bearish main-sig-cross if required
    if self.__df.CROSS_SIG_DN.iloc[-1] == True:
      self.__events.MACD_Bearish_Main_Signal_Crossover = True
      for listener in bearish_main_signal_cross_listeners:
        listener(self.__df)
    # notify common signals if required
    if self.__events.any():
      for listener in common_event_listeners:
        listener(self.__events, self.__df)
    # finish returning a reference of the calculated dataframe    
    return self.__events, self.__df


  #-------------------------------------------------
  def getDataFeed(self):
    """Get internal data feed reference
    
    Returns:
    self.__df -- Internal dataframe reference
    """
    return self.__df


  #-------------------------------------------------
  def getBullishDivergences(self):
    """Get last calculated bullish divergences
    
    Returns:
    self.__bull_div_zones -- Internal bullish divergence list
    """
    return self.__bull_div_zones


  #-------------------------------------------------
  def getBearishDivergences(self):
    """Get last calculated bearish divergences
    
    Returns:
    self.__bear_div_zones -- Internal bearish divergence list
    """
    return self.__bear_div_zones




