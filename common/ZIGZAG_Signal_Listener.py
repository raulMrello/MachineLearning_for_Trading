#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################################################
# Librerías de manejo de datos 
import pandas as pd
import numpy as np

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

####################################################################################
# Otras utilidades
import datetime
import time
import os
import sys
import math
import pickle
from enum import Enum
import logging


####################################################################################
####################################################################################
####################################################################################


class ZIGZAG_Events():
  def __init__(self):
    self.clear()

  def clear(self):
    self.ZIGZAG_StartMinSearch = False
    self.ZIGZAG_StartMaxSearch = False

  def any(self):
    if self.ZIGZAG_StartMinSearch or self.ZIGZAG_StartMaxSearch:
      return True
    return False

  def info(self):
    result =''
    if self.ZIGZAG_StartMinSearch:
      result += 'ZigZagStartMinSearch '
    if self.ZIGZAG_StartMaxSearch:
      result += 'ZigZagStartMaxSearch '
    return result


####################################################################################
####################################################################################
####################################################################################

class ZIGZAG_Signal_Listener():
  def __init__(self, level=logging.WARN):    
    self.__logger = logging.getLogger(__name__)
    self.__logger.setLevel(level)
    self.__df = None
    self.__events = ZIGZAG_Events()
    self.__logger.info('Created!')

  def ZIGZAG(self, df, minbars=12, bb_period=20, bb_dev = 2.0, nan_value = 0.0, dropna=True, level=logging.WARN):    
    """Builds a ZIGZAG indicator based on Bollinger Bands overbought and oversell signals

    Keyword arguments:
      df -- Datafeed to apply indicator
      minbars -- Min. number of bars per flip to avoid discarding (default 12)
      bb_period -- Bollinger bands period (default 20)
      bb_dev -- Bollinger bands deviation (default 2.0)
      nan_value -- Values for zigzag indicator during search phase (default 0.0)
      dropna -- Flag to delete NaN values, else fill them with nan_value
      level -- logging level (default WARN)
    """
    class ActionCtrl():
      class ActionType(Enum):
        NoActions = 0
        SearchingHigh = 1
        SearchingLow = 2
      def __init__(self, high, low, idx, delta, level=logging.WARN):
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(level)
        self.curr = ActionCtrl.ActionType.NoActions
        self.last_high = high
        self.last_high_idx = idx
        self.beforelast_high = high
        self.beforelast_high_idx = idx
        self.last_swing_high_idx = idx
        self.last_low = low
        self.last_low_idx = idx
        self.beforelast_low = low
        self.beforelast_low_idx = idx        
        self.last_swing_low_idx = idx
        self.delta = delta
        self.events = ZIGZAG_Events()
        self.__logger.debug('New action at idx={}: last_high={}, last_low={}, min-delta={}'.format(idx, self.last_high, self.last_low, self.delta))
      #---end action::__init__

      def __result(self):
        if self.curr == ActionCtrl.ActionType.SearchingHigh:
          return 'high'
        elif self.curr == ActionCtrl.ActionType.SearchingLow:
          return 'low'
        return 'no-action'
      #---end action::__result

      # this function updates MAX|MIN values with last recorded depending on the current action
      def zigzag(self, x, df):
        log = 'Procesing [{}]:'.format(x.name)
        self.events.clear()

        # check if HIGH must be updated
        max_value = x.HIGH 
        if self.curr == ActionCtrl.ActionType.SearchingHigh and max_value > self.last_high:
          #self.beforelast_high = self.last_high
          #self.beforelast_high_idx = self.last_high_idx          
          self.last_high = max_value
          self.last_high_idx = x.name
          log += ' new HIGH={}'.format(max_value)   
          self.__logger.debug(log)
          return self.__result()

        # check if LOW must be updated
        min_value = x.LOW 
        if self.curr == ActionCtrl.ActionType.SearchingLow and min_value < self.last_low:
          #self.beforelast_low = self.last_low
          #self.beforelast_low_idx = self.last_low_idx
          self.last_low = min_value
          self.last_low_idx = x.name
          log += ' new LOW={}'.format(min_value)
          self.__logger.debug(log)
          return self.__result()

        # check if search HIGH starts
        if self.curr != ActionCtrl.ActionType.SearchingHigh and max_value > x.BOLLINGER_HI:
          _prev_action = self.curr
          self.events.ZIGZAG_StartMaxSearch = True
          self.curr = ActionCtrl.ActionType.SearchingHigh
          # check delta condition
          curr_delta = (x.name - self.last_high_idx)
          if _prev_action == ActionCtrl.ActionType.NoActions:
            # in first swing doesnt apply
            curr_delta = self.delta + 1
          if curr_delta < self.delta:
            log += ' ERR_DELTA \/ ={}' .format(curr_delta)
            df.at[self.last_high_idx,'ZIGZAG'] =  nan_value
            df.at[self.last_high_idx,'ACTION'] =  'high'
            df.loc[self.last_high_idx:x.name, 'FLIP'] = False
            if max_value > self.last_high:
              log += ' replace_HIGH @[{}]=>{}'.format(self.last_high_idx,max_value)               
              self.last_high = max_value
              self.last_high_idx = x.name
            else:
              log += ' keep_HIGH @[{}]=>{}'.format(self.last_high_idx,self.last_high)
            df.at[self.last_low_idx,'ZIGZAG'] =  nan_value
            df.at[self.last_low_idx,'ACTION'] =  'high' 
            log += ' remove LOW @[{}]'.format(self.last_low_idx)
            self.last_low = self.beforelast_low
            self.last_low_idx = self.beforelast_low_idx
            self.__logger.info(log)  
          else:
            # save last low     
            df.at[self.last_low_idx,'ZIGZAG'] =  self.last_low   
            self.beforelast_low = self.last_low
            self.beforelast_low_idx = self.last_low_idx            
            # starts high recording
            #self.beforelast_high = self.last_high
            #self.beforelast_high_idx = self.last_high_idx
            self.last_high = max_value
            self.last_high_idx = x.name
            log += ' save LOW @[{}]={}, (FLIP) new HIGH=>{}'.format(self.last_low_idx, self.last_low, max_value)    
            df.at[x.name,'FLIP'] =  True
            self.__logger.debug(log)
          return self.__result()

        # check if search LOW starts
        if self.curr != ActionCtrl.ActionType.SearchingLow and min_value < x.BOLLINGER_LO:
          _prev_action = self.curr
          self.events.ZIGZAG_StartMinSearch = True
          self.curr = ActionCtrl.ActionType.SearchingLow
          # check delta condition
          curr_delta = (x.name - self.last_low_idx)
          if _prev_action == ActionCtrl.ActionType.NoActions:
            # in first swing doesnt apply
            curr_delta = self.delta + 1          
          if curr_delta < self.delta:
            log += ' ERR_DELTA /\ ={}' .format(curr_delta) 
            df.at[self.last_low_idx,'ZIGZAG'] =  nan_value
            df.at[self.last_low_idx,'ACTION'] =  'low'
            df.loc[self.last_high_idx:x.name, 'FLIP'] = False
            if min_value < self.last_low:
              log += ' replace_LOW @[{}]=>{}'.format(self.last_low_idx,min_value)              
              self.last_low = min_value
              self.last_low_idx = x.name  
            else:
              log += ' keep_LOW @[{}]=>{}'.format(self.last_low_idx,self.last_low)
            df.at[self.last_high_idx,'ZIGZAG'] =  nan_value
            df.at[self.last_high_idx,'ACTION'] =  'low' 
            log += ' remove HIGH @[{}]'.format(self.last_high_idx)
            self.last_high = self.beforelast_high
            self.last_high_idx = self.beforelast_high_idx
            self.__logger.info(log)  
          else:
            # save last high
            df.at[self.last_high_idx,'ZIGZAG'] =  self.last_high
            self.beforelast_high = self.last_high
            self.beforelast_high_idx = self.last_high_idx            
            # starts low recording
            #self.beforelast_low = self.last_low
            #self.beforelast_low_idx = self.last_low_idx
            self.last_low = min_value
            self.last_low_idx = x.name
            log += ' save HIGH @[{}]={}, (FLIP) new LOW=>{}'.format(self.last_high_idx, self.last_high, min_value)        
            df.at[x.name,'FLIP'] =  True
            self.__logger.debug(log)
          return self.__result()

        if self.curr == ActionCtrl.ActionType.SearchingLow:
          log += ' curr LOW @[{}]=>{}'.format(self.last_low_idx,self.last_low)
        elif self.curr == ActionCtrl.ActionType.SearchingHigh:
          log += ' curr HIGH @[{}]=>{}'.format(self.last_high_idx,self.last_high)
        self.__logger.debug(log)
        return self.__result()    
      #---end action::zigzag

      # registers previous zzpoints and their indexes at current row.Px and row.Px_idx
      def points(self, row, df, nan_value):
        log = 'row [{}] zigzag={}: '.format(row.name, row.ZIGZAG)

        # get last 5 zigzag points
        zzpoints = df.ZIGZAG[(df.index < row.name) & (df.ZIGZAG != nan_value) & (df.ACTION.str.contains('in-progress')==False)]
        # at least requires 6 points, else discard
        if zzpoints.shape[0] < 6: 
          log += 'exit-zzpoints-count={} '.format(zzpoints.shape[0])
          self.__logger.debug(log)
          return
        # update rows
        for i in range(1,7):
          df.at[row.name,'P{}'.format(i)] = zzpoints.iloc[-i]          
          df.at[row.name,'P{}_idx'.format(i)] = zzpoints.index[-i]
        log += 'Updated!! '
        self.__logger.debug(log)
      #---end action::points

    # clear events
    self.__events.clear()

    # copy dataframe and calculate bollinger bands if not yet present
    _df = df.copy()
    _df['BOLLINGER_HI'], _df['BOLLINGER_MA'], _df['BOLLINGER_LO'] = talib.BBANDS(_df.CLOSE, timeperiod=bb_period, nbdevup=bb_dev, nbdevdn=bb_dev, matype=0)
    _df['BOLLINGER_WIDTH'] = _df['BOLLINGER_HI'] - _df['BOLLINGER_LO']
    boll_b = (_df.CLOSE - _df['BOLLINGER_LO'])/_df['BOLLINGER_WIDTH']
    boll_b[np.isnan(boll_b)]=0.5
    boll_b[np.isinf(boll_b)]=0.5
    _df['BOLLINGER_b'] = boll_b
    if dropna:
      _df.dropna(inplace=True)
      _df.reset_index(drop=True, inplace=True)
    else:
      _df.fillna(value=nan_value, inplace=True)

    # Initially no actions are in progress, record first high and low values creating an ActionCtrl object
    action = ActionCtrl(
              high= _df['HIGH'][0], #max(_df['OPEN'][0], _df['CLOSE'][0]), 
              low = _df['LOW'][0], #min(_df['OPEN'][0], _df['CLOSE'][0]), 
              idx = _df.iloc[0].name, 
              delta= minbars,
              level=level)

    _df['ZIGZAG'] = nan_value
    _df['ACTION'] = 'no-action'
    _df['FLIP'] = False
    _df['ACTION'] = _df.apply(lambda x: action.zigzag(x, _df), axis=1)

    # fills last element as pending
    if _df.ZIGZAG.iloc[-1] == nan_value:
      if _df.ACTION.iloc[-1] == 'high':
        _df.at[action.last_high_idx,'ZIGZAG'] =  action.last_high
        _df.at[action.last_high_idx,'ACTION'] =  'high-in-progress'  
      else:
        _df.at[action.last_low_idx,'ZIGZAG'] =  action.last_low
        _df.at[action.last_low_idx,'ACTION'] =  'low-in-progress'  
    
    # now adds point p1 to p6 backtrace of values and indexes:
    for i in range(1,7):
      _df['P{}'.format(i)] = nan_value
      _df['P{}_idx'.format(i)] = 0

    _df.apply(lambda x: action.points(x, _df, nan_value), axis=1)
    self.__df = _df
    self.__action = action
    return self.__df, self.__action.events

  def getDataFrame(self):
    return self.__df