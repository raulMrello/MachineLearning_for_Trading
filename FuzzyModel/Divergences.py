#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################################################
# Librerías de manejo de datos 
import pandas as pd
import numpy as np

####################################################################################
# Librerías de visualización
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
from plotly.tools import FigureFactory as FF
import plotly.tools as tls

####################################################################################
# TA-Lib: instalación y carga de la librería
import talib
from ZIGZAG_Signal_Listener import ZIGZAG_Signal_Listener

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

class Divergences():
  def __init__(self, level=logging.WARN):    
    self.__logger = logging.getLogger(__name__)
    self.__logger.setLevel(level)
    self.__df = None
    self.__logger.info('Created!')
    self.__zigzag = ZIGZAG_Signal_Listener(level)

  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def searchDivergences(self, df, zigzag_cfg = dict(), level=logging.WARN, exitAt='end'):    
  #-------------------------------------------------------------------
    """Builds from dataframe df, next indicators: MACD, RSI, Stochastic with default
       parameters. Then builds a zigzag indicator.

       Returns resulting dataframe and events raised at last bar.

    Keyword arguments:
      df -- Datafeed to apply indicator
      minbars -- Min. number of bars per flip to avoid discarding (default 12)
      zigzag_cfg -- Dictionary with zigzag configuration (default None)
      level -- logging level (default WARN)
      exitAt -- Label for exit at debugging phase
    """
    
    self.__logger.setLevel(level)

    #builds zigzag
    _minbars   = zigzag_cfg['minbars'] if 'minbars' in zigzag_cfg.keys() else 12  
    _bb_period = zigzag_cfg['bb_period'] if 'bb_period' in zigzag_cfg.keys() else 2 
    _bb_dev    = zigzag_cfg['bb_dev'] if 'bb_dev' in zigzag_cfg.keys() else 2.0
    _bb_sma    = zigzag_cfg['bb_sma'] if 'bb_sma' in zigzag_cfg.keys() else [100]
    _nan_value = zigzag_cfg['nan_value'] if 'nan_value' in zigzag_cfg.keys() else 0.0 
    _zlevel    = zigzag_cfg['level'] if 'level ' in zigzag_cfg.keys() else logging.WARN
    _df, _evt_zigzag =  self.__zigzag.ZIGZAG( df, 
                                              minbars   = _minbars,
                                              bb_period = _bb_period,
                                              bb_dev    = _bb_dev,
                                              bb_sma    = _bb_sma,
                                              nan_value = _nan_value,
                                              level     = _zlevel)

    if exitAt == 'zigzag-calculation':
      return _df.copy()

    # build MACD_main, RSI and Stochastic
    _df['MACD_main'], _df['MACD_signal'], _df['MACD_hist'] = talib.MACD(_df.CLOSE, fastperiod=12, slowperiod=26, signalperiod=9)
    _df['RSI'] = talib.RSI(_df.CLOSE, timeperiod=14)
    _df['STOCH_K'], _df['STOCH_d'] = talib.STOCH(_df.HIGH, _df.LOW, _df.CLOSE, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    # remove Nans and reindex from sample 0
    _df.dropna(inplace=True)
    _df.reset_index(drop=True, inplace=True)

    # add result columns
    _df['DIVERGENCE_MACD'] = 'none'
    _df['DIVERGENCE_MACD_FROM'] = 0
    _df['DIVERGENCE_RSI'] = 'none'
    _df['DIVERGENCE_RSI_FROM'] = 0
    _df['DIVERGENCE_STOCH'] = 'none'
    _df['DIVERGENCE_STOCH_FROM'] = 0
    
    if exitAt == 'oscillator-built':
      return _df.copy()

    # executes divergence localization process:
    # 1. Set a default trend: requires 3 max and 3 min points
    # 1a.If max increasing or min increasing -> Bullish trend
    # 1b.If max decreasing or min decreasing -> Bearish trend
    # 1c.Else discard.

    def search(row, df, nan_value, logger, exitAt):
      log = 'row [{}]: '.format(row.name)

      # skip rows where no zzpoints
      if row.ZIGZAG == nan_value: 
        log += 'error-zigzag-isnan'
        logger.debug(log)
        return

      # get last 6 zigzag points
      zzpoints = df.ZIGZAG[(df.index <= row.name) & (df.ZIGZAG != nan_value)]
      # at least requires 5 points, else discard
      if zzpoints.shape[0] < 5: 
        log += 'error-zzpoints-count={} '.format(zzpoints.shape[0])
        logger.debug(log)
        return

      # check if curr sample is max, min or the same as previous
      curr_is = 'unknown'
      # check curr sample is max 
      if zzpoints.iloc[-1] > zzpoints.iloc[-2]:
        log += 'last is MAX '
        curr_is = 'max'

      #check if is min
      elif zzpoints.iloc[-1] < zzpoints.iloc[-2]:
        log += 'last is MIN '
        curr_is = 'min'   

      # last 2 samples are equal, then finish
      else:
        log += 'error-no-minmax '
        logger.debug(log)
        return
      
      # at this point, exists a condition to evaluate.
      # Get idx of last 6 points (3 zigzags)
      p0_idx = zzpoints.index[-1]
      p1_idx = zzpoints.index[-2]
      p2_idx = zzpoints.index[-3]
      p3_idx = zzpoints.index[-4]
      p4_idx = zzpoints.index[-5]
      p5_idx = zzpoints.index[-6]
      log += 'p0={}, p1={}, p2={}, p3={}, p4={}, p5={} '.format(p0_idx, p1_idx, p2_idx, p3_idx, p4_idx, p5_idx)

      # check if is bullish trend: is curr_is_MAX then 3max&2min if curr_is_MIN then 3min&2max
      if zzpoints.iloc[-1] > zzpoints.iloc[-3] and zzpoints.iloc[-3] > zzpoints.iloc[-5] and zzpoints.iloc[-2] > zzpoints.iloc[-4]:
        log += 'bullish-trend '            
        # if last is max check regular divergences
        if curr_is == 'max':
          # search regular-bearish-divergences on MACD
          log += 'check macd: [{}]{} < [{}]{}'.format(p0_idx, df.MACD_main.iloc[p0_idx], p2_idx, df.MACD_main.iloc[p2_idx])              
          if df.MACD_main.iloc[p0_idx] < df.MACD_main.iloc[p2_idx]:
            log += 'check macd: [{}]{} < [{}]{}'.format(p2_idx, df.MACD_main.iloc[p2_idx], p4_idx, df.MACD_main.iloc[p4_idx])
            if df.MACD_main.iloc[p2_idx] < df.MACD_main.iloc[p4_idx]:
              log += 'double-regular-bearish-divergence '
              df.at[p0_idx, 'DIVERGENCE_MACD'] = 'double-regular-bearish-divergence'
              df.at[p0_idx, 'DIVERGENCE_MACD_FROM'] = p4_idx
            else:
              log += 'regular-bearish-divergence '
              df.at[p0_idx, 'DIVERGENCE_MACD'] = 'regular-bearish-divergence'
              df.at[p0_idx, 'DIVERGENCE_MACD_FROM'] = p2_idx
          # search regular-bearish-divergences on RSI
          log += 'check rsi: [{}]{} < [{}]{}'.format(p0_idx, df.RSI.iloc[p0_idx], p2_idx, df.RSI.iloc[p2_idx])
          if df.RSI.iloc[p0_idx] < df.RSI.iloc[p2_idx]:
            log += 'check rsi: [{}]{} < [{}]{}'.format(p2_idx, df.RSI.iloc[p2_idx], p4_idx, df.RSI.iloc[p4_idx])
            if df.RSI.iloc[p2_idx] < df.RSI.iloc[p4_idx]:
              log += 'double-regular-bearish-divergence '
              df.at[p0_idx, 'DIVERGENCE_RSI'] = 'double-regular-bearish-divergence'
              df.at[p0_idx, 'DIVERGENCE_RSI_FROM'] = p4_idx
            else:
              log += 'regular-bearish-divergence '
              df.at[p0_idx, 'DIVERGENCE_RSI'] = 'regular-bearish-divergence'
              df.at[p0_idx, 'DIVERGENCE_RSI_FROM'] = p2_idx
        
        # last is min, so check hidden divergences
        else:
          # search hidden-bullish-divergences on MACD
          log += 'check macd: [{}]{} < [{}]{}'.format(p0_idx, df.MACD_main.iloc[p0_idx], p2_idx, df.MACD_main.iloc[p2_idx])              
          if df.MACD_main.iloc[p0_idx] < df.MACD_main.iloc[p2_idx]:
            log += 'check macd: [{}]{} < [{}]{}'.format(p2_idx, df.MACD_main.iloc[p2_idx], p4_idx, df.MACD_main.iloc[p4_idx])
            if df.MACD_main.iloc[p2_idx] < df.MACD_main.iloc[p4_idx]:
              log += 'double-hidden-bullish-divergence '
              df.at[p0_idx, 'DIVERGENCE_MACD'] = 'double-hidden-bullish-divergence'
              df.at[p0_idx, 'DIVERGENCE_MACD_FROM'] = p4_idx
            else:
              log += 'hidden-bullish-divergence '
              df.at[p0_idx, 'DIVERGENCE_MACD'] = 'hidden-bullish-divergence'
              df.at[p0_idx, 'DIVERGENCE_MACD_FROM'] = p2_idx
          # search regular-bullish-divergences on RSI
          log += 'check rsi: [{}]{} < [{}]{}'.format(p0_idx, df.RSI.iloc[p0_idx], p2_idx, df.RSI.iloc[p2_idx])
          if df.RSI.iloc[p0_idx] < df.RSI.iloc[p2_idx]:
            log += 'check rsi: [{}]{} < [{}]{}'.format(p2_idx, df.RSI.iloc[p2_idx], p4_idx, df.RSI.iloc[p4_idx])
            if df.RSI.iloc[p2_idx] < df.RSI.iloc[p4_idx]:
              log += 'double-hidden-bullish-divergence '
              df.at[p0_idx, 'DIVERGENCE_RSI'] = 'double-hidden-bullish-divergence'
              df.at[p0_idx, 'DIVERGENCE_RSI_FROM'] = p4_idx
            else:
              log += 'hidden-bullish-divergence '
              df.at[p0_idx, 'DIVERGENCE_RSI'] = 'hidden-bullish-divergence'
              df.at[p0_idx, 'DIVERGENCE_RSI_FROM'] = p2_idx
        logger.debug(log)
        return

      # check if is bearish trend
      if zzpoints.iloc[-1] < zzpoints.iloc[-3] and zzpoints.iloc[-3] < zzpoints.iloc[-5] and zzpoints.iloc[-2] < zzpoints.iloc[-4]:
        log += 'bearish-trend '
        # if last is max check hidden divergences
        if curr_is == 'max':
          # search hidden-bearish-divergences on MACD
          log += 'check macd: [{}]{} > [{}]{}'.format(p0_idx, df.MACD_main.iloc[p0_idx], p2_idx, df.MACD_main.iloc[p2_idx])              
          if df.MACD_main.iloc[p0_idx] > df.MACD_main.iloc[p2_idx]:
            log += 'check macd: [{}]{} > [{}]{}'.format(p2_idx, df.MACD_main.iloc[p2_idx], p4_idx, df.MACD_main.iloc[p4_idx])
            if df.MACD_main.iloc[p2_idx] > df.MACD_main.iloc[p4_idx]:
              log += 'double-hidden-bearish-divergence '
              df.at[p0_idx, 'DIVERGENCE_MACD'] = 'double-hidden-bearish-divergence'
              df.at[p0_idx, 'DIVERGENCE_MACD_FROM'] = p4_idx
            else:
              log += 'hidden-bearish-divergence '
              df.at[p0_idx, 'DIVERGENCE_MACD'] = 'hidden-bearish-divergence'
              df.at[p0_idx, 'DIVERGENCE_MACD_FROM'] = p2_idx
          # search hidden-bearish-divergences on RSI
          log += 'check rsi: [{}]{} > [{}]{}'.format(p0_idx, df.RSI.iloc[p0_idx], p2_idx, df.RSI.iloc[p2_idx])
          if df.RSI.iloc[p0_idx] > df.RSI.iloc[p2_idx]:
            log += 'check rsi: [{}]{} > [{}]{}'.format(p2_idx, df.RSI.iloc[p2_idx], p4_idx, df.RSI.iloc[p4_idx])
            if df.RSI.iloc[p2_idx] > df.RSI.iloc[p4_idx]:
              log += 'double-hidden-bearish-divergence '
              df.at[p0_idx, 'DIVERGENCE_RSI'] = 'double-hidden-bearish-divergence'
              df.at[p0_idx, 'DIVERGENCE_RSI_FROM'] = p4_idx
            else:
              log += 'hidden-bearish-divergence '
              df.at[p0_idx, 'DIVERGENCE_RSI'] = 'hidden-bearish-divergence'
              df.at[p0_idx, 'DIVERGENCE_RSI_FROM'] = p2_idx
        
        # last is min, so check regular divergences
        else:
          # search regular-bullish-divergences on MACD
          log += 'check macd: [{}]{} > [{}]{}'.format(p0_idx, df.MACD_main.iloc[p0_idx], p2_idx, df.MACD_main.iloc[p2_idx])              
          if df.MACD_main.iloc[p0_idx] > df.MACD_main.iloc[p2_idx]:
            log += 'check macd: [{}]{} > [{}]{}'.format(p2_idx, df.MACD_main.iloc[p2_idx], p4_idx, df.MACD_main.iloc[p4_idx])
            if df.MACD_main.iloc[p2_idx] > df.MACD_main.iloc[p4_idx]:
              log += 'double-regular-bullish-divergence '
              df.at[p0_idx, 'DIVERGENCE_MACD'] = 'double-regular-bullish-divergence'
              df.at[p0_idx, 'DIVERGENCE_MACD_FROM'] = p4_idx
            else:
              log += 'regular-bullish-divergence '
              df.at[p0_idx, 'DIVERGENCE_MACD'] = 'regular-bullish-divergence'
              df.at[p0_idx, 'DIVERGENCE_MACD_FROM'] = p2_idx
          # search regular-bullish-divergences on RSI
          log += 'check rsi: [{}]{} > [{}]{}'.format(p0_idx, df.RSI.iloc[p0_idx], p2_idx, df.RSI.iloc[p2_idx])
          if df.RSI.iloc[p0_idx] > df.RSI.iloc[p2_idx]:
            log += 'check rsi: [{}]{} > [{}]{}'.format(p2_idx, df.RSI.iloc[p2_idx], p4_idx, df.RSI.iloc[p4_idx])
            if df.RSI.iloc[p2_idx] > df.RSI.iloc[p4_idx]:
              log += 'double-regular-bullish-divergence '
              df.at[p0_idx, 'DIVERGENCE_RSI'] = 'double-regular-bullish-divergence'
              df.at[p0_idx, 'DIVERGENCE_RSI_FROM'] = p4_idx
            else:
              log += 'regular-bullish-divergence '
              df.at[p0_idx, 'DIVERGENCE_RSI'] = 'regular-bullish-divergence'
              df.at[p0_idx, 'DIVERGENCE_RSI_FROM'] = p2_idx

        logger.debug(log)
        return

      # is an undefined trend, then discard calculation
      log += 'error-no-trend'      
      logger.debug(log)
      #---end-of-search-function

    # execute search
    _df.apply(lambda x: search(x, _df, _nan_value, self.__logger, exitAt), axis=1)
    self.__df = _df

    # check signals on last row
    _event = self.getCurrentEvent()
    return self.__df, _event


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def getCurrentEvent(self):
  #-------------------------------------------------------------------
    _event = {'macd': None, 'rsi': None}
    if self.__df.DIVERGENCE_MACD.iloc[-1] != 'none':
      _event['macd'] = self.__df.DIVERGENCE_MACD.iloc[-1]
    if self.__df.DIVERGENCE_RSI.iloc[-1] != 'none':
      _event['rsi'] = self.__df.DIVERGENCE_RSI.iloc[-1]
    return _event


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def drawIndicator(self):
  #-------------------------------------------------------------------
    _divergences = self.__df[(self.__df.DIVERGENCE_MACD == self.__df.DIVERGENCE_RSI) & (self.__df.DIVERGENCE_MACD != 'none')][['TIME','OPEN','HIGH','LOW','CLOSE','ZIGZAG','ACTION','DIVERGENCE_MACD', 'DIVERGENCE_MACD_FROM']]

    def buildDivergenceSignal(row, df, fig):
      _from = row.DIVERGENCE_MACD_FROM
      _to = row.name  
      _trace_price = go.Scatter(x=np.array([_from,_to]), y=np.array([df.ZIGZAG[_from], df.ZIGZAG[_to]]), line=scatter.Line(color='blue', width=1))
      fig.append_trace(_trace_price, 1, 1)
      _trace_macd = go.Scatter(x=np.array([_from,_to]), y=np.array([df.MACD_main[_from], df.MACD_main[_to]]), line=scatter.Line(color='black', width=1))
      fig.append_trace(_trace_macd, 2, 1)
      _trace_rsi = go.Scatter(x=np.array([_from,_to]), y=np.array([df.RSI[_from], df.RSI[_to]]), line=scatter.Line(color='black', width=1))
      fig.append_trace(_trace_rsi, 3, 1)

    # Plot ohlc,zigzag, MACD and RSI
    # setup plotting figure with 3 rows and 1 column
    fig = plotly.tools.make_subplots(rows=3, cols=1, subplot_titles=('Price', 'Oscillators'), shared_xaxes=True, vertical_spacing=0.1)

    trace_ohlc = go.Ohlc(x=self.__df.index.values, open=self.__df.OPEN, high=self.__df.HIGH, low=self.__df.LOW, close=self.__df.CLOSE, name='Candlestick')
    fig.append_trace(trace_ohlc, 1, 1)

    _dfz = self.__df[self.__df.ZIGZAG > 0].copy()
    trace_zigzag = go.Scatter(x=_dfz.reset_index()['index'], y=_dfz.ZIGZAG, name='zigzag', line=scatter.Line(color='black', width=1))
    fig.append_trace(trace_zigzag, 1, 1)

    trace_macd = go.Scatter(x=self.__df.index.values, y=self.__df.MACD_main, name='macd', line=scatter.Line(color='blue', width=1))
    fig.append_trace(trace_macd, 2, 1)

    trace_rsi = go.Scatter(x=self.__df.index.values, y=self.__df.RSI, name='rsi', line=scatter.Line(color='red', width=1))
    fig.append_trace(trace_rsi, 3, 1)

    # add signals of divergence to both oscillators and price
    _divergences.apply(lambda x: buildDivergenceSignal(x, self.__df, fig), axis=1)

    fig['layout'].update(height=600, title='Divergences')

    # reference result
    self.__fig = fig
    return self.__fig

