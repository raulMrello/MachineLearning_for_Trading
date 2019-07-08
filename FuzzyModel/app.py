import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
####################################################################################

import pandas as pd
import plotly.graph_objs as go
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
from plotly.tools import FigureFactory as FF
import plotly.tools as tls

####################################################################################

import sys
import time
import numpy as np

####################################################################################

# import main class and event class
from MACD_Signal_Listener import MACD_Signal_Listener, MACD_Events

####################################################################################
####################################################################################
####################################################################################

# loads df dataframe
df = pd.read_csv('csv_data/EURUSD_M15.csv', sep=';')
df = df[['OPEN','HIGH','LOW','CLOSE']]

####################################################################################

# defines a common event listener, to be called on MACD event recognition
def macd_listener(events, df):
  print('Received events={}'.format(events.info()))

####################################################################################

# creates a MACD Signal Listener with default parameters
msl = MACD_Signal_Listener(param_num_bars_per_swing=5, param_num_bars_per_minmax_wdow=2)

####################################################################################

# setup example
last_sample  = -2000
execution_count = 0

####################################################################################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    dcc.Graph(id='macd-graph'),
    dcc.Interval(
            id='interval-timer',
            interval=1000, # in milliseconds
            n_intervals=0
        )
])


@app.callback(
    Output('macd-graph', 'figure'),
    [Input('interval-timer', 'n_intervals')])
def datafeed_update(n):
  last_sample = -2000
  execution_count = n

  # setup plotting figure with 2 rows and 1 column
  fig = plotly.tools.make_subplots(rows=2, cols=1, subplot_titles=('Price', 'MACD'), shared_xaxes=True, vertical_spacing=0.1)
  fig['layout'].update(height=600, title='Price & MACD', xaxis=dict(rangeslider=dict(visible = False)))

  # updates datafeed samples
  _from_sample = last_sample + execution_count
  _to_sample = _from_sample + 1000

  # executes MACD signal generator
  _events, _df_result = msl.MACD( df[_from_sample : _to_sample],
                                  applied = 'CLOSE', 
                                  fastperiod=12, 
                                  slowperiod=26, 
                                  signalperiod=9,
                                  common_event_listeners=[macd_listener],
                                  bullish_divergence_listeners=[],
                                  bearish_divergence_listeners=[],
                                  bullish_main_zero_cross_listeners=[],
                                  bearish_main_zero_cross_listeners=[],
                                  bullish_main_signal_cross_listeners=[],
                                  bearish_main_signal_cross_listeners=[])


  # Plot results
  _df_draw = _df_result
  curr_sample = 0
  last_sample = _df_draw.shape[0] - 1
  
  # build candlestick trace
  trace_ohlc = go.Ohlc(x=_df_draw.index.values, open=_df_draw.OPEN, high=_df_draw.HIGH, low=_df_draw.LOW, close=_df_draw.CLOSE, name='Candlestick')
  fig.append_trace(trace_ohlc, 1, 1)

  # build MACD traces
  trace_macd = go.Scatter(x=_df_draw.index.values, y=_df_draw.MACD, name='MACD', line=scatter.Line(color='blue', width=1))
  fig.append_trace(trace_macd, 2, 1)
  trace_macdsig = go.Scatter(x=_df_draw.index.values, y=_df_draw.MACDS, name='MACD_sig', line=scatter.Line(color='red', width=1))
  fig.append_trace(trace_macdsig, 2, 1)

  # build BEAR_DIVERGENCES on both charts (black)
  # select divergence zones
  div_traces_row1 = []
  div_traces_row2 = []
  # get divergences as an array and just select those in the plotting range
  bear_div_zones = msl.getBearishDivergences()
  if len(bear_div_zones) > 0:
    _bear_div_zones = np.asarray(bear_div_zones)
    _bear_div_zones = _bear_div_zones[(_bear_div_zones[:,0] >= curr_sample) & (_bear_div_zones[:,0] <= last_sample)]
    #print('Bearish Divergence zones={}'.format(len(_bear_div_zones)))
    # for each one, build a pair of traces, one for each row
    for d in _bear_div_zones:
      #print('added bear_div at {}'.format(d - curr_sample))
      # add trace to row1 list
      div_traces_row1.append(go.Scatter(
              x= d - curr_sample, 
              y= [_df_draw.HIGH.at[d[0]], _df_draw.HIGH.at[d[1]]],     
              name='bear_div_r1 at {}'.format(d - curr_sample),
              line=scatter.Line(color='black', width=1)))
      # add trace to row2 list
      div_traces_row2.append(go.Scatter(
              x= d - curr_sample, 
              y= [_df_draw.MACD.at[d[0]], _df_draw.MACD.at[d[1]]],     
              name='bear_div_r2 at {}'.format(d - curr_sample),
              line=scatter.Line(color='black', width=1)))


  # build BULL_DIVERGENCES on both charts (orange)
  # get divergences as an array and just select those in the plotting range
  bull_div_zones = msl.getBullishDivergences()
  if len(bull_div_zones) > 0:
    _bull_div_zones = np.asarray(bull_div_zones)
    _bull_div_zones = _bull_div_zones[(_bull_div_zones[:,0] >= curr_sample) & (_bull_div_zones[:,0] <= last_sample)]
    #print('Bullish Divergence zones={}'.format(len(_bull_div_zones)))
    # for each one, build a pair of traces, one for each row
    for d in _bull_div_zones:
      #print('added bull_div at {}'.format(d - curr_sample))
      # add trace to row1 list
      div_traces_row1.append(go.Scatter(
              x= d - curr_sample, 
              y= [_df_draw.LOW.at[d[0]], _df_draw.LOW.at[d[1]]],     
              name='bull_div_r1 at {}'.format(d - curr_sample),
              line=scatter.Line(color='orange', width=1)))
      # add trace to row2 list
      div_traces_row2.append(go.Scatter(
              x= d - curr_sample, 
              y= [_df_draw.MACD.at[d[0]], _df_draw.MACD.at[d[1]]],     
              name='bull_div_r2 at {}'.format(d - curr_sample),
              line=scatter.Line(color='orange', width=1)))

  for d in div_traces_row1:
    fig.append_trace(d, 1, 1)
  for d in div_traces_row2:
    fig.append_trace(d, 2, 1)
  return fig


if __name__ == '__main__':
    app.run_server(debug=True)

