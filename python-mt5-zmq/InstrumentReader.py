#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    InstrumentReader.py
    
    Template class for reading instrument market data using the Darwinex ZeroMQ Connector
    for Python 3 and MetaTrader 4 release v2.0.2
    
    The strategy launches 2 thread: one for reading bid-ask prices and other for reading
    OHLC rates at M1 timeframe
    
    Each reader must:
        
        1) Configure symbol for bid-ask reporting
        
        2) Configure instrument for OHLc reporting
           
        3) Receive data
        
    --
    
    @author: raulMrello        
"""

import os

#############################################################################
#############################################################################
_path = './dwx-zeromq-connector/v2.0.2/python/'
os.chdir(_path)

#############################################################################
#############################################################################
from examples.template.strategies.base.DWX_ZMQ_Strategy import DWX_ZMQ_Strategy
from pandas import Timedelta, to_datetime
from threading import Thread, Lock
from time import sleep
import random

class InstrumentReader(DWX_ZMQ_Strategy):
    
    def __init__(self, _name="InstrumentReader",
                 _symbols=['EURUSD'],
                 _instruments=['EURUSD_M1'],
                 _broker_gmt=3,
                 _verbose=True):
        
        super().__init__(_name,
                         _symbols,
                         _broker_gmt,
                         _verbose)
        
        # This strategy's variables
        self._instruments = _instruments
        self._verbose = _verbose

        # Reporting threads
        self._reporters = []
        
        # lock for acquire/release of ZeroMQ connector
        self._lock = Lock()
        
    ##########################################################################
    
    def _run_(self):
        
        """
        Logic:
            
            For each symbol in self._symbols and instrument in self._instruments:
                
                1) Request symbol|instrumen configuration
                2) Poll price|rate data
        """
        
        # Launch bid-ask reporters!
        for _symbol in self._symbols:
            
            _t = Thread(name="{}_Reporter".format(_symbol[0]),
                        target=self._price_reporter_, args=(_symbol))
            
            _t.daemon = True
            _t.start()
            
            print('[{}_Reporter] Alright, here we go.'.format(_symbol[0]))
            
            self._reporters.append(_t)

        # Launch rates reporters!
        for _instrument in self._instruments:
            
            _t = Thread(name="{}_Reporter".format(_instrument[0]),
                        target=self._rate_reporter_, args=(_instrument))
            
            _t.daemon = True
            _t.start()
            
            print('[{}_Reporter] Alright, here we go.'.format(_instrument[0]))
            
            self._reporters.append(_t)


        
        print('\n\n+--------------+\n+ LIVE UPDATES +\n+--------------+\n')
        
        # _verbose can print too much information.. so let's start a thread
        # that prints an update for instructions flowing through ZeroMQ
        self._updater_ = Thread(name='Live_Updater',
                               target=self._updater_,
                               args=(self._delay,))
        
        self._updater_.daemon = True
        self._updater_.start()
        
    ##########################################################################
    
    def _updater_(self, _delay=0.1):
        
        while self._market_open:
            
            try:
                # Acquire lock
                self._lock.acquire()
                
                print('\r{}'.format(str(self._zmq._get_response_())), end='', flush=True)
                
            finally:
                # Release lock
                self._lock.release()
        
            sleep(self._delay)
            
    ##########################################################################
    
    def _price_reporter_(self, _symbol):
        
        # Note: Just for this example, only the Order Type is dynamic.
        _default_order = self._zmq._generate_default_order_dict()
        _default_order['_symbol'] = _symbol[0]
        _default_order['_lots'] = _symbol[1]
        _default_order['_SL'] = _default_order['_TP'] = 100
        _default_order['_comment'] = '{}_Trader'.format(_symbol[0])
        
        """
        Default Order:
        --
        {'_action': 'OPEN',
         '_type': 0,
         '_symbol': EURUSD,
         '_price':0.0,
         '_SL': 100,                     # 10 pips
         '_TP': 100,                     # 10 pips
         '_comment': 'EURUSD_Trader',
         '_lots': 0.01,
         '_magic': 123456}
        """
        
        while self._market_open:
            
            try:
                
                # Acquire lock
                self._lock.acquire()
            
                #############################
                # SECTION - GET OPEN TRADES #
                #############################
                
                _ot = self._reporting._get_open_trades_('{}_Trader'.format(_symbol[0]),
                                                        self._delay,
                                                        10)
                
                # Reset cycle if nothing received
                if self._zmq._valid_response_(_ot) == False:
                    continue
                
                ###############################
                # SECTION - CLOSE OPEN TRADES #
                ###############################
                
                for i in _ot.index:
                    
                    if abs((Timedelta((to_datetime('now') + Timedelta(self._broker_gmt,'h')) - to_datetime(_ot.at[i,'_open_time'])).total_seconds())) > self._close_t_delta:
                        
                        _ret = self._execution._execute_({'_action': 'CLOSE',
                                                          '_ticket': i,
                                                          '_comment': '{}_Trader'.format(_symbol[0])},
                                                          self._verbose,
                                                          self._delay,
                                                          10)
                       
                        # Reset cycle if nothing received
                        if self._zmq._valid_response_(_ret) == False:
                            break
                        
                        # Sleep between commands to MetaTrader
                        sleep(self._delay)
                
                ##############################
                # SECTION - OPEN MORE TRADES #
                ##############################
                
                if _ot.shape[0] < _max_trades:
                    
                    # Randomly generate 1 (OP_BUY) or 0 (OP_SELL)
                    # using random.getrandbits()
                    _default_order['_type'] = random.getrandbits(1)
                    
                    # Send instruction to MetaTrader
                    _ret = self._execution._execute_(_default_order,
                                                     self._verbose,
                                                     self._delay,
                                                     10)
                  
                    # Reset cycle if nothing received
                    if self._zmq._valid_response_(_ret) == False:
                        break
                
            finally:
                
                # Release lock
                self._lock.release()
            
            # Sleep between cycles
            sleep(self._delay)
            
    ##########################################################################
    
    def _stop_(self):
        
        self._market_open = False
        
        for _t in self._traders:
        
            # Setting _market_open to False will stop each "trader" thread
            # from doing anything more. So wait for them to finish.
            _t.join()
            
            print('\n[{}] .. and that\'s a wrap! Time to head home.\n'.format(_t.getName()))
        
        # Kill the updater too        
        self._updater_.join()
        
        print('\n\n{} .. wait for me.... I\'m going home too! xD\n'.format(self._updater_.getName()))
        
        # Send mass close instruction to MetaTrader in case anything's left.
        self._zmq._DWX_MTX_CLOSE_ALL_TRADES_()
        
    ##########################################################################