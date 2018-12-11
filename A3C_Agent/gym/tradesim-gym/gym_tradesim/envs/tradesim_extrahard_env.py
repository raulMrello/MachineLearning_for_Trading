import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import logging

log = logging.getLogger(__name__)
log.info('%s logger started.',__name__)

    
#######################################################################################
def pull_predicted_data(step):
  return {'high':0.0, 'low':0.0}

    
#######################################################################################
def pull_last_tick(instrument, step):
  return {'ask':0.0, 'bid':0.0}


#######################################################################################
#######################################################################################
# Clase AccountStatus
# Permite acceder al estado de la cuenta, así como realizar las operaciones open-close
#######################################################################################
#######################################################################################

class AccountStatus:
  #-------------------------------------------------------------------------------------
  def __init__(self, instrument, initial_equity, leverage, commissions, cb_tick, cb_market=None):
    """ Constructor:
    Args:
      instrument     : Instrumento. Ejemplo 'eurusd_h1'
      initial_equity : Patrimonio inicial
      leverage       : Apalancamiento
      commissions    : Comisiones fijas por operación
      cb_tick        : Callback de acceso a precios instantáneos ask,bid
      cb_market      : Callback de acceso al mercado (por defecto self._market_query)
    """
    self.instrument = instrument
    self.tick = {'ask':0.0, 'bid':0.0}
    self.cb_tick = cb_tick
    self.cb_market = cb_market if cb_market is not None else self._market_query
    self.risk = 0.01  # 1% riesgo fijo
    self.commissions = commissions
    self.leverage = leverage  # ej. 30 => 30:1 
    self.initial_equity = initial_equity
    self.reset()
  
  #-------------------------------------------------------------------------------------
  def reset(self):
    """ Inicializa el estado de la cuenta:
    """
    self.position = None
    self.floatpl = 0
    self.pl = 0
    self.pl_last = 0
    self.margin = 0    
    self.balance = self.initial_equity    
    self.equity = self.balance + self.floatpl
    self.free_margin = self.equity - self.margin
    self.margin_level = 100 * self.equity
    self.margin_call  = False   

  #-------------------------------------------------------------------------------------
  def updateStat(self, step):
    """ Actualizo el estado de la cuenta. Se ejecutará una vez por step
    """
    self.pl_last = self.pl
    self.tick = self.cb_tick(self.instrument, step)
    if self.position is not None:
      self.position,account_stat = self.cb_market(op='info', position=self.position)
      if self.position['stat'] == 'closed':
        self.position = None
      self.balance  = account_stat['balance']
      self.equity  = account_stat['equity']
      self.margin = account_stat['margin']
      self.floatpl  = account_stat['floatpl']
      self.free_margin  = account_stat['free_margin']
      self.margin_level  = account_stat['margin_level']
      self.margin_call  = account_stat['margin_call']    
      self.pl = account_stat['pl']

  #-------------------------------------------------------------------------------------
  def openPosition(self, type='long', sl=None, tp=None):
    """ Intenta abrir una posición
    Args:
      type  : Tipo de posición 'long', 'short'
      sl    : Stop-loss
      tp    : Take-profit
    Returns: 
      Éxito, Descripción
    """
    if self.position is not None:
      return False, 'There are opened positions'
    position = {}
    position['volume'] = self.risk * self.equity
    if position['volume'] > self.free_margin:
      return False, 'Not enough money'

    position['margin'] = position['volume']/self.leverage
    position['type'] = type
    position['sl'] = sl
    position['tp'] = tp

    # invoco callback para obtener precio de apertura
    position,account_stat = self.cb_market(op='open', position=position)
    if position['stat'] != 'opened':
      return False, 'Market query error'   

    # actualizo posición en curso
    self.position = position
    
    # actualizo el estado de la cuenta
    self.balance  = account_stat['balance']
    self.equity  = account_stat['equity']
    self.margin = account_stat['margin']
    self.floatpl  = account_stat['floatpl']
    self.pl  = account_stat['pl']
    self.free_margin  = account_stat['free_margin']
    self.margin_level  = account_stat['margin_level']
    self.margin_call  = account_stat['margin_call']    
    return True, 'Opened ' + self.position['type'] + ' at=' + str(self.position['price']) + ' volume=' + str(self.position['volume']) + ' margin=' + str(self.position['margin'])

  #-------------------------------------------------------------------------------------
  def closePosition(self, type='long'):
    """ Intenta cerrar una posición abierta de un tipo dado
    Args:
      type  : Tipo de posición 'long', 'short'
    Returns: 
      Éxito, Descripción
    """
    if self.position is None:
      return False, 'No opened positions'
    if self.position['type'] != type:
      return False, 'Type error'
    self.position,account_stat = self.cb_market(op='close', position=self.position)
    if self.position['stat'] != 'closed':
      return False, 'Market query error'
    self.position = None
    self.balance  = account_stat['balance']
    self.equity  = account_stat['equity']
    self.margin = account_stat['margin']
    self.floatpl  = account_stat['floatpl']
    self.free_margin  = account_stat['free_margin']
    self.margin_level  = account_stat['margin_level']
    self.margin_call  = account_stat['margin_call'] 
    self.pl = account_stat['pl']
    return True, 'Closed position'

  #-------------------------------------------------------------------------------------
  def _market_query(self, op='none', position=None):
    """ Callback interna en caso de no proporcionar 'cb_market'
    Args:
      op       : Tipo de operación 'open, close, info, none'
      position : Posición involucrada o None
    Returns: 
      Position,AccountStat
    """    
    account_stat = {}
    if op=='open':
      position['open_price'] = self.tick['ask'] if position['type']=='long' else self.tick['bid']  
      position['price'] = position['open_price']
      position['stat'] = 'opened'
      account_stat['balance'] = self.balance
      account_stat['pl'] = self.pl
      account_stat['floatpl'] = 0
      account_stat['equity'] = self.equity + account_stat['floatpl']
      # actualizo el margen total
      account_stat['margin'] = position['margin']
      # actualizo el margen libre
      account_stat['free_margin'] = account_stat['equity'] - account_stat['margin']
      # calculo el nivel de margen
      account_stat['margin_level'] = 100 * (account_stat['equity']/account_stat['margin'])
      # chequeo si salta la llamada por nivel de margen
      account_stat['margin_call'] = False
      if account_stat['margin_level'] <= 100:
        account_stat['margin_call'] = True

    elif op=='close':
      position['close_price'] = self.tick['ask'] if position['type']=='short' else self.tick['bid'] 
      position['price'] = position['close_price']
      position['closedpl'] = position['volume'] * (position['close_price'] - position['open_price']) if position['type']=='long' else (position['open_price'] - position['close_price'])
      position['closedpl'] -= self.commissions
      position['stat'] = 'closed'
      account_stat['pl'] = self.pl
      account_stat['floatpl'] = 0
      account_stat['balance'] = self.balance + position['closedpl']
      account_stat['equity'] = account_stat['balance']
      account_stat['pl'] = account_stat['equity'] - self.initial_equity
      # actualizo el margen total
      account_stat['margin'] = 0
      # actualizo el margen libre
      account_stat['free_margin'] = account_stat['equity'] 
      # calculo el nivel de margen
      account_stat['margin_level'] = 100 * account_stat['equity']
      # chequeo si salta la llamada por nivel de margen
      account_stat['margin_call'] = False
      if account_stat['margin_level'] <= 100:
        account_stat['margin_call'] = True
    
    elif op=='info':
      if position is None:
        account_stat['floatpl'] = 0
        account_stat['balance'] = self.balance
        account_stat['equity'] = self.equity
        account_stat['pl'] = self.pl
        # actualizo el margen total
        account_stat['margin'] = 0
        # actualizo el margen libre
        account_stat['free_margin'] = account_stat['equity'] - account_stat['margin']
        # calculo el nivel de margen
        account_stat['margin_level'] = 100 * (account_stat['equity'])
        # chequeo si salta la llamada por nivel de margen
        account_stat['margin_call'] = False
        if account_stat['margin_level'] <= 100:
          account_stat['margin_call'] = True
      else:
        position['price'] = self.tick['ask'] if position['type']=='short' else self.tick['bid']
        account_stat['floatpl'] = position['volume'] * (position['price'] - position['open_price']) if position['type']=='long' else (position['open_price'] - position['price'])
        account_stat['balance'] = self.balance
        account_stat['equity'] = self.equity + account_stat['floatpl']
        account_stat['pl'] = self.pl
        # actualizo el margen total
        account_stat['margin'] = position['margin']
        # actualizo el margen libre
        account_stat['free_margin'] = account_stat['equity'] - account_stat['margin']
        # calculo el nivel de margen
        account_stat['margin_level'] = 100 * (account_stat['equity']/account_stat['margin'])
        # chequeo si salta la llamada por nivel de margen
        account_stat['margin_call'] = False
        if account_stat['margin_level'] <= 100:
          account_stat['margin_call'] = True

    return position,account_stat


#######################################################################################
#######################################################################################
# Clase TradeSimEnv
#######################################################################################
#######################################################################################


class TradeSimEnv(gym.Env):

  metadata = {'render.modes': ['human']}
  NumActions = 5
  NumStates = 12
  InitialEquity = 1000.0
  Leverage = 1.0
  Commissions = 0.05
  TakeStopMargin = 0.2

  #---------------------------------------------------------------------------
  def __init__( self):
    """ 
    Construye el entorno (estados, acciones):
    """
    self.observation_space = spaces.Dict({
      'price_high' : spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
      'price_low' : spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
      'tick_ask' : spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
      'tick_bid' : spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
      'acc_balance': spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),      
      'acc_floatpl' : spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
      'acc_pl' : spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
      'pos_long': spaces.Box(low=0, high=1, shape=(), dtype = np.uint8),
      'pos_short': spaces.Box(low=0, high=1, shape=(), dtype = np.uint8),
      'pos_price': spaces.Box(low=0, high=1.0, shape=(), dtype=np.float32),
      'pos_sl' :  spaces.Box(low=0, high=1.0, shape=(), dtype=np.float32),
      'pos_tp' :  spaces.Box(low=0, high=1.0, shape=(), dtype=np.float32),
    }),
    self.obs = {}
    
    # 0->none   1->open_long   2->close_long  3->open_short  4->close_short
    self.action_space = spaces.Discrete(TradeSimEnv.NumActions)
    
    # inicialización de variables
    self.num_steps = 0
    self.steps = 0


  #---------------------------------------------------------------------------
  def configure(self, 
                steps_per_episode,      
                max_price, 
                max_balance, 
                max_pl, 
                cb_pull_predictions, 
                cb_pull_ticks,
                cb_market_info):
    """ 
    Construye el entorno (estados, acciones):
    Args:
    -------
      steps_per_episode   : Número de pasos del episodio
      max_price           : Valor máximo que puede tomar cualquier precio
      max_balance         : Valor máximo que puede tomar la cuenta
      max_pl              : Valor máximo que pueden tomar los beneficios totales
      cb_pull_predictions : Callback para obtener las predicciones en un step dado
      cb_pull_ticks       : Callback para obtener los ticks en un step dado
      cb_market_info      : Callback para obtener información del terminal de mercado
    """
    """ 
    Construye el entorno (estados, acciones):
    """
    self.observation_space = spaces.Dict({
      'price_high' : spaces.Box(low=0.0, high=max_price, shape=(), dtype=np.float32),
      'price_low' : spaces.Box(low=0.0, high=max_price, shape=(), dtype=np.float32),
      'tick_ask' : spaces.Box(low=0.0, high=max_price, shape=(), dtype=np.float32),
      'tick_bid' : spaces.Box(low=0.0, high=max_price, shape=(), dtype=np.float32),
      'acc_balance': spaces.Box(low=0.0, high=max_balance, shape=(), dtype=np.float32),      
      'acc_floatpl' : spaces.Box(low=0.0, high=max_pl, shape=(), dtype=np.float32),
      'acc_pl' : spaces.Box(low=0.0, high=max_pl, shape=(), dtype=np.float32),
      'pos_long': spaces.Box(low=0, high=1, shape=(), dtype = np.uint8),
      'pos_short': spaces.Box(low=0, high=1, shape=(), dtype = np.uint8),
      'pos_price': spaces.Box(low=0, high=max_price, shape=(), dtype=np.float32),
      'pos_sl' :  spaces.Box(low=0, high=max_price, shape=(), dtype=np.float32),
      'pos_tp' :  spaces.Box(low=0, high=max_price, shape=(), dtype=np.float32),
    }),
    self.obs = {}
    
    # 0->none   1->open_long   2->close_long  3->open_short  4->close_short
    self.action_space = spaces.Discrete(TradeSimEnv.NumActions)
    
    # captura de parámetros
    self.num_steps = steps_per_episode
    self.max_price = max_price
    self.max_balance = max_balance
    self.max_pl = max_pl
    self.cb_pull_predictions = cb_pull_predictions
    
    # inicialización de variables
    self.steps = 0
    self.account = AccountStatus('eurusd_h1', TradeSimEnv.InitialEquity, TradeSimEnv.Leverage, TradeSimEnv.Commissions, cb_pull_ticks, cb_market_info)


  #---------------------------------------------------------------------------
  def _get_state(self):
    obs = {}
    hilo_dict = self.cb_pull_predictions(self.step)
    obs['price_high'] = hilo_dict['high']
    obs['price_low'] =  hilo_dict['low']
    self.account.updateStat(self.step)
    obs['tick_ask']     = self.account.tick['ask']
    obs['tick_bid']     = self.account.tick['bid']
    obs['acc_balance']  = self.account.balance     
    obs['acc_floatpl']  = self.account.floatpl
    obs['acc_pl']       = self.account.pl
    obs['pos_long']     = 0 if self.account.position is None else (1 if self.account.position['type']=='long' else 0)
    obs['pos_short']    = 0 if self.account.position is None else (1 if self.account.position['type']=='short' else 0)
    obs['pos_price']    = 0 if self.account.position is None else self.account.position['price']
    obs['pos_sl']       =  0 if self.account.position is None else self.account.position['sl']
    obs['pos_tp']       =  0 if self.account.position is None else self.account.position['tp']
    self.obs = obs
    return obs


  #---------------------------------------------------------------------------
  def _take_action(self, action):
    tpsl_range = self.obs['price_high'] - self.obs['price_low']
    tpsl_margin = tpsl_range * TradeSimEnv.TakeStopMargin
    tpsl_above = self.obs['price_high'] + tpsl_margin
    tpsl_below = self.obs['price_low'] - tpsl_margin
    action_list = ['none', 'open_long', 'close_long', 'open_short', 'close_short']
    if action_list[action] == 'open_long':      
      done,_ = self.account.openPosition(type='long', sl=tpsl_below, tp=tpsl_above)
    elif action_list[action] == 'close_long':
      done,_ = self.account.closePosition(type='long')
    elif action_list[action] == 'open_short':
      done,_ = self.account.openPosition(type='short', sl=tpsl_above, tp=tpsl_below)
    elif action_list[action] == 'close_short':
      done,_ = self.account.closePosition(type='short')
    

  #---------------------------------------------------------------------------
  def _get_reward(self):
    reward = self.account.pl - self.account.pl_last
    return reward

  #---------------------------------------------------------------------------
  def _get_done(self):
    done = False
    if self.steps >= self.num_steps:
      done = True
    if self.account.margin_call:
      done = True
    return done


  #---------------------------------------------------------------------------
  def reset(self):
    """
    Inicializa el estado del entorno
    Returns:
    -------
      obs : Estado inicial
    """
    self.steps = 0
    self.account.reset()
    return self._get_state()
        

  #---------------------------------------------------------------------------
  def step(self, action):
    """
    Ejecuta una acción y obtiene el resultado de la misma
    Args:
    -------
      action  : Acción a ejecutar
    Returns
    -------
    ob, reward, episode_over, info : tuple
        ob (object) :
            an environment-specific object representing your observation of
            the environment.
        reward (float) :
            amount of reward achieved by the previous action. The scale
            varies between environments, but the goal is always to increase
            your total reward.
        episode_over (bool) :
            whether it's time to reset the environment again. Most (but not
            all) tasks are divided up into well-defined episodes, and done
            being True indicates the episode has terminated. (For example,
            perhaps the pole tipped too far, or you lost your last life.)
        info (dict) :
              diagnostic information useful for debugging. It can sometimes
              be useful for learning (for example, it might contain the raw
              probabilities behind the environment's last state change).
              However, official evaluations of your agent are not allowed to
              use this for learning.

    """
    self._take_action(action)
    reward = self._get_reward()
    self.steps += 1
    ob = self._get_state()
    done = self._get_done()
    return ob, reward, done, {}


  #---------------------------------------------------------------------------
  def render(self, mode='human', close=False):
    return

