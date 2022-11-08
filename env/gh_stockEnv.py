from __future__ import annotations
from typing import Union

import gym
import numpy as np
from numpy import random as rd

## fractional trading - WITH PERCENTAGES - WITH SEQUENCE INPUT TO RNN
## we start with initial_capital amount of cash AND initial_stocks number of stocks
## NEW STOCK ALLOCATION METHOD
class StockTradingEnvSeq(gym.Env):
    def __init__(
        self,
        seq_price_array,
        actual_price_array,
        seq_tech_array,
        seq_turb_array,
        if_train,
        gamma=0.99,
        turbulence_thresh=99,
        min_amount_rate=.1,  ## minimum action to be taken as a percentage
        max_amount_rate=.8, ## maximum amount to be traded as percentage of total_assets
        initial_capital=100, ## total amount of $ to work with;
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=2**-11,
        initial_stocks=None, ## number of stocks held for each ticker at start; this is ON TOP OF initial capital
        lookback = 10,
        **kwargs
    ):
        self.lookback = lookback
        self.seq_price_array = seq_price_array
        self.actual_price_array = actual_price_array.astype(np.float32)
        self.seq_tech_array = seq_tech_array.astype(np.float32)
        self.seq_turb_array = seq_turb_array

        #self.tech_ary_scaled = self.tech_ary_scaled * 2**-7
        self.turbulence_bool = (self.seq_turb_array[:, -1, :] > turbulence_thresh).astype(np.float32)

        self.stock_dim = self.actual_price_array.shape[1]
        self.gamma = gamma
        self.max_amount_rate = max_amount_rate
        self.min_amount_rate = min_amount_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(self.stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )
        self.allocation = np.zeros((self.lookback, self.stock_dim + 1), dtype=np.float32)
        self.total_asset = np.zeros((self.lookback, 1), dtype=np.float32)
        self.alloc_history = []

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = "StockEnv"
        self.state_dim = 2 + 2*(self.stock_dim) + self.seq_tech_array.shape[2] + 1
        # (turbulence, NO turbulence_bool, allocation[HOLD]) + (price, allocation)*stock_dim  + tech_dim + total_asset

        self.action_dim = self.stock_dim ## actions are all stocks
        self.max_step = kwargs.get("max_step", self.actual_price_array.shape[0] - self.lookback)
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = kwargs.get("target_return", np.inf)
        self.episode_return = 0.0
        self.env_num = kwargs.get("env_num", 1)

    def reset(self): ## initial stocks do NOT come out of amount; but random investments DO
        self.day = 0

        if self.if_train:

            ## get allocation for 9 prior "days"
            price = self.actual_price_array[0]
            rand_stock = np.random.uniform(0, 1, size = self.initial_stocks.shape)
            stock_alloc = (.8*rand_stock / rand_stock.sum())
            
            self.allocation[0] = np.concatenate((stock_alloc, np.array([1 - stock_alloc.sum()])))
            self.stocks = (self.allocation[0, :-1]*self.initial_capital/price) + self.initial_stocks
            amount = self.initial_capital - (self.stocks*price).sum() + (self.initial_stocks*price).sum()
            self.amount = np.full((self.lookback, 1), amount, dtype=np.float32)
            self.total_asset[0] = amount

            for i in range(1, self.lookback):
                price = self.actual_price_array[i]
                assets = amount + (self.stocks*price).sum()
                self.allocation[i] = np.concatenate((self.stocks*price / assets, np.array([1 - (self.stocks*price / assets).sum()])))
                self.alloc_history.append(self.allocation[i]) ## ADDED
                self.total_asset[i] = amount + (self.stocks * price).sum()
            
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.amount = np.full((self.lookback, 1), self.initial_capital, dtype=np.float32)
            
            for i in range(0, self.lookback):
                price = self.actual_price_array[i]
                assets = self.amount[-1] + (self.stocks * price).sum()
                self.allocation[i] = np.concatenate((self.stocks*price / assets, np.array([1 - (self.stocks*price / assets).sum()])))
                self.alloc_history.append(self.allocation[i]) ## ADDED
                self.total_asset[i] = assets

        #self.total_asset = self.amount + (self.stocks * self.actual_price_array[self.lookback - 1]).sum()
        self.stocks_cool_down = np.zeros_like(self.stocks)
        self.initial_total_asset = self.total_asset[0,0]
        self.gamma_reward = 0.0

        return self.get_state()  # state

    def step(self, actions):
        ## advance day, stocks_cool_down
        self.day += 1
        self.stocks_cool_down += 1

        ## get non-action-related values needed for calculations
        price = self.actual_price_array[self.day + self.lookback - 1]
        amount = self.amount[-1, 0]
        assets = amount + (self.stocks * price).sum()

        ## set small actions to 0, get positive actions, and calculate total purchase amount for use later
        actions[np.abs(actions) < self.min_amount_rate] = 0
        pos_actions = actions[actions>0]
        purchase_amt = (pos_actions*self.max_amount_rate*assets / self.stock_dim).sum()
        
        ## if no turbulence, buy/sell according to actions
        if self.turbulence_bool[self.day] == 0:
            ## sell
            for i in np.where(actions < 0)[0]:
                ## can only sell what we own
                sell_amount = min(self.stocks[i]*price[i], -actions[i]*self.max_amount_rate*assets / self.stock_dim)
                
                self.stocks[i] -= sell_amount / price[i]
                amount += (sell_amount * (1 - self.sell_cost_pct))
                self.stocks_cool_down[i] = 0
            
            ## buy
            for j in np.where(actions > 0)[0]:
                ## if amount is greater than the purchase amount (adjusted for cost), buy; otherwise, split amount by percentage
                buy_amount = (actions[j]*self.max_amount_rate*assets / self.stock_dim) \
                                if amount >= purchase_amt*(1 + self.buy_cost_pct) \
                                else amount*(actions[j] / pos_actions.sum()) / (1 + self.buy_cost_pct)

                self.stocks[j] += buy_amount / price[j]
                amount -= (buy_amount * (1 + self.buy_cost_pct))
                self.stocks_cool_down[j] = 0

        # sell all when turbulence
        else:
            amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.stocks_cool_down[:] = 0

        ## update amount and allocation
        self.amount = self.update_attr(self.amount, amount)
        total_asset = self.amount[-1, 0] + (self.stocks * price).sum()
        self.allocation = self.update_attr(
            self.allocation, 
            np.concatenate((self.stocks*price / total_asset, np.array([1 - (self.stocks*price / total_asset).sum()])))
            )
        self.alloc_history.append(self.allocation[-1])
        
        ## update reward and total asset; compute gamme_reward, done, and episode_return if needed
        reward = (total_asset - self.total_asset[-1, 0]) * self.reward_scaling
        self.total_asset = self.update_attr(self.total_asset, total_asset)

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = (self.day == self.max_step)
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        return self.get_state(), reward, done, dict()

    def get_state(self):
        return np.hstack((
            self.seq_turb_array[self.day],
            self.seq_price_array[self.day],
            self.seq_tech_array[self.day],
            self.allocation,
            self.total_asset
            ))[np.newaxis,:,:]
    
    @staticmethod
    def update_attr(attr: np.ndarray, entry: Union[int, float, np.ndarray]) -> np.ndarray:
        attr = np.roll(attr, shift = -1, axis = 0) ## shift entries back by 1 to make room for most recent entry
        attr[-1] = entry
        return attr
    
    ## not curently used, used in prior iteration
    @staticmethod
    def softmax(x:np.ndarray) -> np.ndarray:
        return np.exp(x) / np.exp(x).sum()