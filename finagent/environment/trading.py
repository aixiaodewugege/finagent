import os
from datetime import datetime
from typing import Any
import numpy as np
import gym
from finagent.registry import ENVIRONMENT

@ENVIRONMENT.register_module(force=True)
class EnvironmentTrading(gym.Env):
    def __init__(self,
                 mode: str = "train",
                 dataset: Any = None,
                 selected_asset: str = "AAPL",
                 asset_type: str = "company",
                 start_date: str = None,
                 end_date: str = None,
                 look_back_days: int = 14,
                 look_forward_days: int = 14,
                 initial_amount: float = 1e4,
                 transaction_cost_pct: float = 1e-3,
                 discount: float = 1.0,
                 ):
        super(EnvironmentTrading, self).__init__()

        self.mode = mode
        self.dataset = dataset
        self.selected_asset = selected_asset
        self.asset_type = asset_type
        self.symbol = selected_asset

        self.prices = self.dataset.prices
        self.news = self.dataset.news
        self.guidances = None
        self.sentiments = None
        self.economics = None

        self.prices_df = self.prices[self.selected_asset]
        self.news_df = self.news[self.selected_asset]
        if self.guidances is not None:
            self.guidances_df = self.guidances[self.selected_asset]
        else:
            self.guidances_df = None

        if self.sentiments is not None:
            self.sentiments_df = self.sentiments[self.selected_asset]
        else:
            self.sentiments_df = None

        if self.economics is not None:
            self.economics_df = self.economics
        else:
            self.economics_df = None

        self.start_date = start_date
        self.end_date = end_date
        self.start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(self.end_date, "%Y-%m-%d")

        self.look_back_days = look_back_days
        self.look_forward_days = look_forward_days

        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.discount = discount

        self.prices_df = self.prices_df.reset_index(drop=True)
        self.news_df = self.news_df.reset_index(drop=True)

        if self.guidances_df is not None:
            self.guidances_df = self.guidances_df.reset_index(drop=True)
        if self.sentiments_df is not None:
            self.sentiments_df = self.sentiments_df.reset_index(drop=True)
        if self.economics_df is not None:
            self.economics_df = self.economics_df.reset_index(drop=True)

        self.init_day = self.prices_df[self.prices_df["timestamp"] >= start_date].index.values[0]
        self.end_day = self.prices_df[self.prices_df["timestamp"] <= end_date].index.values[-1]

        self.prices_df = self.prices_df.set_index("timestamp")
        self.news_df = self.news_df.set_index("timestamp")

        self.day = self.init_day
        self.value = self.initial_amount
        self.cash = self.initial_amount
        self.position = 0
        self.ret = 0
        self.date = self.get_current_date()
        self.price = self.get_current_price()
        self.discount = 1.0
        self.total_return = 0

        self.action_map = {
            "SELL": -1,
            "HOLD": 0,
            "BUY": 1,
        }

        self.action_dim = 3  # buy, hold, sell
        self.action_radius = int(np.floor(self.action_dim / 2))

    def get_current_date(self):
        return self.prices_df.index[self.day]

    def get_current_price(self):
        return self.prices_df.iloc[self.day]["close"]

    def current_value(self, price):
        return self.cash + self.position * price

    def get_state(self):

        state = {}

        days_ago = self.prices_df.index[self.day - self.look_back_days]
        days_future = self.prices_df.index[min(self.day + self.look_forward_days, len(self.prices_df) - 1)]

        price = self.prices_df[self.prices_df.index <= days_future]
        price = price[price.index >= days_ago]

        news = self.news_df[self.news_df.index <= days_future]
        news = news[news.index >= days_ago]


        state["price"] = price
        state["news"] = news

        return state

    def reset(self, **kwargs):
        self.day = self.init_day
        self.value = self.initial_amount
        self.cash = self.initial_amount
        self.position = 0
        self.ret = 0
        self.date = self.get_current_date()
        self.price = self.get_current_price()
        self.discount = 1.0
        self.total_return = 0
        self.total_profit = 0
        self.action = "HOLD"

        state = self.get_state()

        info = {
            "symbol": str(self.symbol),
            "asset_type": str(self.asset_type),
            "day": int(self.day),
            "value": float(self.value),
            "cash": float(self.cash),
            "position": int(self.position),
            "ret": float(self.ret),
            "date": self.date.strftime('%Y-%m-%d'),
            "price": float(self.price),
            "discount": float(self.discount),
            "total_profit": float(self.total_profit),
            "total_return": float(self.total_return),
            "action": self.action
        }

        return state, info

    def eval_buy_position(self, price):
        # evaluate buy position
        # price * position + price * position * transaction_cost_pct <= cash
        # position <= cash / price / (1 + transaction_cost_pct)
        return int(np.floor(self.cash / price / (1 + self.transaction_cost_pct)))

    def eval_sell_position(self):
        # evaluate sell position
        return int(self.position)
    
    # def eval_sell_position(self):
    #     max_short_position = int(self.initial_amount / self.price / (1 + self.transaction_cost_pct))
    #     return int(self.position + max_short_position)


    def buy(self, price, amount=1):

        # evaluate buy position
        eval_buy_postion = self.eval_buy_position(price)

        # predict buy position
        buy_position = int(np.floor((1.0 * np.abs(amount / self.action_radius)) * eval_buy_postion))
        if buy_position == 0:
            self.action = "HOLD"
        else:
            self.action = "BUY"

        self.cash -= buy_position * price * (1 + self.transaction_cost_pct)
        self.position += buy_position
        self.value = self.current_value(price)

    def sell(self, price, amount=-1):

        # evaluate sell position
        eval_sell_postion = self.eval_sell_position()

        # predict sell position
        sell_position = int(np.floor((1.0 * np.abs(amount / self.action_radius)) * eval_sell_postion))
        if sell_position == 0:
            self.action = "HOLD"
        else:
            self.action = "SELL"

        self.cash += sell_position * price * (1 - self.transaction_cost_pct)
        self.position -= sell_position
        self.value = self.current_value(price)

    def hold_on(self, price, amount=0):
        self.action = "HOLD"
        self.value = self.current_value(price)

    def step(self, action: int = 0):

        pre_value = self.value

        if action > 0:
            self.buy(self.price, amount=action)
        elif action < 0:
            self.sell(self.price, amount=action)
        else:
            self.hold_on(self.price, amount=action)

        post_value = self.value

        reward = (post_value - pre_value) / pre_value

        self.day = self.day + 1

        if self.day < self.end_day:
            done = False
            truncted = False
        else:
            done = True
            truncted = True

        next_state = self.get_state()
        self.state = next_state

        self.value = post_value
        self.cash = self.cash
        self.position = self.position
        self.ret = (post_value - pre_value) / pre_value
        self.date = self.get_current_date()

        self.price = self.get_current_price()
        self.total_return += self.discount * reward
        self.discount *= 0.99
        self.total_profit = 100 * (self.value - self.initial_amount) / self.initial_amount

        info = {
            "symbol": str(self.symbol),
            "asset_type": str(self.asset_type),
            "day": int(self.day),
            "value": float(self.value),
            "cash": float(self.cash),
            "position": int(self.position),
            "ret": float(self.ret),
            "date": self.date.strftime('%Y-%m-%d'),
            "price": float(self.price),
            "discount": float(self.discount),
            "total_profit": float(self.total_profit),
            "total_return": float(self.total_return),
            "action": str(self.action)
        }

        return next_state, reward, done, truncted, info
