# Classes.py
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
from gymnasium import spaces

import time
nextId = 1
def next_order_id(isbuy: bool):
    global nextId

    if isbuy: prefix = 1
    else: prefix = 2

    order_id = prefix * 1000000 + nextId
    nextId += 1
    return order_id

# Shared portfolio registry
portfolios = {}

class Portfolio:
    def __init__(self, id, cash, holdings=None):
        self.id = id
        self.cash = cash
        self.holdings = holdings if holdings is not None else {}

class Stock:
    def __init__(self, id, pricepoint, name, borderArr, sofferArr):
        self.id = id
        self.pricepoint = pricepoint
        self.name = name
        self.borderArr = borderArr
        self.sofferArr = sofferArr

class PPinpoint:
    def __init__(self, highestindx=-1, lowestindx=-1):
        self.highestindx = highestindx
        self.lowestindx = lowestindx

class Order:
    def __init__(self, id, price, quantity, marketorlimit, stock_id, portfolio_id):
        self.id = id
        self.price = price
        self.quantity = quantity
        self.marketorlimit = marketorlimit
        self.timestamp = time.time()
        self.stock_id = stock_id
        self.portfolio_id = portfolio_id

############################################CNN FOR PPO/SAC ######################################ž

class CustomCNN(BaseFeaturesExtractor):

  def __init__(
      self,
      observation_space: spaces.Box,
      features_dim: int = 128,
      price_features: int = 1,
      ind_features: int = 5,
  ):
    super().__init__(observation_space, features_dim)

    self.price_features = price_features
    self.ind_features = ind_features

    self.cnn = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1), 
        nn.ReLU(),
        nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
    )

    cnn_out_dim = 32 * price_features
    total_concat_dim = cnn_out_dim + ind_features

    self.linear = nn.Sequential(
        nn.Linear(total_concat_dim, features_dim), nn.ReLU()
    )

  def forward(self, observations: th.Tensor) -> th.Tensor:
    priceobs = observations[:, : self.price_features].unsqueeze(1) 
    ind = observations[:, self.price_features :] 

    cnn_feat = self.cnn(priceobs)  
    combined = th.cat([cnn_feat, ind], dim=1) 

    return self.linear(combined)

######################################## RNN FOR PPO ####################################################3

class CustomRNN(BaseFeaturesExtractor):

    def __init__(

        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        price_features: int = 1,
        ind_features: int = 5,
        hidden_size: int = 64,
    ):
        super().__init__(observation_space, features_dim)

        self.price_features = price_features
        self.ind_features = ind_features

        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)

        total_concat_dim = hidden_size + ind_features
        self.linear = nn.Sequential(
            nn.Linear(total_concat_dim, features_dim), nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        priceobs = observations[:, : self.price_features].unsqueeze(-1)
        ind = observations[:, self.price_features :] 

        rnn_out, h_n = self.rnn(priceobs)
        last_hidden = h_n.squeeze(0)

        combined = th.cat([last_hidden, ind], dim=1) 
        return self.linear(combined)

policy_kwargs_def = None

policy_kwargs_cnn = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(
        price_features=10, ind_features=14, features_dim=128
    ),
)

policy_kwargs_rnn = dict(
    features_extractor_class=CustomRNN,
    features_extractor_kwargs=dict(
        price_features=10, ind_features=14, hidden_size=64, features_dim=128
    ),
)