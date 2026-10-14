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
      price_features: int = 100,
      vol_features: int = 100,
      ind_features: int = 4,
  ):
    super().__init__(observation_space, features_dim)

    self.price_features = price_features
    self.ind_features = ind_features
    self.vol_features = vol_features

    self.cnn = nn.Sequential(
        nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1), 
        nn.ReLU(),
        nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
    )

    cnn_out_dim = 64 * price_features
    total_concat_dim = cnn_out_dim + ind_features

    self.linear = nn.Sequential(
        nn.Linear(total_concat_dim, features_dim), nn.ReLU()
    )

  def forward(self, observations: th.Tensor) -> th.Tensor:
    priceobs = observations[:, : self.price_features]
    volobs = observations[:, self.price_features : self.price_features + self.vol_features]
    ind = observations[:, self.price_features + self.vol_features :] 

    cnn_comb=th.stack([priceobs, volobs], dim=1) 
    comb = self.cnn(cnn_comb)
    combined = th.cat([comb, ind], dim=1)

    return self.linear(combined)

######################################## RNN FOR PPO ####################################################3

class CustomRNN(BaseFeaturesExtractor):

    def __init__(

        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        price_features: int = 10,
        ind_features: int = 14,
        hidden_size: int = 64,
    ):
        super().__init__(observation_space, features_dim)

        self.price_features = price_features
        self.ind_features = ind_features

        self.rnn = nn.RNN(input_size=1, 
                          hidden_size=hidden_size, 
                          num_layers=1, 
                          batch_first=True)

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

######################################## GRU FOR PPO ####################################################3

import torch
import torch.nn as nn

class CustomGRU(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        price_features: int = 10,
        ind_features: int = 14,
        hidden_size: int = 64,
        num_layers: int = 1,
    ):
        super().__init__(observation_space, features_dim)

        self.price_features = price_features
        self.ind_features = ind_features

        # GRU replaces vanilla RNN
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        total_concat_dim = hidden_size + ind_features
        self.linear = nn.Sequential(
            nn.Linear(total_concat_dim, features_dim), 
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        priceobs = observations[:, : self.price_features].unsqueeze(-1)
        ind = observations[:, self.price_features :]
        gru_out, h_n = self.gru(priceobs)
        last_hidden = h_n[-1]
        combined = th.cat([last_hidden, ind], dim=1)
        return self.linear(combined)

############################################CNN FOR PPO/SAC ######################################ž
class Custom2dCNN(BaseFeaturesExtractor):

  def __init__(
      self,
      observation_space: spaces.Box,
      features_dim: int = 128,
      price_features: int = 100,
      vol_features: int = 100,
      ind_features: int = 4,
    ):

    super().__init__(observation_space, features_dim)

    self.price_features = price_features
    self.ind_features = ind_features
    self.vol_features = vol_features

    self.cnn = nn.Sequential(
        nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1), 
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
    )

    cnn_out_dim = 64 * price_features           #ker 2 stacked en na drugem
    total_concat_dim = cnn_out_dim + ind_features

    self.linear = nn.Sequential(
        nn.Linear(total_concat_dim, features_dim), nn.ReLU()
    )

  def forward(self, observations: th.Tensor) -> th.Tensor:
    priceobs = observations[:, : self.price_features]
    volobs = observations[:, self.price_features : self.price_features + self.vol_features]
    ind = observations[:, self.price_features + self.vol_features :] 

    cnn_comb=th.stack([priceobs, volobs], dim=1).unsqueeze(2) #doda se visino ker je moj 1d tipo za 1d cnn
    #th stack naredi nov dimension
    #th cat obdrzi

    comb = self.cnn(cnn_comb)
    combined = th.cat([comb, ind], dim=1) #da skupaj

    return self.linear(combined)

############################################DICTIONARIES ######################################ž

# The default MLP extractor takes its input size from observation_space.
policy_kwargs_def_100 = {}

policy_kwargs_cnn_100 = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(
        price_features=100, ind_features=4, vol_features=100, features_dim=128
    ),
)

policy_kwargs_rnn_100 = dict(
    features_extractor_class=CustomRNN,
    features_extractor_kwargs=dict(
        price_features=100, ind_features=104, hidden_size=64, features_dim=128
    ),
)

policy_kwargs_2dcnn_100 = dict(
    features_extractor_class=Custom2dCNN,
    features_extractor_kwargs=dict(
        price_features=100, ind_features=4, vol_features=100, features_dim=128
    ),
)

policy_kwargs_gru_100 = dict(
    features_extractor_class=CustomGRU,
    features_extractor_kwargs=dict(
        price_features=100, ind_features=104, hidden_size=64, features_dim=128
    ),
)

policy_kwargs_def_5 = {}

policy_kwargs_cnn_5 = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(
        price_features=5, ind_features=4, vol_features=5, features_dim=128
    ),
)

policy_kwargs_rnn_5 = dict(
    features_extractor_class=CustomRNN,
    features_extractor_kwargs=dict(
        price_features=5, ind_features=9, hidden_size=64, features_dim=128
    ),
)

policy_kwargs_2dcnn_5 = dict(
    features_extractor_class=Custom2dCNN,
    features_extractor_kwargs=dict(
        price_features=5, ind_features=4, vol_features=5, features_dim=128
    ),
)

policy_kwargs_gru_5 = dict(
    features_extractor_class=CustomGRU,
    features_extractor_kwargs=dict(
        price_features=5, ind_features=9, hidden_size=64, features_dim=128
    ),
)

POLICY_KWARGS_BY_LOOKBACK = {
    100: {
        "def": policy_kwargs_def_100,
        "cnn": policy_kwargs_cnn_100,
        "rnn": policy_kwargs_rnn_100,
        "2dcnn": policy_kwargs_2dcnn_100,
        "gru": policy_kwargs_gru_100,
    },
    5: {
        "def": policy_kwargs_def_5,
        "cnn": policy_kwargs_cnn_5,
        "rnn": policy_kwargs_rnn_5,
        "2dcnn": policy_kwargs_2dcnn_5,
        "gru": policy_kwargs_gru_5,
    },
}

# Preserve the existing 100-day imports.
policy_kwargs_def = policy_kwargs_def_100
policy_kwargs_cnn = policy_kwargs_cnn_100
policy_kwargs_rnn = policy_kwargs_rnn_100
policy_kwargs_2dcnn = policy_kwargs_2dcnn_100
policy_kwargs_gru = policy_kwargs_gru_100


def get_policy_kwargs(lookback, architecture):
    """Size the feature extractor for the selected observation window."""
    from copy import deepcopy
    from numbers import Integral

    if isinstance(lookback, bool) or not isinstance(lookback, Integral) or lookback <= 0:
        raise ValueError("lookback must be a positive integer")
    kwargs = deepcopy(POLICY_KWARGS_BY_LOOKBACK[100][architecture])
    if architecture != "def":
        features = kwargs["features_extractor_kwargs"]
        features["price_features"] = int(lookback)
        if architecture in ("cnn", "2dcnn"):
            features["vol_features"] = int(lookback)
        else:
            features["ind_features"] = int(lookback) + 4
    return kwargs
