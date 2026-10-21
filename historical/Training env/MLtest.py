import numpy as np
import config as C
from collections import deque
#https://scikit-learn.org/stable/modules/ensemble.html#forest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

'''
za mojo stvar rabim dat nevem 
x = [[close diff, open diff]]
'''
'''
to je linear regression, obstaja tudi gradient voost (xgboost) dober kao za tabular data
xdboost je train small shallow trees, they learn what the prev one went wrong
basically policy gradient

ltsm, gru, cnn so neural nets ampak overfitting

random forrest (many decision treesv)
yes or no questions with greedy decision
veliko dreves
avg of all trees je final prediciton
'''

'''
za vsako 
'''

class PriceCNN(nn.Module):
    def __init__(self, in_channels=5, num_classes=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)

#DATA PREPARATION
df_all = C.DATAFRAME["df1"]
scaler = MinMaxScaler()

x = []
x_sequence = []
y = []
train_flags = []

def ema_last(x, span):
    x = np.asarray(x, dtype=float)
    alpha = 2 / (span + 1)
    ema = x[0]
    for v in x[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema

def rsi_last(close, period=14):
    close = np.asarray(close, dtype=float)
    deltas = np.diff(close)                 # day-to-day price changes
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Wilder's smoothing: start with simple averages, then update recursively
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)

def prepare_forest_features(close, high, low, open_prices, volume, returns, obv):
    #TREND#
    d1ret = close[-1] / close[-2] -1 #last days difference in price
    d5ret = close[-1] / close[-6] - 1 #the difference in price in the last 0 days
    d20ret = close[-1] / close[0] - 1 #the difference in the price in the last 20 days
    intraday = close[-1] / open_prices[-1] - 1 #intra day return last day
    volume_change = volume[-1] / volume[-2] - 1
    t = np.arange(len(close))         #linear regression of previous few pointS
    slope = np.polyfit(t, np.log(close), 1)[0]      #daily log return

    #MOMENTUM
    fast = ema_last(close, 5)         #exponential moving average, biases the recent trends
    slow = ema_last(close, 20) 
    ema_spread = fast / slow - 1
    rsi = rsi_last(close, 14) / 100 - 0.5 
    MACD = fast-slow #moving avg convergence divergence how far the short-term trend is from the long-term trend
    roc10 = close[-1] / close[-11] - 1 #rate of change in last 
    roc20 = close[-1] / close[-20] - 1
    

    twentydayvol = np.std(returns) #volatility over 20 day period
    fivedayvol = np.std(np.asarray(returns)[-5:]) 

    # OBV is accumulated when each daily observation enters the window.
    volume_std = np.std(volume)
    volz = (volume[-1] - np.mean(volume)) / volume_std if volume_std > 0 else 0.0


    inside = np.array([

        ######TREND
        d1ret,
        d5ret,
        d20ret,
        intraday, #last day intra day
        ema_spread, #exponential slope, biasing the last few points
        slope, #linear regression slope   

        ##MOMENTUUM
        rsi,  #rsi, moving on avg upo or down
        MACD, 
        roc10,
        roc20,

        #VOLATILITY
        twentydayvol,
        fivedayvol,
        max(high[-1] - low[-1],abs(high[-1] - close[-2]),abs(low[-1] - close[-2])), #max rolling change per day (tr)

        #VOLUME
        obv, #major market moving depends on the volume
        volz, #how unusual the trading volume is compared to the recent volume


    ], dtype=float)
    return inside, obv

def prepare_cnn_sequence(close, high, low, open_prices, volume):
    #CNNN DATA PREPARATION##########################################
    closes = np.asarray(close, dtype=np.float32) #making arrays
    highs = np.asarray(high, dtype=np.float32)
    lows = np.asarray(low, dtype=np.float32)
    opens = np.asarray(open_prices, dtype=np.float32)
    volumes = np.asarray(volume, dtype=np.float32)

    close_returns = np.zeros(20, dtype=np.float32)
    close_returns[1:] = closes[1:] / closes[:-1] - 1

    overnight_returns = np.zeros(20, dtype=np.float32)
    overnight_returns[1:] = opens[1:] / closes[:-1] - 1

    intraday_returns = closes / opens - 1
    spreads = (highs - lows) / closes

    volume_changes = np.zeros(20, dtype=np.float32)
    volume_changes[1:] = np.log(
        (volumes[1:] + 1) / (volumes[:-1] + 1)
    )

    return np.stack([
        close_returns,
        overnight_returns,
        intraday_returns,
        spreads,
        volume_changes,
    ])


# Build examples for every ticker. Each example must contain scalar values.
tickers = df_all.columns.get_level_values("Ticker").unique()

for ticker in tickers:
    df = df_all.xs(ticker, axis=1, level="Ticker").dropna() #remove na colums

    if len(df) < 21:
        continue

    close = deque(maxlen=20)
    high = deque(maxlen=20)
    low = deque(maxlen=20)
    open_prices = deque(maxlen=20)
    volume = deque(maxlen=20)
    returns = deque(maxlen=20)
    obv = 0.0

    #na zacetku gre samo za 20 naprej in napolne arraye
    for i in range(20):
        close.append(float(df["Close"].iloc[i]))
        high.append(float(df["High"].iloc[i]))
        low.append(float(df["Low"].iloc[i]))
        open_prices.append(float(df["Open"].iloc[i]))
        volume.append(float(df["Volume"].iloc[i]))
        if (i == 0): continue
        else:
            returns.append(close[i] / close[i - 1] - 1)

        if (i == 0): continue
        else: 
            if (close[i] > close[i-1]): obv+= volume[i]
            elif (close[i] == close[i-1]): obv += 0
            else: obv -= volume[i]


    split_index = int(len(df) * 0.8)

    for i in range(20, len(df)): #od [20-260 tipo dela to]
        inside, obv = prepare_forest_features(close, high, low, open_prices, volume, returns, obv)
        sequence = prepare_cnn_sequence(close, high, low, open_prices, volume)


        x.append(inside) #sequence for the random forrest
        x_sequence.append(sequence) #sequence for the cnn

        current_close = float(df["Close"].iloc[i])

        y.append(int(current_close > close[-1])) #closing prices for both

        #updated the metrics
        train_flags.append(i < split_index)
        returns.append(current_close / close[-1] - 1)
        if current_close > close[-1]:
            obv += float(df["Volume"].iloc[i])
        elif current_close < close[-1]:
            obv -= float(df["Volume"].iloc[i])
        close.append(current_close)
        high.append(float(df["High"].iloc[i]))
        low.append(float(df["Low"].iloc[i]))
        open_prices.append(float(df["Open"].iloc[i]))
        volume.append(float(df["Volume"].iloc[i]))

#setting up the arrays for testing and training
x = np.asarray(x) #forrest
x_sequence = np.asarray(x_sequence, dtype=np.float32) #conv

y = np.asarray(y) #both
train_flags = np.asarray(train_flags)

# Chronological 80/20 split within every ticker.
#for the random fofrrest
x_train = x[train_flags]
x_test = x[~train_flags]
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

#for the conv
x_sequence_train = x_sequence[train_flags]
x_sequence_test = x_sequence[~train_flags]

#for both
y_train = y[train_flags]
y_test = y[~train_flags] #reverse boolean

#random forrest#######################################################
def forrest():

    clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    train_accuracy = accuracy_score(y_train, clf.predict(x_train))
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Choose the constant prediction using training labels only.
    majority_label = int(y_train.mean() > 0.5)
    baseline = accuracy_score(y_test, np.full_like(y_test, majority_label))
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy: {accuracy * 100:.2f}%") #accuracy of getting the prediction right on a certain day
    print(f"Training-majority baseline ({'up' if majority_label else 'not up'}): {baseline * 100:.2f}%")
    print(f"Always-up baseline: {y_test.mean() * 100:.2f}%")
    print(f"Difference from training-majority baseline: {(accuracy - baseline) * 100:+.2f} percentage points")
    print(f"Train/test examples: {len(y_train)}/{len(y_test)}")
    print("Percentage of up days:", y_test.mean())  #percentage of all up days in the dataset
    print("Predicted up percentage:", y_pred.mean()) #the percentage the model predictef

#CNNN #####################################################################
#CONVOLUTIONAL NETWORK

def conv():
    # Expected CNN shape: (samples, 5 channels, 20 days)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    #normalisation stat eq x-mean/std
    channel_mean = x_sequence_train.mean(axis=(0, 2), keepdims=True)
    channel_std = x_sequence_train.std(axis=(0, 2), keepdims=True)
    channel_std[channel_std < 1e-8] = 1.0

    x_sequence_train = (x_sequence_train - channel_mean) / channel_std
    x_sequence_test = (x_sequence_test - channel_mean) / channel_std

    # Convert NumPy arrays to PyTorch tensors
    x_train_tensor = torch.from_numpy(x_sequence_train)
    y_train_tensor = torch.from_numpy(y_train)

    x_test_tensor = torch.from_numpy(x_sequence_test)
    y_test_tensor = torch.from_numpy(y_test)

    # Create datasets
    trainset = TensorDataset(x_train_tensor, y_train_tensor)
    testset = TensorDataset(x_test_tensor, y_test_tensor)

    # Create data loaders
    trainloader = DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    testloader = DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    classes = ("down", "up")

    device = torch.device("cuda")
    net = PriceCNN(in_channels=5, num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.004)

    #trinainfg the cnn
    for epoch in range(1):
        net.train()
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            targets = labels.float().unsqueeze(1)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            predicted = (torch.sigmoid(outputs) >= 0.5).long().squeeze(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    correct = 0
    total = 0

    net.eval()

    with torch.no_grad():
        for sequences, labels in testloader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = net(sequences)
            probabilities_up = torch.sigmoid(outputs).squeeze(1)
            predicted = (probabilities_up >= 0.5).long()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Test accuracy: %.2f%%" % (100 * correct / total))

forrest()
