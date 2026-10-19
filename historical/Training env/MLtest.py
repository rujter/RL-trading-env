"""Run a random forest on technical indicators, or the existing CNN.

Import prepare_dataset() for named pandas X_train/X_test and aligned labels.
No data loading or training happens on import. See ML_FEATURES.md for usage.
"""

import argparse
from collections import deque

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from ml_features import build_features, load_prices, prepare_dataset

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

def forrest(df_all=None, vix=None, model=None):
    """Fit any sklearn-compatible classifier; return it and the data split."""
    x_train, x_test, y_train, y_test = prepare_dataset(df_all, vix)
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"Features: {x_train.shape[1]}; train: {len(y_train)}; test: {len(y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Percentage of up days:", y_test.mean())
    print("Predicted up percentage:", y_pred.mean())
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=x_train.columns)
        print("Top feature importances:\n", importance.sort_values(ascending=False).head(10))
    return model, (x_train, x_test, y_train, y_test)


# Retain the earlier entry point as well.
price_prediction = forrest


def prepare_sequences(df_all):
    """Keep the original five-channel, 20-session CNN inputs."""
    df_all = df_all.copy()
    df_all.index = pd.to_datetime(df_all.index)
    df_all = df_all.sort_index()
    if len(df_all) < 22:
        raise ValueError("Need more history for a CNN train/test split.")
    split_date = df_all.index[int(len(df_all) * 0.8)]
    x_sequence = []
    y = []
    train_flags = []

    # Build examples for every ticker. Each example must contain scalar values.
    tickers = df_all.columns.get_level_values("Ticker").unique()
    for ticker in tickers:
        if ticker == "^VIX":
            continue
        df = df_all.xs(ticker, axis=1, level="Ticker").dropna() #remove na colums

        df = df.loc[df[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)]

        if len(df) < 21:
            continue

        close = deque(maxlen=20)
        high = deque(maxlen=20)
        low = deque(maxlen=20)
        open_prices = deque(maxlen=20)
        volume = deque(maxlen=20)

        for i in range(20):
            close.append(float(df["Close"].iloc[i]))
            high.append(float(df["High"].iloc[i]))
            low.append(float(df["Low"].iloc[i]))
            open_prices.append(float(df["Open"].iloc[i]))
            volume.append(float(df["Volume"].iloc[i]))



        for i in range(20, len(df)):
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

            sequence = np.stack([
                close_returns,
                overnight_returns,
                intraday_returns,
                spreads,
                volume_changes,
            ])

            current_close = float(df["Close"].iloc[i])

            x_sequence.append(sequence)
            y.append(int(current_close > close[-1]))
            train_flags.append(df.index[i] < split_date)

            close.append(current_close)
            high.append(float(df["High"].iloc[i]))
            low.append(float(df["Low"].iloc[i]))
            open_prices.append(float(df["Open"].iloc[i]))
            volume.append(float(df["Volume"].iloc[i]))

    if not x_sequence:
        raise ValueError("Need at least 21 valid sessions for CNN sequences.")
    x_sequence = np.asarray(x_sequence, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    train_flags = np.asarray(train_flags, dtype=bool)
    if not train_flags.any() or train_flags.all():
        raise ValueError("CNN train/test split is empty.")
    return (x_sequence[train_flags], x_sequence[~train_flags],
            y[train_flags], y[~train_flags])


def conv(x_sequence_train, x_sequence_test, y_train, y_test, epochs=20):
    """Train the CNN; import PyTorch only when requested."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PriceCNN(in_channels=5, num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.004)

    #trinainfg the cnn
    for epoch in range(epochs):
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
    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("forest", "cnn"), default="forest")
    parser.add_argument("--cache", help="yfinance MultiIndex CSV; defaults to adjacent cache.csv")
    parser.add_argument("--vix-csv", help="Optional CSV with Date and Close columns for VIX")
    args = parser.parse_args()
    df_all = load_prices(args.cache)
    if args.model == "cnn":
        conv(*prepare_sequences(df_all))
    else:
        vix = None
        if args.vix_csv:
            vix = pd.read_csv(args.vix_csv, index_col="Date", parse_dates=True)["Close"]
        forrest(df_all, vix)


if __name__ == "__main__":
    main()
