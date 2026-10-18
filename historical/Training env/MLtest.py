import numpy as np
import config as C
from collections import deque
#https://scikit-learn.org/stable/modules/ensemble.html#forest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

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

df_all = C.DATAFRAME["df1"]
scaler = MinMaxScaler()

def price_prediction():
    x = []
    y = []
    train_flags = []

    # Build examples for every ticker. Each example must contain scalar values.
    tickers = df_all.columns.get_level_values("Ticker").unique()

    for ticker in tickers:
        df = df_all.xs(ticker, axis=1, level="Ticker").dropna()

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


        split_index = int(len(df) * 0.8)

        for i in range(20, len(df)):
            d1ret = close[-1] / close[-2] - 1
            d5ret = close[-1] / close[-6] - 1
            d20ret = close[-1] / close[0] - 1

            previous_d1ret = close[-2] / close[-3] - 1
            d2ret = d1ret - previous_d1ret

            overnight = open_prices[-1] / close[-2] - 1
            intraday = close[-1] / open_prices[-1] - 1
            spread = (high[-1] - low[-1]) / close[-1]
            volume_change = volume[-1] / volume[-2] - 1

            inside = np.array([
                d1ret,
                d5ret,
                d20ret,
                d2ret,
                overnight,
                intraday,
                spread,
            ], dtype=float)

            current_close = float(df["Close"].iloc[i])

            x.append(inside)
            y.append(int(current_close > close[-1]))
            train_flags.append(i < split_index)

            close.append(current_close)
            high.append(float(df["High"].iloc[i]))
            low.append(float(df["Low"].iloc[i]))
            open_prices.append(float(df["Open"].iloc[i]))
            volume.append(float(df["Volume"].iloc[i]))

    x = np.asarray(x)
    y = np.asarray(y)
    train_flags = np.asarray(train_flags)

    # Chronological 80/20 split within every ticker.
    x_train = x[train_flags]
    x_test = x[~train_flags]
    y_train = y[train_flags]
    y_test = y[~train_flags] #reverse boolean

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Percentage of up days:", y_test.mean())
    print("Predicted up percentage:", y_pred.mean())

price_prediction()