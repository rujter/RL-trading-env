# ML indicators

Run from this folder with NumPy, pandas and scikit-learn installed:

```powershell
python MLtest.py
python MLtest.py --vix-csv vix.csv
python MLtest.py --model cnn
```

The default trains a reproducible random forest and prints accuracy and feature
importances. The CNN keeps its five original sequence channels and requires
PyTorch; `conv()` now receives its arrays as arguments. Imports do not train models.

To use another classifier:

```python
from MLtest import prepare_dataset
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = prepare_dataset()
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(X_train.columns.tolist())
```

`prepare_dataset(df_all, vix=series)` accepts the existing yfinance-style
MultiIndex frame. `build_features(single_ticker_ohlcv, vix=series)` returns
indicators alone, including the latest unlabeled date for inference. Dataframes
retain feature names; use `.to_numpy()` if an estimator needs arrays. Trees need
no scaling; for other estimators, fit any scaler on training data only.

Features include the seven original price-action inputs plus SMA 20/50/200,
relative SMA distances, EMA 20/50 and 50/200 gaps/states/crossover events,
20-session regression slope, Wilder RSI 14, MACD 12/26 with signal 9 and histogram,
ROC 10/20, annualized 20-session realized volatility, Wilder ATR 14,
Bollinger width 20 (two standard deviations each side), OBV slope 20 normalized
by mean volume, volume Z-score 20, CMF 20, 252-session high/low proximity,
and drawdown from the highest close in the supplied history. Price-unit
SMA/slope/MACD/ATR features also have relative versions for pooled tickers.
See `build_features()` for units and conventions.

VIX is included automatically when the cache contains `^VIX`. Otherwise supply
a dated VIX close Series or a CSV with `Date,Close` columns. The existing cache
contains no VIX: the code warns and omits that column rather than inventing it.
Provided VIX is matched by exact date; missing dates are excluded from training.

Features on day t use prices through its close; labels indicate whether the next
observed session closes higher. These are after-close predictions. Warmup needs
252 valid sessions, and the last row has no label. Complete dates are split 80/20
with one cutoff across all tickers. Training rows whose labels reach the test
period are purged. The CNN retains its shorter 20-session lookback.

The cache's historical adjusted prices and downloader-filled values are used as
provided; the feature code does not reconstruct point-in-time raw market data.

Validation: `python -m unittest test_ml_features.py`.

Indicator reference: [Fidelity indicator guide](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/overview).
Estimator reference: [scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
