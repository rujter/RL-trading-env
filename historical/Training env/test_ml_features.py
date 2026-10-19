import unittest
import warnings

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ml_features import build_features, prepare_dataset


def prices(n=340, flat=False):
    close = np.full(n, 100.0) if flat else 100 + np.arange(n, dtype=float)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                         "Close": close, "Volume": np.full(n, 1000.0)},
                        index=pd.bdate_range("2023-01-02", periods=n))


def panel(**tickers):
    result = pd.concat(tickers, axis=1).swaplevel(axis=1)
    result.columns.names = ["Price", "Ticker"]
    return result


class FeatureTests(unittest.TestCase):
    def test_known_trend_values(self):
        df = prices()
        f = build_features(df)
        row = f.iloc[-1]
        self.assertAlmostEqual(row.sma_20, df.Close.iloc[-20:].mean())
        self.assertAlmostEqual(row.sma_50, df.Close.iloc[-50:].mean())
        self.assertAlmostEqual(row.sma_200, df.Close.iloc[-200:].mean())
        self.assertAlmostEqual(row.linear_slope_20, 1)
        self.assertAlmostEqual(row.rsi_14, 100)
        self.assertAlmostEqual(row.atr_14, 2)
        self.assertAlmostEqual(row.obv_trend_20, 1)
        self.assertAlmostEqual(row.roc_10, (df.Close.iloc[-1] / df.Close.iloc[-11] - 1) * 100)
        self.assertAlmostEqual(row.distance_52w_high, df.Close.iloc[-1] / df.High.iloc[-252:].max() - 1)
        self.assertAlmostEqual(row.drawdown_from_peak, 0)
        self.assertTrue(f.iloc[:251].distance_52w_high.isna().all())
        self.assertTrue(np.isfinite(f.iloc[251:]).all().all())

    def test_flat_prices_zero_volume_and_range(self):
        df = prices(flat=True)
        df["High"] = df.Close
        df["Low"] = df.Close
        df["Volume"] = 0
        f = build_features(df).iloc[251:]
        self.assertTrue(np.isfinite(f).all().all())
        self.assertTrue(f.rsi_14.eq(50).all())
        for name in ("cmf_20", "volume_zscore_20", "obv_trend_20", "realised_vol_20", "atr_14"):
            self.assertTrue(f[name].eq(0).all())

    def test_future_changes_do_not_change_past_features(self):
        df = prices()
        vix = pd.Series(20.0, index=df.index)
        expected = build_features(df, vix)
        changed = df.copy()
        changed.iloc[300:] *= 2
        vix.iloc[300:] = 90
        actual = build_features(changed, vix)
        assert_frame_equal(expected.iloc[:300], actual.iloc[:300])

    def test_vix_exact_alignment(self):
        df = prices()
        vix = pd.Series([25.0], index=[df.index[300]])
        f = build_features(df, vix)
        self.assertEqual(f.vix_level.iloc[300], 25)
        self.assertEqual(f.vix_level.notna().sum(), 1)

    def test_labels_cutoff_and_vix_not_a_target(self):
        df = prices()
        df.loc[df.index[280], "Close"] -= 0.5
        data = panel(AAA=df, BBB=df.iloc[10:], **{"^VIX": prices(flat=True)})
        x_train, x_test, y_train, y_test = prepare_dataset(data)
        self.assertIn("vix_level", x_train.columns)
        self.assertEqual(set(x_train.index.get_level_values("Ticker")), {"AAA", "BBB"})
        cutoff = x_test.index.get_level_values("Date").min()
        for x, y in ((x_train, y_train), (x_test, y_test)):
            self.assertTrue(x.index.equals(y.index))
            self.assertTrue(np.isfinite(x).all().all())
            for (date, ticker), label in y.items():
                pos = df.index.get_loc(date)
                self.assertLess(pos, len(df) - 1)
                self.assertEqual(label, int(df.Close.iloc[pos + 1] > df.Close.iloc[pos]))
                if x is x_train:
                    self.assertLess(df.index[pos + 1], cutoff)

    def test_missing_vix_and_insufficient_history(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            x_train, _, _, _ = prepare_dataset(panel(AAA=prices()))
        self.assertNotIn("vix_level", x_train.columns)
        self.assertTrue(any("VIX absent" in str(w.message) for w in caught))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(ValueError, "253"):
                prepare_dataset(panel(AAA=prices(100)))


if __name__ == "__main__":
    unittest.main()
