import contextlib
import io
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'Quantitative Trading Strategy V1.py'))
Config = M['Config']


def prices(n=5):
    return pd.DataFrame({'Open': 100., 'High': 101., 'Low': 99., 'Close': 100.,
                         'Volume': 1000., 'atr_14': 1., 'ema_20': 100.,
                         'event': [1] + [0] * (n - 1)}, index=pd.date_range('2020-01-01', periods=n))


class StrategyTests(unittest.TestCase):
    def test_rsi_one_sided_and_flat(self):
        for values, expected in [(range(20), 100), (range(20, 0, -1), 0), ([10] * 20, 50)]:
            self.assertEqual(M['rsi'](pd.Series(values)).iloc[-1], expected)

    def test_adx_equal_directional_moves_are_neutral(self):
        df = pd.DataFrame({'High': [10, 12, 14], 'Low': [8, 6, 4], 'Close': [9, 9, 9]})
        self.assertTrue((M['adx'](df) == 0).all())

    def test_config_rejects_invalid_values(self):
        for kw in [{'retrain_every_events': 0}, {'train_event_ratio': 1}, {'fee_rate': -1}, {'leverage': float('nan')}, {'max_holding_bars': 1}]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                Config(**kw)

    def test_costs_reconcile_equity_and_trade_returns(self):
        cfg = Config(fee_rate=.01, slippage_rate=0, max_holding_bars=3)
        result, trades = M['backtest_strategy'](prices(), cfg)
        self.assertAlmostEqual(result.strategy_equity.iloc[-1], .99 ** 2)
        np.testing.assert_allclose((1 + result.strategy_daily_ret).cumprod(), result.strategy_equity)
        self.assertAlmostEqual(M['pair_trades'](trades).trade_return.iloc[0], .99 ** 2 - 1)
        self.assertAlmostEqual(result.drawdown.iloc[-1], .99 ** 2 - 1)
        self.assertAlmostEqual(M['compute_max_drawdown'](result.strategy_equity), .99 ** 2 - 1)

    def test_labels_and_backtest_share_net_return(self):
        cfg = Config(fee_rate=.01, slippage_rate=.001, leverage=1.5, max_holding_bars=3)
        df = prices()
        df['Close'] = [100, 101, 102, 103, 103]
        df['High'] = df.Close + 1
        df['Low'] = df.Close - .1
        expected, _, _ = M['simulate_trade_from_pos'](df.reset_index(drop=True), 0, cfg)
        _, trades = M['backtest_strategy'](df, cfg)
        self.assertAlmostEqual(M['pair_trades'](trades).trade_return.iloc[0], expected)

    def test_stop_does_not_use_same_bar_future_close(self):
        cfg = Config(max_holding_bars=3)
        df = prices()
        df['Close'] = [100, 110, 111, 112, 113]
        df['Low'] = [99, 99, 110, 111, 112]
        df['High'] = df.Close + 1
        _, exit_pos, reason = M['simulate_trade_from_pos'](df.reset_index(drop=True), 0, cfg)
        self.assertEqual((exit_pos, reason), (3, 'TIME_EXIT'))
        _, trades = M['backtest_strategy'](df, cfg)
        self.assertEqual(trades.iloc[1]['reason'], 'TIME_EXIT')

    def test_censored_events_are_not_labeled(self):
        df = prices(2)
        for col in M['FEATURE_COLS']:
            df[col] = 1.
        with self.assertRaises(ValueError):
            M['build_event_dataset'](df, Config())

    def test_purge_unresolved_labels(self):
        dates = pd.date_range('2020-01-01', periods=40)
        events = pd.DataFrame({'date': dates, 'exit_date': dates + pd.Timedelta(days=5), 'label': 0})
        for col in M['FEATURE_COLS']:
            events[col] = np.linspace(0, 1, len(events))
        cfg = Config(train_event_ratio=.75)
        eligible = M['resolved_training_events'](events, 30, cfg)
        self.assertTrue((eligible.exit_date < events.iloc[30].date).all())
        a, _ = M['walk_forward_event_predict'](events, cfg)
        events.loc[25:, 'label'] = 1  # These outcomes were unavailable at the first prediction.
        b, _ = M['walk_forward_event_predict'](events, cfg)
        self.assertEqual(a.iloc[0].prob_take, b.iloc[0].prob_take)
        self.assertEqual(a.iloc[0].prob_take, 0.)

    def test_final_position_is_closed(self):
        result, trades = M['backtest_strategy'](prices(2), Config())
        self.assertEqual(result.position.iloc[-1], 0)
        self.assertEqual(trades.iloc[-1]['reason'], 'END_OF_DATA')

    def test_multiclass_walkforward_and_meta_backtest(self):
        dates = pd.date_range('2020-01-01', periods=100)
        events = pd.DataFrame({'date': dates, 'exit_date': dates + pd.Timedelta(days=1),
                               'label': np.arange(100) % 2})
        rng = np.random.default_rng(8)
        for col in M['FEATURE_COLS']:
            events[col] = rng.normal(size=100)
        cfg = Config()
        predictions, model = M['walk_forward_event_predict'](events, cfg)
        self.assertIsNotNone(model)
        self.assertTrue(predictions.prob_take.between(0, 1).all())
        df = prices(100)
        df['event'] = 1
        result, _ = M['backtest_strategy'](df, cfg, predictions, use_meta=True)
        self.assertEqual(result.date.iloc[0], predictions.date.min())
        np.testing.assert_allclose((1 + result.strategy_daily_ret).cumprod(), result.strategy_equity)

    def test_cooldown_blocks_trigger_bar_reentry(self):
        df = prices(5)
        df['Close'] = [100, 70, 70, 70, 70]
        df['Low'] = df.Close - 1
        df['High'] = df.Close + 1
        df['event'] = 1
        _, trades = M['backtest_strategy'](df, Config(cooldown_bars=2))
        entries = trades[trades.action == 'ENTRY']['date'].tolist()
        self.assertEqual(entries, [df.index[0], df.index[3]])

    def test_empty_inputs_and_nonchronological_prices(self):
        with self.assertRaises(ValueError):
            M['build_features'](prices(1), Config())
        with self.assertRaises(ValueError):
            M['build_features'](prices().iloc[::-1], Config())
        with self.assertRaises(ValueError):
            M['performance_report'](pd.DataFrame(), Config(), pd.DataFrame())

    def test_download_failure(self):
        with patch.object(M['yf'], 'download', return_value=pd.DataFrame()), self.assertRaises(ValueError):
            M['download_data'](Config())

    def test_feature_prefix_does_not_depend_on_future(self):
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(.002, .015, 400)))
        df = pd.DataFrame({'Open': close, 'Close': close, 'High': close * 1.005,
                           'Low': close * .995, 'Volume': rng.uniform(100, 200, 400)},
                          index=pd.date_range('2020-01-01', periods=400))
        cfg = Config()
        whole = M['build_features'](df, cfg)
        prefix = M['build_features'](df.iloc[:250], cfg)
        pd.testing.assert_frame_equal(prefix, whole.loc[prefix.index])
        with patch.object(M['yf'], 'download', return_value=df), patch.object(M['plt'], 'show'), contextlib.redirect_stdout(io.StringIO()):
            output = M['main'](cfg)
        self.assertFalse(output['primary_bt'].empty)
        self.assertTrue(np.isfinite(output['primary_bt'].strategy_equity).all())


if __name__ == '__main__':
    unittest.main()
