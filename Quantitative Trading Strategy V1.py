import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass

import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# Config
# =========================================================
@dataclass
class Config:
    ticker: str = "BTC-USD"
    start_date: str = "2018-01-01"
    end_date: str = None
    interval: str = "1d"

    # primary strategy
    breakout_lookback: int = 10
    adx_threshold: float = 12
    vol_ratio_min: float = 0.50

    # trade management
    trail_atr_mult: float = 3.0
    max_holding_bars: int = 35
    min_holding_bars: int = 3
    exit_confirm_bars: int = 2

    # meta-label
    meta_label_min_return: float = 0.0
    prob_threshold: float = 0.52

    # walk-forward on events
    train_event_ratio: float = 0.65
    lookback_train_events: int = 120
    retrain_every_events: int = 10

    # risk
    max_drawdown_limit: float = 0.25
    cooldown_bars: int = 10

    # costs
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005
    leverage: float = 1.0
    annualization: int = 365

    # logistic regression
    logreg_C: float = 0.3
    random_state: int = 42  


CFG = Config()


# =========================================================
# Data Download
# =========================================================
def download_data(cfg: Config) -> pd.DataFrame:
    df = yf.download(
        tickers=cfg.ticker,
        start=cfg.start_date,
        end=cfg.end_date,
        interval=cfg.interval,
        auto_adjust=True,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError("未下载到数据，请检查 ticker / 日期 / 网络。")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).title() for c in df.columns]

    expected = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")

    df = df[expected].copy()
    df = df.loc[:, ~df.columns.duplicated()]
    df.dropna(inplace=True)
    return df


# =========================================================
# Indicators
# =========================================================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val.replace(0, np.nan)

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_val.fillna(0)


# =========================================================
# Feature Engineering
# =========================================================
def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()

    close = out["Close"].squeeze()
    high = out["High"].squeeze()
    low = out["Low"].squeeze()
    volume = out["Volume"].squeeze()

    # trend
    out["ema_20"] = close.ewm(span=20, adjust=False).mean()
    out["ema_50"] = close.ewm(span=50, adjust=False).mean()
    out["ema_200"] = close.ewm(span=200, adjust=False).mean()

    # returns / vol
    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()

    # momentum
    out["rsi_14"] = rsi(close, 14)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(close)
    out["macd_slope"] = out["macd_hist"].diff()

    # ATR / ADX
    atr_input = pd.DataFrame({"High": high, "Low": low, "Close": close})
    out["atr_14"] = atr(atr_input, 14)
    out["atr_pct"] = out["atr_14"] / close.replace(0, np.nan)
    out["adx_14"] = adx(atr_input, 14)

    # distance to trend
    out["dist_ema20"] = close / out["ema_20"] - 1.0
    out["dist_ema50"] = close / out["ema_50"] - 1.0
    out["dist_ema200"] = close / out["ema_200"] - 1.0
    out["ema50_ema200_spread"] = out["ema_50"] / out["ema_200"] - 1.0

    # breakout
    out["rolling_high_breakout"] = high.rolling(cfg.breakout_lookback).max().shift(1)
    out["breakout_long"] = (close > out["rolling_high_breakout"]).astype(int)

    # zscore
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["zscore_20"] = (close - sma20) / std20.replace(0, np.nan)

    # volume
    out["vol_ma_20"] = volume.rolling(20).mean()
    out["vol_ratio"] = volume / out["vol_ma_20"].replace(0, np.nan)
    out["vol_chg"] = volume.pct_change()

    # regime
    out["bull_regime"] = ((out["ema_50"] > out["ema_200"]) & (close > out["ema_200"])).astype(int)
    out["above_ema20"] = (close > out["ema_20"]).astype(int)

    # primary setup: 只在这些点上考虑做多
    long_setup = (
        (out["bull_regime"] == 1) &
        (out["breakout_long"] == 1) &
        (out["adx_14"] >= cfg.adx_threshold) &
        (out["vol_ratio"] >= cfg.vol_ratio_min)
    )

    # 只取 setup 从 0 -> 1 的时点，避免连续多天重复触发
    out["event"] = (long_setup & (~long_setup.shift(1).fillna(False))).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return out


FEATURE_COLS = [
    "ret_10",
    "rsi_14",
    "macd_hist",
    "atr_pct",
    "adx_14",
    "dist_ema20",
    "ema50_ema200_spread"
]


# =========================================================
# Label candidate trades using the SAME exit logic
# =========================================================
def simulate_trade_from_pos(data: pd.DataFrame, pos: int, cfg: Config):
    cost = cfg.fee_rate + cfg.slippage_rate

    entry_price = float(data.loc[pos, "Close"])
    trail_stop = entry_price - cfg.trail_atr_mult * float(data.loc[pos, "atr_14"])
    below_count = 0

    last_pos = min(pos + cfg.max_holding_bars, len(data) - 1)
    exit_pos = last_pos
    exit_reason = "TIME_EXIT"

    for j in range(pos + 1, last_pos + 1):
        close_j = float(data.loc[j, "Close"])
        low_j = float(data.loc[j, "Low"])
        atr_j = float(data.loc[j, "atr_14"])
        ema20_j = float(data.loc[j, "ema_20"])

        trail_stop = max(trail_stop, close_j - cfg.trail_atr_mult * atr_j)

        # ATR trailing stop：随时可触发
        if low_j <= trail_stop:
            exit_pos = j
            exit_reason = "ATR_TRAIL_STOP"
            break

        # 趋势退出：至少拿够 min_holding_bars 再判断
        if close_j < ema20_j:
            below_count += 1
        else:
            below_count = 0

        if (j - pos) >= cfg.min_holding_bars and below_count >= cfg.exit_confirm_bars:
            exit_pos = j
            exit_reason = "TREND_EXIT"
            break

    exit_price = float(data.loc[exit_pos, "Close"])
    trade_return = exit_price / entry_price - 1.0 - 2 * cost
    return trade_return, exit_pos, exit_reason


def build_event_dataset(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    data = df.copy().reset_index()
    date_col = data.columns[0]
    data = data.rename(columns={date_col: "date"})

    event_positions = np.where(data["event"].values == 1)[0]
    records = []

    for pos in event_positions:
        trade_ret, exit_pos, exit_reason = simulate_trade_from_pos(data, pos, cfg)

        rec = {
            "date": data.loc[pos, "date"],
            "pos": pos,
            "label": int(trade_ret > cfg.meta_label_min_return),
            "trade_return": trade_ret,
            "exit_date": data.loc[exit_pos, "date"],
            "exit_reason": exit_reason,
            "entry_price": float(data.loc[pos, "Close"]),
            "exit_price": float(data.loc[exit_pos, "Close"]),
        }

        for col in FEATURE_COLS:
            rec[col] = data.loc[pos, col]

        records.append(rec)

    events = pd.DataFrame(records)
    if events.empty:
        raise ValueError("没有生成任何候选事件，请放松 primary setup 条件。")
    return events.sort_values("date").reset_index(drop=True)


# =========================================================
# Walk-forward predict on EVENT DATASET
# =========================================================
def walk_forward_event_predict(events: pd.DataFrame, cfg: Config):
    if len(events) < 30:
        raise ValueError("事件数太少，连最基本的 walk-forward 都不够。")

    split_idx = int(len(events) * cfg.train_event_ratio)
    if split_idx < 20 or split_idx >= len(events):
        raise ValueError("train_event_ratio 导致训练/测试切分不合理。")

    pred_rows = []
    model = None
    base_rate = 0.5

    for i in range(split_idx, len(events)):
        if (model is None) or ((i - split_idx) % cfg.retrain_every_events == 0):
            train_start = max(0, i - cfg.lookback_train_events)
            train_slice = events.iloc[train_start:i].copy()

            X_train = train_slice[FEATURE_COLS]
            y_train = train_slice["label"]

            if y_train.nunique() < 2:
                model = None
                base_rate = float(y_train.mean()) if len(y_train) > 0 else 0.5
            else:
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(
                        C=cfg.logreg_C,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=cfg.random_state
                    ))
                ])
                model.fit(X_train, y_train)
                base_rate = float(y_train.mean())

        row = events.iloc[[i]].copy()

        if model is None:
            prob = base_rate
        else:
            prob = float(model.predict_proba(row[FEATURE_COLS])[0, 1])

        row["prob_take"] = prob
        row["pred_take"] = int(prob >= 0.5)
        pred_rows.append(row)

    pred_events = pd.concat(pred_rows, axis=0).reset_index(drop=True)

    # 用测试起点前最后一个窗口训练一个最终模型，供输出系数
    last_train = events.iloc[max(0, split_idx - cfg.lookback_train_events):split_idx].copy()
    last_model = None
    if last_train["label"].nunique() >= 2:
        last_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=cfg.logreg_C,
                class_weight="balanced",
                max_iter=2000,
                random_state=cfg.random_state
            ))
        ])
        last_model.fit(last_train[FEATURE_COLS], last_train["label"])

    return pred_events, last_model

# =========================================================
# Backtest primary baseline / meta-filtered
# =========================================================
def backtest_strategy(df: pd.DataFrame, cfg: Config, pred_events: pd.DataFrame = None, use_meta: bool = False):
    data = df.copy().reset_index()
    date_col = data.columns[0]
    data = data.rename(columns={date_col: "date"})

    if use_meta:
        if pred_events is None or pred_events.empty:
            raise ValueError("use_meta=True 时必须提供 pred_events。")
        test_start = pred_events["date"].min()
        event_prob_map = dict(zip(pred_events["date"], pred_events["prob_take"]))
    else:
        # primary baseline 和 meta 使用相同测试起点，方便公平比较
        if pred_events is not None and not pred_events.empty:
            test_start = pred_events["date"].min()
        else:
            test_start = data["date"].iloc[0]
        event_prob_map = {}

    data = data[data["date"] >= test_start].copy().reset_index(drop=True)

    equity = 1.0
    peak_equity = 1.0
    cooldown = 0

    position = 0
    entry_price = np.nan
    trail_stop = np.nan
    hold_bars = 0
    below_count = 0

    cost = cfg.fee_rate + cfg.slippage_rate

    equity_curve = []
    daily_ret_list = []
    pos_list = []
    dd_list = []
    trade_log = []

    prev_close = np.nan

    for i in range(len(data)):
        row = data.loc[i]
        dt = row["date"]
        close = float(row["Close"])
        low = float(row["Low"])
        atr_val = float(row["atr_14"])
        ema20 = float(row["ema_20"])

        # mark-to-market
        if i > 0 and position == 1:
            asset_ret = (close / prev_close - 1.0) * cfg.leverage
            equity *= (1.0 + asset_ret)
            daily_ret_list.append(asset_ret)
        else:
            daily_ret_list.append(0.0)

        # drawdown
        peak_equity = max(peak_equity, equity)
        drawdown = equity / peak_equity - 1.0

        if drawdown <= -cfg.max_drawdown_limit and cooldown == 0:
            if position == 1:
                equity *= (1.0 - cost)
                trade_log.append({
                    "date": dt,
                    "action": "EXIT",
                    "price": close,
                    "reason": "MAX_DD",
                    "equity": equity
                })
                position = 0
                entry_price = np.nan
                trail_stop = np.nan
                hold_bars = 0
                below_count = 0

            cooldown = cfg.cooldown_bars
            peak_equity = equity
            drawdown = 0.0

        # manage open position
        if position == 1:
            hold_bars += 1
            trail_stop = max(trail_stop, close - cfg.trail_atr_mult * atr_val)

            exit_reason = None

            if low <= trail_stop:
                exit_reason = "ATR_TRAIL_STOP"
            else:
                if close < ema20:
                    below_count += 1
                else:
                    below_count = 0

                if hold_bars >= cfg.min_holding_bars and below_count >= cfg.exit_confirm_bars:
                    exit_reason = "TREND_EXIT"
                elif hold_bars >= cfg.max_holding_bars:
                    exit_reason = "TIME_EXIT"

            if exit_reason is not None:
                equity *= (1.0 - cost)
                trade_log.append({
                    "date": dt,
                    "action": "EXIT",
                    "price": close,
                    "reason": exit_reason,
                    "equity": equity
                })
                position = 0
                entry_price = np.nan
                trail_stop = np.nan
                hold_bars = 0
                below_count = 0

        # cooldown
        if cooldown > 0:
            cooldown -= 1

        # entry
        if position == 0 and cooldown == 0 and int(row["event"]) == 1:
            allow_entry = True

            if use_meta:
                prob = event_prob_map.get(dt, np.nan)
                allow_entry = (not pd.isna(prob)) and (prob >= cfg.prob_threshold)
            else:
                prob = np.nan

            if allow_entry:
                position = 1
                entry_price = close
                trail_stop = close - cfg.trail_atr_mult * atr_val
                hold_bars = 0
                below_count = 0

                equity *= (1.0 - cost)
                trade_log.append({
                    "date": dt,
                    "action": "ENTRY",
                    "price": close,
                    "reason": f"ENTRY_prob={prob:.3f}" if use_meta else "PRIMARY_ENTRY",
                    "equity": equity
                })

        equity_curve.append(equity)
        daily_ret_list[-1] = daily_ret_list[-1] if len(daily_ret_list) > 0 else 0.0
        pos_list.append(position)
        dd_list.append(drawdown)
        prev_close = close

    result = data.copy()
    result["strategy_equity"] = equity_curve
    result["strategy_daily_ret"] = daily_ret_list
    result["position"] = pos_list
    result["drawdown"] = dd_list
    result["benchmark_equity"] = (1 + result["Close"].pct_change().fillna(0)).cumprod()

    trades = pd.DataFrame(trade_log)
    return result, trades


# =========================================================
# Reports
# =========================================================
def compute_max_drawdown(equity_series: pd.Series) -> float:
    peak = equity_series.cummax()
    dd = equity_series / peak - 1.0
    return float(dd.min())


def pair_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    entries = trades[trades["action"] == "ENTRY"].reset_index(drop=True)
    exits = trades[trades["action"] == "EXIT"].reset_index(drop=True)

    n = min(len(entries), len(exits))
    if n == 0:
        return pd.DataFrame()

    pairs = pd.DataFrame({
        "entry_date": entries.loc[:n-1, "date"].values,
        "entry_price": entries.loc[:n-1, "price"].values,
        "exit_date": exits.loc[:n-1, "date"].values,
        "exit_price": exits.loc[:n-1, "price"].values,
        "exit_reason": exits.loc[:n-1, "reason"].values,
    })
    pairs["trade_return"] = pairs["exit_price"] / pairs["entry_price"] - 1.0
    return pairs


def performance_report(bt_df: pd.DataFrame, cfg: Config, trades: pd.DataFrame) -> dict:
    daily_ret = bt_df["strategy_daily_ret"].fillna(0.0)

    total_return = bt_df["strategy_equity"].iloc[-1] - 1.0
    ann_return = bt_df["strategy_equity"].iloc[-1] ** (cfg.annualization / len(bt_df)) - 1.0
    ann_vol = daily_ret.std() * np.sqrt(cfg.annualization)
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(cfg.annualization)) if daily_ret.std() > 0 else np.nan
    max_dd = compute_max_drawdown(bt_df["strategy_equity"])

    bench_total = bt_df["benchmark_equity"].iloc[-1] - 1.0
    bench_mdd = compute_max_drawdown(bt_df["benchmark_equity"])

    paired = pair_trades(trades)
    win_rate = float(paired["trade_return"].gt(0).mean()) if not paired.empty else np.nan
    avg_trade_ret = float(paired["trade_return"].mean()) if not paired.empty else np.nan

    return {
        "Strategy Total Return": total_return,
        "Strategy Annualized Return": ann_return,
        "Strategy Annualized Volatility": ann_vol,
        "Strategy Sharpe": sharpe,
        "Strategy Max Drawdown": max_dd,
        "Benchmark Total Return": bench_total,
        "Benchmark Max Drawdown": bench_mdd,
        "Entry Count": int((trades["action"] == "ENTRY").sum()) if not trades.empty else 0,
        "Exit Count": int((trades["action"] == "EXIT").sum()) if not trades.empty else 0,
        "Average Exposure": float(bt_df["position"].mean()),
        "Trade Win Rate": win_rate,
        "Average Trade Return": avg_trade_ret
    }


def meta_report(pred_events: pd.DataFrame, cfg: Config) -> dict:
    y_true = pred_events["label"]
    y_prob = pred_events["prob_take"]
    y_pred_05 = (y_prob >= 0.5).astype(int)
    y_pred_thr = (y_prob >= cfg.prob_threshold).astype(int)

    out = {
        "Event Count": int(len(pred_events)),
        "Positive Label Rate": float(y_true.mean()),
        "Accuracy@0.5": accuracy_score(y_true, y_pred_05),
        "Precision@thr": precision_score(y_true, y_pred_thr, zero_division=0),
        "Recall@thr": recall_score(y_true, y_pred_thr, zero_division=0),
        "Accept Rate@thr": float(y_pred_thr.mean()),
        "Accepted Positive Rate": float(y_true[y_pred_thr == 1].mean()) if (y_pred_thr == 1).any() else np.nan
    }

    if y_true.nunique() > 1:
        out["ROC_AUC"] = roc_auc_score(y_true, y_prob)
    else:
        out["ROC_AUC"] = np.nan

    return out


def feature_importance_df(model, feature_cols):
    if model is None:
        return pd.DataFrame(columns=["feature", "coef", "abs_coef"])

    clf = model.named_steps["clf"]
    coef = clf.coef_[0]

    out = pd.DataFrame({
        "feature": feature_cols,
        "coef": coef,
        "abs_coef": np.abs(coef)
    }).sort_values("abs_coef", ascending=False)

    return out


def print_report_block(title: str, report: dict):
    print(f"\n{title}")
    for k, v in report.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"{k:28s}: {v:.6f}")
        else:
            print(f"{k:28s}: {v}")


# =========================================================
# Plot
# =========================================================
def plot_all(primary_bt: pd.DataFrame, meta_bt: pd.DataFrame):
    plt.figure(figsize=(13, 6))
    plt.plot(primary_bt.index, primary_bt["benchmark_equity"], label="Benchmark Equity")
    plt.plot(primary_bt.index, primary_bt["strategy_equity"], label="Primary Equity")
    plt.plot(meta_bt.index, meta_bt["strategy_equity"], label="Meta-Filtered Equity")
    plt.title("BTC-USD | Primary vs Meta-Filtered vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(13, 4))
    plt.plot(primary_bt.index, primary_bt["position"], label="Primary Position")
    plt.plot(meta_bt.index, meta_bt["position"], label="Meta Position")
    plt.title("Position Over Time")
    plt.xlabel("Date")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================================================
# Main
# =========================================================
def main(cfg: Config = CFG):
    print("=" * 95)
    print(f"Primary + Meta-Label Model | Ticker={cfg.ticker} | Interval={cfg.interval}")
    print("=" * 95)

    # 1) 数据与特征
    raw = download_data(cfg)
    feat = build_features(raw, cfg)
    events = build_event_dataset(feat, cfg)

    print(f"\n[INFO] Total events generated: {len(events)}")

    # 2) 事件太少时，自动关闭 Meta Filter
    use_meta = True
    if len(events) < 80:
        print("[WARN] 事件数小于 80，ML 过滤没有统计意义，本次自动关闭 Meta Filter。")
        use_meta = False
        pred_events = pd.DataFrame(columns=["date", "prob_take", "label"])
        last_model = None
    else:
        pred_events, last_model = walk_forward_event_predict(events, cfg)

    # 3) Meta 质量报告
    if use_meta:
        meta_stats = meta_report(pred_events, cfg)
    else:
        meta_stats = {
            "Event Count": len(events),
            "Positive Label Rate": float(events["label"].mean()) if len(events) > 0 else np.nan,
            "Accuracy@0.5": np.nan,
            "Precision@thr": np.nan,
            "Recall@thr": np.nan,
            "Accept Rate@thr": np.nan,
            "Accepted Positive Rate": np.nan,
            "ROC_AUC": np.nan
        }

    # 4) Primary baseline 回测
    primary_bt, primary_trades = backtest_strategy(
        feat, cfg, pred_events=pred_events if use_meta else None, use_meta=False
    )
    primary_perf = performance_report(primary_bt, cfg, primary_trades)

    # 5) Meta-filtered 回测
    if use_meta:
        meta_bt, meta_trades = backtest_strategy(
            feat, cfg, pred_events=pred_events, use_meta=True
        )
        meta_perf = performance_report(meta_bt, cfg, meta_trades)
    else:
        meta_bt = primary_bt.copy()
        meta_trades = pd.DataFrame()
        meta_perf = {
            "Info": "Meta Filter disabled due to low event count."
        }

    # 6) 特征系数 / 重要性
    fi = feature_importance_df(last_model, FEATURE_COLS)

    # 7) 打印报告
    print_report_block("[1] Meta Label Quality", meta_stats)
    print_report_block("[2] Primary Baseline Performance", primary_perf)
    print_report_block("[3] Meta-Filtered Performance", meta_perf)

    print("\n[4] Top Feature Importances / Coefficients")
    if fi.empty:
        print("No feature importance available.")
    else:
        print(fi.head(12).to_string(index=False))

    print("\n[5] Recent Meta Trades")
    if meta_trades.empty:
        print("No meta-filtered trades generated.")
    else:
        print(meta_trades.tail(12).to_string(index=False))

    paired_meta = pair_trades(meta_trades)
    print("\n[6] Closed Meta Trades")
    if paired_meta.empty:
        print("No closed meta trades.")
    else:
        print(paired_meta.tail(10).to_string(index=False))

    # 8) 画图
    plot_all(primary_bt, meta_bt)

    return {
        "raw_data": raw,
        "feature_data": feat,
        "event_data": events,
        "pred_events": pred_events,
        "primary_bt": primary_bt,
        "primary_trades": primary_trades,
        "meta_bt": meta_bt,
        "meta_trades": meta_trades,
        "meta_stats": meta_stats,
        "primary_perf": primary_perf,
        "meta_perf": meta_perf,
        "feature_importance": fi,
    }


if __name__ == "__main__":
    results = main(CFG)