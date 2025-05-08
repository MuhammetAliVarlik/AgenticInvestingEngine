from langchain_core.tools import tool
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import joblib
import ta
import os
import numpy as np

MODEL_DIR = "models"
# Özellik sütunlarını modül düzeyinde tanımlıyoruz
FEATURE_COLS = [
    'Close', 'ma_5', 'returns', 'rsi', 'ema34', 'ema89',
    'macd', 'macd_signal', 'macd_diff',
    'bb_high', 'bb_low', 'bb_mid', 'bb_pct',
    'atr', 'adx', 'adx_pos', 'adx_neg',
    'obv', 'stoch_k', 'stoch_d'
]

def get_model_path(symbol: str) -> str:
    return os.path.join(MODEL_DIR, f"{symbol.upper()}_rsi_model.joblib")
def calculate_rsi_fibo_levels(df, rsi_col='rsi', lookback=50):
    
    df = df.copy()
    df['RSI_-1'] = df[rsi_col].shift(1)
    df['RSI_-2'] = df[rsi_col].shift(2)
    df['XX'] = df['RSI_-1']
    df['YY'] = df['RSI_-1']
    df['XXY'] = df['RSI_-1']

    df['MXY'] = np.where(df['RSI_-1'] > df['RSI_-2'], df['YY'],
                  np.where(df['RSI_-1'] < df['RSI_-2'], df['XX'], df['XXY']))

    df['TEPE'] = df['MXY'].rolling(window=lookback).max()
    df['DIP'] = df['MXY'].rolling(window=lookback).min()

    df['M236'] = df['DIP'] + ((df['TEPE'] - df['DIP']) * 0.236)
    df['M386'] = df['DIP'] + ((df['TEPE'] - df['DIP']) * 0.386)
    df['M500'] = df['DIP'] + ((df['TEPE'] - df['DIP']) * 0.500)
    df['M618'] = df['DIP'] + ((df['TEPE'] - df['DIP']) * 0.618)
    df['M786'] = df['DIP'] + ((df['TEPE'] - df['DIP']) * 0.786)

    return df

def train_model(
    ticker: str,
    interval: str = "1d",
    period: str = "180d"
):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Veri çekimi
    df = yf.download(ticker, interval=interval, period=period)[['Open', 'High', 'Low', 'Close', 'Volume']]
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    close=df["Close"].squeeze()
    high=df["High"].squeeze()
    low=df["Low"].squeeze()
    volume=df["Volume"].squeeze()
    # Temel teknik indikatörler
    df['rsi'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df['ema34'] = ta.trend.EMAIndicator(close, window=34).ema_indicator()
    df['ema89'] = ta.trend.EMAIndicator(close, window=89).ema_indicator()
    df['ma_5'] = close.rolling(5).mean()
    df['returns'] = close.pct_change()

    # Ek indikatörler
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_pct'] = (close - df['bb_low']) / (df['bb_high'] - df['bb_low'])

    df['atr'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()

    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Hedef sütun
    df['rsi_target'] = df['rsi'].shift(-1)
    df.dropna(inplace=True)
    if df.shape[0] < len(FEATURE_COLS):
        raise ValueError("Not enough data after cleaning")

    # Özellik matrisi ve hedef
    X = df[FEATURE_COLS]
    y = df['rsi_target']

    # Model eğitimi ve kaydetme
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    model.fit(X, y)
    joblib.dump(model, get_model_path(ticker))

    return model, df


@tool
def rsi_predictor(ticker: str) -> dict:
    """
    Predicts the next-period RSI and performs a comprehensive technical analysis.

    Features:
    - Close: last traded price.
    - ma_5: 5-period simple moving average.
    - returns: period-over-period price change.
    - RSI: relative strength index (predicted next period).
    - EMA34: 34-period exponential moving average.
    - EMA89: 89-period exponential moving average.
    - MACD: EMA12 - EMA26 momentum.
    - MACD_Signal: 9-period EMA of MACD.
    - MACD_Diff: MACD - MACD_Signal histogram.
    - BB_High: upper Bollinger Band (20,2).
    - BB_Low: lower Bollinger Band (20,2).
    - BB_Mid: middle Bollinger Band (20,2).
    - BB_Pct: position in Bollinger Bands (%B).
    - ATR: 14-period average true range.
    - ADX: average directional index (14).
    - ADX_Pos: positive directional indicator (+DI).
    - ADX_Neg: negative directional indicator (-DI).
    - OBV: on-balance volume.
    - Stoch_K: stochastic oscillator %K.
    - Stoch_D: stochastic oscillator %D.
    - price_above_ema34: price vs. EMA34 boolean.
    - rsi_fibo_divergence: RSI vs. price divergence flag.
    - channel_position: bottom/middle/top in 20-period range.

    Parameters:
        ticker (str): Stock ticker symbol.

    Returns:
        dict: {
            symbol, predicted_next_rsi, signal,
            price_above_ema34, rsi_fibo_divergence, channel_position,
            ema34, ema89,
            macd, macd_signal, macd_diff,
            bb_high, bb_low, bb_mid, bb_pct,
            atr, adx, adx_pos, adx_neg,
            obv, stoch_k, stoch_d
        }
    """
    try:
        model, df = train_model(ticker)
        df = calculate_rsi_fibo_levels(df)
        latest = df.iloc[-1]
        feat = df.loc[latest.name, FEATURE_COLS].values.reshape(1, -1)
        pred_rsi = model.predict(feat)[0]

        signal = (
            'bullish' if pred_rsi < 30 else
            'bearish' if pred_rsi > 70 else
            'neutral'
        )

        price_above_ema34 = latest['Close'] > df['ema34'].iloc[-1]
        rsi_diff = df['rsi'].diff().iloc[-5:]
        price_diff = df['Close'].diff().iloc[-5:].squeeze()
        rsi_fibo_divergence = "Yes" if (rsi_diff.mean() > 0 and price_diff.mean() < 0) or (rsi_diff.mean() < 0 and price_diff.mean() > 0) else "No"

        recent_lows = df['Low'].rolling(window=20).min().iloc[-1].squeeze()
        recent_highs = df['High'].rolling(window=20).max().iloc[-1].squeeze()
        channel_position = "bottom" if latest['Close'].squeeze() < (recent_lows + (recent_highs - recent_lows) * 0.25) else (
                            "top" if latest['Close'].squeeze() > (recent_lows + (recent_highs - recent_lows) * 0.75) else "middle")

        results = {
            'symbol': ticker.upper(),
            'predicted_next_rsi': round(pred_rsi, 2),
            'signal': signal,
            'ema34': round(latest['ema34'].squeeze(), 2),
            'ema89': round(latest['ema89'].squeeze(), 2),
            "price_above_ema34": bool(price_above_ema34.squeeze()),
            "rsi_fibo_divergence": rsi_fibo_divergence,
            "channel_position": channel_position,
            'macd': round(latest['macd'].squeeze(), 4),
            'macd_signal': round(latest['macd_signal'].squeeze(), 4),
            'bb_pct': round(latest['bb_pct'].squeeze(), 4),
            'atr': round(latest['atr'].squeeze(), 4),
            'adx': round(latest['adx'].squeeze(), 2),
            'obv': int(latest['obv'].squeeze()),
            'stoch_k': round(latest['stoch_k'].squeeze(), 2),
            'stoch_d': round(latest['stoch_d'].squeeze(), 2),
            'rsi_fib_236': round(latest['M236'].squeeze(), 2),
            'rsi_fib_386': round(latest['M386'].squeeze(), 2),
            'rsi_fib_500': round(latest['M500'].squeeze(), 2),
            'rsi_fib_618': round(latest['M618'].squeeze(), 2),
            'rsi_fib_786': round(latest['M786'].squeeze(), 2),
            'rsi_range_high': round(latest['TEPE'].squeeze(), 2),
            'rsi_range_low': round(latest['DIP'].squeeze(), 2)
        }
        return results

    except Exception as e:
        return {'error': str(e)}
