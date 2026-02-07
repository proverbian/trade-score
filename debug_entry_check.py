import pandas as pd

from backtest import fetch_fx, atr_hilo
from app import s_r


def check(pair: str, entry_time_s: str, level: float, pivot_window: int = 2):
    entry_time = pd.Timestamp(entry_time_s)
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize('UTC')

    h1 = fetch_fx(pair, '60m', '30d')
    m5 = fetch_fx(pair, '5m', '7d')

    print('PAIR:', pair)
    print('H1 tz:', getattr(h1.index, 'tz', None), 'rows:', len(h1))
    print('M5 tz:', getattr(m5.index, 'tz', None), 'rows:', len(m5))

    if h1.empty or m5.empty:
        print('Missing data')
        return

    # Find nearest M5 candle at/just before entry_time
    m5_upto = m5.loc[m5.index <= entry_time]
    if m5_upto.empty:
        print('No M5 bars <= entry_time')
        return

    bar_t = m5_upto.index[-1]
    bar = m5_upto.iloc[-1]

    close_h1 = h1['Close']
    pivots = s_r.find_pivots(close_h1, window=pivot_window)
    pv = min(pivots, key=lambda p: abs(float(p['price']) - float(level)))
    pv_i = int(pv['i'])
    atr = atr_hilo(h1, 14)
    a = float(atr.iloc[pv_i])
    tol = 0.15 * a

    print('\nENTRY CHECK')
    print('entry_time (from CSV):', entry_time)
    print('m5 bar used (<= entry_time):', bar_t)
    print('m5 OHLC:', {k: float(bar[k]) for k in ['Open', 'High', 'Low', 'Close']})
    print('level:', float(level))
    print('pivot matched:', pv, 'pivot_time:', h1.index[pv_i])
    print('ATR@pivot:', a, 'tol:', tol)
    print('SELL retest condition: High >= level - tol AND Close <= level AND Close<=Open')
    print('High >= level - tol ?', float(bar['High']) >= (float(level) - tol), 'high=', float(bar['High']), 'need>=', (float(level) - tol))
    print('Close <= level ?', float(bar['Close']) <= float(level), 'close=', float(bar['Close']))
    print('Bearish candle ?', float(bar['Close']) <= float(bar['Open']))

    # Day ranges to compare with charts
    utc_day0 = pd.Timestamp(entry_time.date()).tz_localize('UTC')
    day_m5 = m5.loc[(m5.index >= utc_day0) & (m5.index < utc_day0 + pd.Timedelta(days=1))]
    print('\nRANGE CHECK (entry day UTC)')
    if not day_m5.empty:
        print('max High:', float(day_m5['High'].max()), 'min Low:', float(day_m5['Low'].min()))

    win = m5.loc[(m5.index >= entry_time - pd.Timedelta(hours=6)) & (m5.index <= entry_time + pd.Timedelta(hours=6))]
    print('\nRANGE AROUND ENTRY (±6h)')
    if not win.empty:
        print('max High:', float(win['High'].max()), 'min Low:', float(win['Low'].min()))

        win2 = win.copy()
        win2['dist'] = (win2['High'].astype(float) - float(level)).abs()
        closest = win2.sort_values('dist').head(10)[['Open', 'High', 'Low', 'Close', 'dist']]
        print('\n10 closest-by-high candles (±6h):')
        print(closest.to_string())


if __name__ == '__main__':
    # Example row from weekly_trades.csv
    check('EURUSD', '2026-02-03 07:15:00+00:00', 1.1823126077651978)
