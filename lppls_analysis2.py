import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy import stats
from datetime import datetime

# 1. データ取得 --------------------------------------------------------------
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df['logp'] = np.log(df['Close'])
    df['t'] = (df.index - df.index[0]).days
    return df[['t', 'logp']]

# 2. LPPLS RSS 関数 --------------------------------------------------------
def lppls_rss(params, t, y, filt):
    m, omega, tc = params
    dt = tc - t
    if np.any(dt <= 0):
        return np.inf
    # フィルタ条件
    if not (filt['m'][0] < m < filt['m'][1] and filt['omega'][0] < omega < filt['omega'][1]):
        return np.inf
    # デザイン行列
    x1 = dt**m
    x2 = x1 * np.cos(omega * np.log(dt))
    x3 = x1 * np.sin(omega * np.log(dt))
    X = np.vstack([np.ones_like(t), x1, x2, x3]).T
    # 線形回帰で betas を解く
    try:
        betas, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.inf
    residuals = y - X.dot(betas)
    return np.sum(residuals**2)

# 3. LPPLS フィット ---------------------------------------------------------
def fit_lppls(t, y, filt):
    # 初期値
    init = [np.mean(filt['m']), np.mean(filt['omega']), t.max() + 10]
    bounds = [(filt['m'][0], filt['m'][1]), (filt['omega'][0], filt['omega'][1]), (t.max(), t.max() + 100)]
    res = minimize(lppls_rss, x0=init, args=(t, y, filt), bounds=bounds, method='L-BFGS-B')
    if not res.success:
        raise RuntimeError("LPPLS fit failed")
    m, omega, tc = res.x
    # 最終ベータ再計算
    dt = tc - t
    X = np.vstack([np.ones_like(t), dt**m, dt**m*np.cos(omega*np.log(dt)), dt**m*np.sin(omega*np.log(dt))]).T
    betas, *_ = np.linalg.lstsq(X, y, rcond=None)
    return dict(m=m, omega=omega, tc=tc, betas=betas, rss=res.fun)

# 4. ローリング予測 --------------------------------------------------------
def rolling_lppls(df, filt, window_min=125, window_max=750):
    results = []
    for w in range(window_min, window_max+1):
        if len(df) < w:
            break
        sub = df.iloc[-w:]
        t = sub['t'].values
        y = sub['logp'].values
        try:
            fit = fit_lppls(t, y, filt)
            results.append((w, fit))
        except RuntimeError:
            continue
    return results

# 5. メイン ---------------------------------------------------------------
def main():
    # 設定
    ticker = '^N225'
    start = '2018-01-01'
    end = datetime.today().strftime('%Y-%m-%d')
    filt = {'m': (0.01, 0.99), 'omega': (4, 25)}

    # データ取得
    df = fetch_data(ticker, start, end)

    # ローリングLPPLS
    res = rolling_lppls(df, filt)

    # 最新ウィンドウのtcを表示
    if res:
        w, fit = res[-1]
        tc_date = df.index[0] + pd.Timedelta(days=int(fit['tc']))
        print(f"Window: {w}, Predicted tc: {tc_date.date()}, m={fit['m']:.4f}, omega={fit['omega']:.2f}")

if __name__ == '__main__':
    main()
