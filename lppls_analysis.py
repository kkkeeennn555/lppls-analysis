import yfinance as yf
df = yf.download("^N225", start="2018-01-01", end="2025-06-07")
prices = np.log(df["Close"]).values
dates  = np.arange(len(prices))
from scipy.optimize import minimize
def lppls_residuals(params, t, y, filt):
    m, omega, tc = params
    dt = tc - t
    if np.any(dt <= 0): return np.inf
    # 範囲外は罰則
    if not(filt["m_min"] < m < filt["m_max"]): return np.inf
    if not(filt["ω_min"]< omega < filt["ω_max"]): return np.inf
    # デザイン行列を作成
    x1 = dt**m
    x2 = dt**m * np.cos(omega * np.log(dt))
    x3 = dt**m * np.sin(omega * np.log(dt))
    X = np.vstack([np.ones_like(t), x1, x2, x3]).T
    # 条件数チェック
    if np.linalg.cond(X) > 1e8: 
        return np.inf
    # 線形回帰
    β, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X.dot(β)
    return np.sum(res**2)
filt = {"m_min":0.01,"m_max":0.99,"ω_min":4,"ω_max":25}
init = [0.5, 10, len(prices)+20]  # m, ω, tc の初期値
res = minimize(lppls_residuals, init, args=(dates, prices, filt),
               method="Nelder-Mead", options={"maxiter":5000})
m_hat, ω_hat, tc_hat = res.x
# β を再取得して振幅や ds 指標を計算
dt = tc_hat - dates
X = ...
β, *_ = np.linalg.lstsq(X, prices, rcond=None)
A, B, C1, C2 = β
C = np.hypot(C1, C2)
ds = abs(B) / (ω_hat * C)
damping = (m_hat * abs(B)) / (ω_hat * C)
import json
with open("results/summary.json","w") as f:
    json.dump({
      "m":m_hat, "omega":ω_hat, "tc":int(tc_hat),
      "ds":float(ds), "damping":float(damping)
    }, f, ensure_ascii=False, indent=2)
