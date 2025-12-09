# --- Cell P1 (robust): build V_price and blended V_time_price ---
import pandas as pd, numpy as np, os, re

DATA_DIR = "/content/drive/MyDrive/retailrocket"   # change if needed

def infer_ts_unit_series(s: pd.Series) -> str:
    """Robust unit detector; defaults to 'ms' (RetailRocket events)."""
    if s is None or len(s) == 0:
        return "ms"
    s_num = pd.to_numeric(s, errors="coerce")
    mx = s_num.max()
    if pd.isna(mx):
        return "ms"
    return "ms" if mx > 1e11 else "s"

# 1) Load properties (only needed columns)
cols = ["timestamp","itemid","property","value"]
p1 = pd.read_csv(os.path.join(DATA_DIR, "item_properties_part1.csv"), usecols=cols)
p2 = pd.read_csv(os.path.join(DATA_DIR, "item_properties_part2.csv"), usecols=cols)
props = pd.concat([p1, p2], ignore_index=True)

# 2) Keep rows that look like price/discount info (wider net than just "price")
pat = re.compile(r"(price|oldprice|current|cost|discount|sale)", flags=re.IGNORECASE)
mask_priceish = props["property"].astype(str).str.contains(pat, na=False)
props = props[mask_priceish].copy()

# 3) Extract numeric price from messy strings (e.g., "USD 1,299.00", "1299,00 ₽")
val_str = props["value"].astype(str)
# take the FIRST decimal-like token: "-?\d+(?:[.,]\d+)?"
num_tok = val_str.str.extract(r"(-?\d+(?:[.,]\d+)?)", expand=False)
num_tok = num_tok.str.replace(",", ".", regex=False)
props["price"] = pd.to_numeric(num_tok, errors="coerce")

# 4) Drop rows with missing critical fields
props = props.dropna(subset=["price","timestamp","itemid"])

# If nothing left, make a neutral V_price and the blends, then exit this cell gracefully
if len(props) == 0:
    print("[P1] No usable price rows found. Using neutral V_price=0.")
    V_price = pd.DataFrame(0.0, index=p_time.index, columns=p_time.columns)
    def blend_potentials(V_pop: pd.DataFrame, V_price: pd.DataFrame, w: float):
        Vp = (1.0 - w)*V_pop + w*V_price
        # center by day so free-energy comparisons aren't dominated by offsets
        return Vp.sub(Vp.mean(axis=1), axis=0)
    V_time_w0  = blend_potentials(V_time, V_price, 0.0)
    V_time_w02 = blend_potentials(V_time, V_price, 0.2)
    V_time_w04 = blend_potentials(V_time, V_price, 0.4)
    V_time_w06 = blend_potentials(V_time, V_price, 0.6)
    print("Built neutral blends (w=0/0.2/0.4/0.6). Shapes:", V_time_w04.shape)
else:
    # 5) Timestamp → datetime (detect seconds vs ms robustly)
    unit_prop = infer_ts_unit_series(props["timestamp"])
    props["dt"] = pd.to_datetime(props["timestamp"], unit=unit_prop, utc=True).dt.tz_convert(None)

    # 6) Daily median price per item
    daily_price = (props
                   .groupby([pd.Grouper(key="dt", freq="1D"), "itemid"])["price"]
                   .median()
                   .rename("price")
                   .reset_index())

    # Map original itemids -> iids used by p_time/V_time
    iid_map = {int(item): i for i, item in enumerate(item_index)}
    daily_price = daily_price[daily_price["itemid"].isin(iid_map)].copy()
    if len(daily_price) == 0:
        print("[P1] No daily prices overlap modeled items. Using neutral V_price=0.")
        V_price = pd.DataFrame(0.0, index=p_time.index, columns=p_time.columns)
    else:
        daily_price["iid"] = daily_price["itemid"].map(iid_map)

        # 7) Pivot to [time x iid] and align with p_time’s dates
        price_frame = daily_price.pivot_table(index="dt", columns="iid", values="price", aggfunc="median")
        price_frame = price_frame.sort_index()

        # align to p_time index; forward/back fill across days
        price_frame = price_frame.reindex(p_time.index).ffill().bfill()

        # 8) Rolling median (14 days) to define a discount signal in [0,1]
        roll_med = price_frame.rolling(window=14, min_periods=5).median()
        disc = ((roll_med - price_frame) / roll_med).clip(lower=0.0, upper=1.0)
        disc = disc.fillna(0.0)

        # 9) Convert to a potential: larger discount => smaller V
        eps = 1e-12
        V_price = -np.log(1.0 + disc + eps)
        V_price = pd.DataFrame(V_price, index=price_frame.index, columns=price_frame.columns)

    # 10) Blend with popularity potential and center by day
    def blend_potentials(V_pop: pd.DataFrame, V_price: pd.DataFrame, w: float):
        Vp = (1.0 - w)*V_pop + w*V_price
        return Vp.sub(Vp.mean(axis=1), axis=0)

    V_time_w0  = blend_potentials(V_time, V_price, 0.0)   # centered pop-only baseline
    V_time_w02 = blend_potentials(V_time, V_price, 0.2)
    V_time_w04 = blend_potentials(V_time, V_price, 0.4)
    V_time_w06 = blend_potentials(V_time, V_price, 0.6)
    print("Built V_price and blends (w=0/0.2/0.4/0.6). Shapes:", V_time_w04.shape)

# --- Cell P2: evaluate best config on V blends ---
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply
import numpy as np

# Re-use best hyper-params found in H5/H4
M_best         = 150
ALPHA_best     = 0.5
SIGMA_best     = 0.03
RANK_best      = 5
N_ENS_best     = 3
DELTA_SEC_best = 24*60*60
LAST_days      = 60

# Choose last N days window for eval
dates = p_time.index[-LAST_days-1:]
def eval_with_V(V_alt: pd.DataFrame, label="w=?"):
    # subset columns by top-π
    sub_idx = np.argsort(-pi)[:M_best]
    p_sub = p_time.loc[dates, p_time.columns.isin(sub_idx)].reindex(columns=sub_idx)
    V_sub = V_alt.loc[p_sub.index, sub_idx]

    # generator on the same subset
    Q_M   = Q_rev[sub_idx,:][:,sub_idx]
    Q_M_c = csc_matrix(Q_M)

    # noise directions
    S = 0.5*(Q_M + Q_M.T)
    try:
        import scipy.sparse.linalg as sla
        evals, evecs = sla.eigs(-S, k=min(RANK_best, M_best-2))
        evecs = np.real(evecs)
    except Exception:
        evecs = np.linalg.qr(np.random.randn(M_best, RANK_best))[0]

    # run one pass like H4
    def renorm(x):
        x = np.maximum(x,0); s=x.sum(); 
        return x/s if s>1e-12 else np.ones_like(x)/len(x)
    def tv(p,q): return 0.5*np.abs(p-q).sum()
    def kl(p,q,eps=1e-12):
        p = np.maximum(p,eps); q=np.maximum(q,eps)
        return float((p*(np.log(p)-np.log(q))).sum())

    kl_ctmc, tv_ctmc, kl_det, tv_det, kl_sto, tv_sto, is_event = [],[],[],[],[],[],[]
    p_diff = np.abs(p_sub.diff().dropna().values).sum(axis=1)
    thr = np.quantile(p_diff, 0.9)

    for t in range(len(p_sub.index)-1):
        pt = p_sub.iloc[t].values
        pt1 = p_sub.iloc[t+1].values
        Vnxt = V_sub.iloc[t+1].values

        # event? (relative to p_time dynamics, not model)
        is_event.append(np.abs(p_sub.iloc[t+1].values - p_sub.iloc[t].values).sum() > thr)

        y_ctmc = expm_multiply(DELTA_SEC_best*Q_M_c, pt)
        y_det  = renorm(y_ctmc * np.exp(-ALPHA_best * Vnxt))

        ens=[]
        for _ in range(N_ENS_best):
            z    = np.random.randn(evecs.shape[1])
            y    = renorm(y_det + (evecs @ z) * SIGMA_best)
            ens.append(y)
        y_sto = renorm(np.mean(np.vstack(ens), axis=0))

        kl_ctmc.append(kl(pt1,y_ctmc)); tv_ctmc.append(tv(pt1,y_ctmc))
        kl_det.append(kl(pt1,y_det));   tv_det.append(tv(pt1,y_det))
        kl_sto.append(kl(pt1,y_sto));   tv_sto.append(tv(pt1,y_sto))

    ev = np.array(is_event, bool)
    def avg(x, m=None): 
        x=np.array(x,float); 
        return x.mean() if (m is None or not m.any()) else x[m].mean()

    print(f"[{label}] KL_all={avg(kl_sto):.4f}  TV_all={avg(tv_sto):.4f}  "
          f"KL_ev={avg(kl_sto,ev):.4f}  TV_ev={avg(tv_sto,ev):.4f}")
    return dict(KL_all=avg(kl_sto), TV_all=avg(tv_sto),
                KL_ev=avg(kl_sto,ev), TV_ev=avg(tv_sto,ev))

res_w0  = eval_with_V(V_time_w0,  "w=0.0 (centered baseline)")
res_w02 = eval_with_V(V_time_w02, "w=0.2")
res_w04 = eval_with_V(V_time_w04, "w=0.4")
res_w06 = eval_with_V(V_time_w06, "w=0.6")

