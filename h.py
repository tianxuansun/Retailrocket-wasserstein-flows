# === Cell H1: forecasting utilities (parameterized) ===
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply

def renorm_simplex(x, eps=1e-12):
    x = np.maximum(x, 0.0)
    s = x.sum()
    return x / s if s > eps else np.ones_like(x)/len(x)

def tv_distance(p, q):
    return 0.5 * np.abs(p - q).sum()

def kl_div(p, q, eps=1e-12):
    p = np.maximum(p, eps); q = np.maximum(q, eps)
    return float((p * (np.log(p) - np.log(q))).sum())

def detect_event_windows(p_sub, q=0.9):
    """Return boolean mask over t=0..T-2 of large day-to-day L1 changes."""
    diffs = np.abs(p_sub.diff()).dropna().values
    thr = np.quantile(diffs.sum(axis=1), q)
    return (diffs.sum(axis=1) > thr)

def eval_one_model(Q_rev, p_time, V_time, sub_idx,
                   DELTA_SEC, ALPHA, NOISE_RANK, SIGMA, N_ENS=10, rng_seed=0):
    """
    Evaluate three forecasters on the subset 'sub_idx':
      - CTMC-only: exp(Δ Q_rev)
      - Deterministic drift: CTMC-only * exp(-α V_{t+Δ}) then renormalize
      - Stochastic: deterministic + low-rank noise (ensemble mean)
    Returns: dict of lists + boolean event mask
    """
    rng = np.random.default_rng(rng_seed)

    # subset & column align
    p_sub = p_time.loc[:, p_time.columns.isin(sub_idx)].copy().reindex(columns=sub_idx)
    V_sub = V_time.loc[p_sub.index, sub_idx].copy()
    dates = p_sub.index.to_list()
    Tsteps = len(dates) - 1

    # operator on subset
    Q_M = Q_rev[sub_idx, :][:, sub_idx]
    Q_M_csc = csc_matrix(Q_M)

    # noise directions: top-k of symmetric part or random orthonormal
    S = 0.5 * (Q_M + Q_M.T)
    try:
        import scipy.sparse.linalg as sla
        k = min(NOISE_RANK, Q_M.shape[0]-2)
        evals, evecs = sla.eigs(-S, k=k)
        evecs = np.real(evecs)
    except Exception:
        evecs = np.linalg.qr(rng.normal(size=(Q_M.shape[0], NOISE_RANK)))[0]

    # loop
    kl_ctmc, tv_ctmc = [], []
    kl_det , tv_det  = [], []
    kl_sto , tv_sto  = [], []
    is_event = detect_event_windows(p_sub).tolist()

    for t in range(Tsteps):
        pt   = p_sub.iloc[t].values
        pt1  = p_sub.iloc[t+1].values
        Vnxt = V_sub.iloc[t+1].values

        # 1) CTMC-only
        y_ctmc = expm_multiply(DELTA_SEC * Q_M_csc, pt)

        # 2) Deterministic (tilt by potential)
        y_det = y_ctmc * np.exp(-ALPHA * Vnxt)
        y_det = renorm_simplex(y_det)

        # 3) Stochastic (low-rank noise around det)
        ens = []
        for _ in range(N_ENS):
            z = rng.normal(size=evecs.shape[1])
            y = y_det + (evecs @ z) * SIGMA
            ens.append(renorm_simplex(y))
        y_sto = renorm_simplex(np.mean(np.vstack(ens), axis=0))

        kl_ctmc.append(kl_div(pt1, y_ctmc)); tv_ctmc.append(tv_distance(pt1, y_ctmc))
        kl_det .append(kl_div(pt1, y_det )); tv_det .append(tv_distance(pt1, y_det ))
        kl_sto .append(kl_div(pt1, y_sto )); tv_sto .append(tv_distance(pt1, y_sto ))

    out = dict(kl_ctmc=kl_ctmc, tv_ctmc=tv_ctmc,
               kl_det=kl_det, tv_det=tv_det,
               kl_sto=kl_sto, tv_sto=tv_sto,
               is_event=is_event)
    return out

# === Cell H2: potential cache + data grid ===
from collections import defaultdict

V_cache = {}  # window -> V_time DataFrame

def get_V_time_for_window(p_time, window):
    if window not in V_cache:
        V_cache[window] = (-np.log(1e-12 + p_time.rolling(window=window, min_periods=1).mean())).copy()
    return V_cache[window]

# Small set of data configs (fast):
# Each tuple: (name, K_ITEMS, SESSION_GAP_SEC, MAX_DWELL_SEC, V_window)
data_grid = [
    ("D1_smallK_fast" , 3000, 20*60, 600,  7),
    ("D2_mediumK_base", 5000, 30*60, 600,  7),   # your current baseline
    ("D3_mediumK_long", 5000, 45*60, 900, 14),
]

# === Cell H3: build a data configuration from events ===
def build_data_config(events, name, K_ITEMS, SESSION_GAP_SEC, MAX_DWELL_SEC, V_window, time_bucket="1D"):
    # 1) (re)sessionize with new gap
    ev = events.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)
    ev = add_sessions_fast(ev, gap_sec=SESSION_GAP_SEC)

    # 2) restrict to top-K items
    ev2, item_index_cfg, _ = restrict_topk_items(ev, K=K_ITEMS)

    # 3) transitions/exposures + stationary + reversible generator
    P_cfg, C_cfg, T_cfg = build_transitions(ev2, max_dwell_sec=MAX_DWELL_SEC)
    pi_cfg = stationary_from_P(P_cfg)
    Q_cfg  = estimate_Q_from_counts_and_exposure(C_cfg, T_cfg)
    Qr_cfg = make_reversible_Q(Q_cfg, pi_cfg)

    # 4) time buckets & potential
    p_time_cfg = compute_time_buckets(ev2, bucket=time_bucket)
    V_time_cfg = get_V_time_for_window(p_time_cfg, V_window)

    return dict(name=name,
                item_index=item_index_cfg,
                P=P_cfg, C=C_cfg, T=T_cfg, pi=pi_cfg, Q=Q_cfg, Q_rev=Qr_cfg,
                p_time=p_time_cfg, V_time=V_time_cfg)

# === H4: Compact & robust hyper-parameter sweep (safe to paste as one cell) ===
import numpy as np
import pandas as pd
from pathlib import Path
import shutil, time, math
from tqdm import tqdm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply, eigsh  # eigsh for symmetric problems

# -------------------------
# 0) Paths: local scratch + Drive target
# -------------------------
LOCAL = Path('/content/run_tmp');           LOCAL.mkdir(parents=True, exist_ok=True)
DRIVE = Path('/content/drive/MyDrive/retailrocket_artifacts/sweeps'); DRIVE.mkdir(parents=True, exist_ok=True)
CSV_LOCAL = LOCAL/'results.csv'
CSV_DRIVE = DRIVE/'results.csv'

# -------------------------
# 1) Utilities
# -------------------------
def renorm_simplex(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0.0)
    s = x.sum()
    if not np.isfinite(s) or s <= eps:
        return np.ones_like(x)/len(x)
    return x / s

def tv_distance(p, q):
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    return 0.5 * np.abs(p - q).sum()

def kl_div(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = np.maximum(p, eps); q = np.maximum(q, eps)
    return float((p * (np.log(p) - np.log(q))).sum())

def avg(vals, mask=None):
    vals = np.array(vals, dtype=float)
    if mask is None or (isinstance(mask, np.ndarray) and mask.size and mask.sum()==0):
        return float(vals.mean()) if vals.size else float('nan')
    m = np.asarray(mask, dtype=bool)
    return float(vals[m].mean()) if m.any() else float('nan')

# -------------------------
# 2) Sweep definition (kept small first; expand later)
# -------------------------
DELTA_SEC = 24*60*60       # 1-day step for the generator
rng = np.random.default_rng(123)

GRIDS = []
# A minimal grid: 4 configs that complete quickly. Scale up after you see the CSV fill.
for M in [150]:                # top-M items
    for last_days in [60]:     # last N days for the time series
        for alpha in [0.5, 1.0]:
            for sigma in [0.00, 0.03]:   # 0.00 = deterministic
                GRIDS.append(dict(M=M, last_days=last_days, alpha=alpha, sigma=sigma,
                                   rank=5, nens=3))

# header (overwrites if re-run)
with open(CSV_LOCAL, 'w') as fh:
    fh.write("cfg_id,M,last_days,alpha,sigma,rank,nens,KL_overall,TV_overall,KL_events,TV_events\n")
shutil.copy2(CSV_LOCAL, CSV_DRIVE)  # initial copy

# -------------------------
# 3) Main loop over configs
# -------------------------
start_all = time.time()
for cfg_id, cfg in enumerate(GRIDS):
    t0 = time.time()
    M         = int(cfg['M'])
    last_days = int(cfg['last_days'])
    alpha     = float(cfg['alpha'])
    sigma     = float(cfg['sigma'])
    rank      = int(cfg['rank'])
    nens      = int(cfg['nens'])

    # (a) choose subset & days
    sub = np.argsort(-pi)[:M]                                     # top-M by stationary mass
    days = p_time.index[-min(last_days, len(p_time.index)):]      # last N days available

    # p_sub, V_sub: ensure columns are exactly 'sub' in that order (labels are item iids)
    p_sub = p_time.loc[days, sub].copy()
    p_sub = p_sub.reindex(columns=sub)  # guarantee order
    V_sub = V_time.loc[days, sub].copy()

    # (b) matrices for this subset
    Q_M = Q_rev[sub, :][:, sub]
    Q_M_csc = csc_matrix(Q_M)  # better for expm_multiply

    # (c) noise directions: top 'rank' eigenvectors of -S, where S is symmetric part
    S = 0.5 * (Q_M + Q_M.T)
    try:
        # Largest algebraic of -S  -> smooth/diffusive directions
        evals, evecs = eigsh((-S).astype(float), k=min(rank, max(M-2, 1)), which='LA')
        evecs = np.asarray(evecs.real, dtype=float)
    except Exception:
        # Fallback: random orthonormal
        evecs, _ = np.linalg.qr(rng.standard_normal(size=(M, rank)))

    # (d) prepare event threshold (90th percentile of L1 jump)
    diffs = np.abs(p_sub.diff().to_numpy()[1:]).sum(axis=1)  # shape (Tsteps,)
    thr = float(np.quantile(diffs, 0.90)) if diffs.size else np.inf

    # (e) iterate over days
    Tsteps = len(days) - 1
    kl_vals, tv_vals, ev_mask = [], [], np.zeros(Tsteps, dtype=bool)

    for t in tqdm(range(Tsteps), desc=f"cfg {cfg_id}: M={M},days={last_days},α={alpha},σ={sigma}", leave=False):
        pt   = p_sub.iloc[t].to_numpy(float)
        pt1  = p_sub.iloc[t+1].to_numpy(float)
        Vnxt = V_sub.iloc[t+1].to_numpy(float)

        # mark event
        ev_mask[t] = (np.abs(pt1 - pt).sum() > thr)

        # 1) CTMC baseline
        y_ctmc = expm_multiply(DELTA_SEC * Q_M_csc, pt)

        # 2) Deterministic tilt by potential
        y_det = renorm_simplex(y_ctmc * np.exp(-alpha * Vnxt))

        # 3) Stochastic: low-rank noise around y_det
        if sigma > 0 and rank > 0 and nens > 0:
            ens = []
            for _ in range(nens):
                z = rng.standard_normal(size=(evecs.shape[1],))
                y = renorm_simplex(y_det + sigma * (evecs @ z))
                ens.append(y)
            y_pred = renorm_simplex(np.mean(np.vstack(ens), axis=0))
        else:
            y_pred = y_det

        # metrics vs truth
        kl_vals.append(kl_div(pt1, y_pred))
        tv_vals.append(tv_distance(pt1, y_pred))

    # (f) aggregate + write one CSV row
    row = [
        cfg_id, M, last_days, alpha, sigma, rank, nens,
        avg(kl_vals), avg(tv_vals),
        avg(kl_vals, ev_mask), avg(tv_vals, ev_mask)
    ]
    with open(CSV_LOCAL, 'a') as fh:
        fh.write(",".join(map(str, row)) + "\n")

    # copy to Drive after each config
    shutil.copy2(CSV_LOCAL, CSV_DRIVE)
    print(f"[cfg {cfg_id}] done in {time.time()-t0:.1f}s → overall KL={row[7]:.4f}, TV={row[8]:.4f}, events KL={row[9]:.4f}, TV={row[10]:.4f}")

print(f"All sweeps finished in {time.time()-start_all:.1f}s")
print("Results saved to:")
print(" -", CSV_LOCAL)
print(" -", CSV_DRIVE)

# Optional: show a quick summary table in the output
try:
    display(pd.read_csv(CSV_LOCAL))
except Exception:
    pass

# === H5 (revised): pick best config from CSV, rebuild subset, re-evaluate & ablations ===
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply, eigsh

# Paths where H4 wrote results
CSV_DRIVE = Path('/content/drive/MyDrive/retailrocket_artifacts/sweeps/results.csv')
CSV_LOCAL = Path('/content/run_tmp/results.csv')
CSV_PATH = CSV_DRIVE if CSV_DRIVE.exists() else CSV_LOCAL
assert CSV_PATH.exists(), "results.csv not found. Run H4 first."

results_df = pd.read_csv(CSV_PATH)
assert len(results_df), "results.csv is empty. Expand the H4 grid and re-run."

# 1) Pick best by overall KL (lower is better)
best_row = results_df.loc[results_df['KL_overall'].idxmin()].to_dict()
print("Best config (from results.csv):")
for k in ['M','last_days','alpha','sigma','rank','nens','KL_overall','TV_overall','KL_events','TV_events']:
    print(f"  {k}: {best_row[k]}")

# 2) Rebuild the chosen data subset
DELTA_SEC = 24*60*60
M         = int(best_row['M'])
last_days = int(best_row['last_days'])
alpha_b   = float(best_row['alpha'])
sigma_b   = float(best_row['sigma'])
rank_b    = int(best_row['rank'])
nens_b    = int(best_row['nens'])

# (a) top-M items by stationary mass
sub = np.argsort(-pi)[:M]
# (b) last N days
days = p_time.index[-min(last_days, len(p_time.index)):]
# (c) subset matrices/frames; ensure column order == sub
Q_M = Q_rev[sub, :][:, sub]
Q_M_csc = csc_matrix(Q_M)
p_sub = p_time.loc[days, sub].reindex(columns=sub).copy()
V_sub = V_time.loc[days, sub].copy()

# Utilities
def renorm_simplex(x, eps=1e-12):
    x = np.maximum(np.asarray(x, float), 0.0)
    s = x.sum()
    return x/s if s > eps else np.ones_like(x)/len(x)

def tv_distance(p, q):
    return 0.5 * np.abs(np.asarray(p,float) - np.asarray(q,float)).sum()

def kl_div(p, q, eps=1e-12):
    p = np.maximum(np.asarray(p,float), eps)
    q = np.maximum(np.asarray(q,float), eps)
    return float((p * (np.log(p) - np.log(q))).sum())

def avg(vals, mask=None):
    vals = np.array(vals, float)
    if mask is None:
        return float(vals.mean())
    m = np.asarray(mask, bool)
    return float(vals[m].mean()) if m.any() else np.nan

# Precompute smooth noise directions from symmetric part
S = 0.5 * (Q_M + Q_M.T)
try:
    # eigenvectors of -S (largest algebraic) → diffusive modes
    evals, evecs = eigsh((-S).astype(float), k=min(rank_b, max(M-2, 1)), which='LA')
    evecs = np.asarray(evecs.real, float)
except Exception:
    # fallback: random orthonormal
    evecs, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((M, max(rank_b,1))))

# Event threshold (90th percentile of L1 jumps)
diffs = np.abs(p_sub.diff().to_numpy()[1:]).sum(axis=1)
thr = float(np.quantile(diffs, 0.90)) if diffs.size else np.inf

def eval_series(alpha, sigma, rank, nens, rng=None):
    """
    Returns a dict with daily arrays:
      kl_ctmc, tv_ctmc, kl_det, tv_det, kl_sto, tv_sto, is_event
    """
    rng = rng or np.random.default_rng(0)
    Tsteps = len(days) - 1
    outs = dict(kl_ctmc=[], tv_ctmc=[], kl_det=[], tv_det=[], kl_sto=[], tv_sto=[], is_event=[])

    for t in range(Tsteps):
        pt   = p_sub.iloc[t].to_numpy(float)
        pt1  = p_sub.iloc[t+1].to_numpy(float)
        Vnxt = V_sub.iloc[t+1].to_numpy(float)

        # mark event
        outs['is_event'].append(np.abs(pt1 - pt).sum() > thr)

        # CTMC baseline
        y_ctmc = expm_multiply(DELTA_SEC * Q_M_csc, pt)

        # Deterministic tilt
        y_det = renorm_simplex(y_ctmc * np.exp(-alpha * Vnxt))

        # Stochastic (if sigma>0), otherwise equals deterministic
        if sigma > 0 and rank > 0 and nens > 0:
            ens = []
            for _ in range(int(nens)):
                z = rng.standard_normal(rank if evecs.ndim==2 else 1)
                noise = (evecs @ z) if evecs.ndim==2 else evecs.squeeze() * z
                y = renorm_simplex(y_det + float(sigma) * noise)
                ens.append(y)
            y_sto = renorm_simplex(np.mean(np.vstack(ens), axis=0))
        else:
            y_sto = y_det

        # metrics
        outs['kl_ctmc'].append(kl_div(pt1, y_ctmc))
        outs['tv_ctmc'].append(tv_distance(pt1, y_ctmc))
        outs['kl_det' ].append(kl_div(pt1, y_det ))
        outs['tv_det' ].append(tv_distance(pt1, y_det ))
        outs['kl_sto' ].append(kl_div(pt1, y_sto ))
        outs['tv_sto' ].append(tv_distance(pt1, y_sto ))

    # convert lists → arrays
    for k in list(outs.keys()):
        outs[k] = np.array(outs[k], dtype=float) if k != 'is_event' else np.array(outs[k], bool)
    return outs

# 3) Evaluate best stochastic & ablations
out_best = eval_series(alpha=alpha_b, sigma=sigma_b, rank=rank_b, nens=nens_b, rng=np.random.default_rng(123))
out_nonoise = eval_series(alpha=alpha_b, sigma=0.0,    rank=rank_b, nens=nens_b, rng=np.random.default_rng(123))
out_nopot   = eval_series(alpha=0.0,    sigma=sigma_b, rank=rank_b, nens=nens_b, rng=np.random.default_rng(123))

ev_mask = out_best['is_event']

def summarize(out):
    return dict(
        KL_all = avg(out['kl_sto']),
        TV_all = avg(out['tv_sto']),
        KL_ev  = avg(out['kl_sto'], ev_mask),
        TV_ev  = avg(out['tv_sto'], ev_mask),
    )

ablation = pd.DataFrame([
    dict(model="CTMC-only",
         KL_all=avg(out_best['kl_ctmc']), TV_all=avg(out_best['tv_ctmc']),
         KL_ev =avg(out_best['kl_ctmc'], ev_mask), TV_ev=avg(out_best['tv_ctmc'], ev_mask)),
    dict(model="Deterministic (α tilt, σ=0)",
         KL_all=avg(out_nonoise['kl_det']), TV_all=avg(out_nonoise['tv_det']),
         KL_ev =avg(out_nonoise['kl_det'], ev_mask), TV_ev=avg(out_nonoise['tv_det'], ev_mask)),
    dict(model=f"Stochastic (α={alpha_b}, σ={sigma_b})", **summarize(out_best)),
    dict(model=f"No potential (α=0, σ={sigma_b})", **summarize(out_nopot)),
])
print("\nAblation summary (overall vs event windows):")
display(ablation)

# === Cell H6: statistical significance (Wilcoxon on paired daily diffs) ===
from scipy.stats import wilcoxon
import numpy as np

kl_det = np.array(out_best["kl_det"])
kl_sto = np.array(out_best["kl_sto"])
tv_det = np.array(out_best["tv_det"])
tv_sto = np.array(out_best["tv_sto"])

# paired differences (sto - det); negative is an improvement
dKL = kl_sto - kl_det
dTV = tv_sto - tv_det

stat_kl, p_kl = wilcoxon(dKL)   # H0: median difference == 0
stat_tv, p_tv = wilcoxon(dTV)

print(f"Wilcoxon (KL): statistic={stat_kl:.1f}, p-value={p_kl:.3g}, median ΔKL={np.median(dKL):.4f}")
print(f"Wilcoxon (TV): statistic={stat_tv:.1f}, p-value={p_tv:.3g}, median ΔTV={np.median(dTV):.4f}")

# Optional: event-only test
ev_mask = np.array(out_best["is_event"], dtype=bool)
if ev_mask.any():
    stat_kl_ev, p_kl_ev = wilcoxon(dKL[ev_mask])
    print(f"Wilcoxon (KL, event windows): p-value={p_kl_ev:.3g}, median ΔKL={np.median(dKL[ev_mask]):.4f}")
