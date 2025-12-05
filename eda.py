from google.colab import drive
drive.mount('/content/drive')

# === Cell 1: imports & config ===
import os, math, gc
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, dia_matrix
from tqdm import tqdm

DATA_DIR = "/content/drive/MyDrive/retailrocket"  # <-- change to your path
K_ITEMS = 5000                      # start small for speed; bump later (e.g., 10k, 20k)
SESSION_GAP_SEC = 30*60             # 30-min inactivity split
MAX_DWELL_SEC = 10*60               # cap per-step dwell at 10 min
TIME_BUCKET = "1D"                  # daily buckets for p_t and V(t)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

# === Cell 2: loaders ===
def load_events():
    usecols = ["timestamp", "visitorid", "event", "itemid", "transactionid"]
    dtypes  = {"timestamp": np.int64, "visitorid": np.int64, "event": "category", "itemid": np.int64, "transactionid": "float64"}
    df = pd.read_csv(os.path.join(DATA_DIR, "events.csv"), usecols=usecols, dtype=dtypes)
    df = df.sort_values(["visitorid", "timestamp"], kind="mergesort").reset_index(drop=True)
    return df

def load_item_properties():
    cols = ["timestamp", "itemid", "property", "value"]
    p1 = pd.read_csv(os.path.join(DATA_DIR, "item_properties_part1.csv"), usecols=cols)
    p2 = pd.read_csv(os.path.join(DATA_DIR, "item_properties_part2.csv"), usecols=cols)
    props = pd.concat([p1, p2], axis=0, ignore_index=True)
    props = props.sort_values(["itemid", "timestamp"]).reset_index(drop=True)
    return props

# === Cell 3: utilities ===
def infer_ts_unit(ts_series: pd.Series) -> str:
    """
    Heuristic: if max timestamp > 1e11, treat as milliseconds; else seconds.
    Retailrocket is in milliseconds.
    """
    mx = int(ts_series.max())
    return "ms" if mx > 1e11 else "s"

def filter_events(df, keep_events=("view","addtocart"), min_ts=None, max_ts=None):
    if min_ts is not None: df = df[df["timestamp"] >= min_ts]
    if max_ts is not None: df = df[df["timestamp"] <  max_ts]
    if keep_events is not None:
        df = df[df["event"].isin(keep_events)]
    return df.reset_index(drop=True)

def add_sessions_fast(df, gap_sec=30*60):
    """
    Vectorized sessionization without groupby.apply (avoids deprecation warning and is faster).
    Assumes df sorted by visitorid, timestamp.
    """
    df = df.copy()
    # mark start of a visitor block
    new_visitor = (df["visitorid"].shift(1) != df["visitorid"])
    # time diff within same visitor
    dt = df["timestamp"].diff().where(~new_visitor, np.inf)
    session_break = (dt.values > gap_sec)
    # cumulative session ids within a visitor; reset when visitor changes
    # build a visitor-wise cumulative sum, then combine with visitor id to make global sessions
    session_incr = session_break.astype(np.int64)
    # cumulative sum but reset per visitor
    csum = df.groupby("visitorid", sort=False).cumcount()  # 0.. within each visitor
    # We can't directly cumsum session_incr per visitor easily; do it with groupby.transform on an array:
    session_local = df.groupby("visitorid", sort=False)["timestamp"].transform(
        lambda s: np.cumsum((s.diff().fillna(gap_sec+1).values > gap_sec).astype(np.int64))
    ).astype(np.int64)
    df["session"] = session_local
    # global session id
    df["session_global"] = (df["visitorid"].astype(str) + "_" + df["session"].astype(str)).astype("category").cat.codes
    return df

# === Cell 4: top-K items ===
def restrict_topk_items(df, K=5000):
    top_items = df["itemid"].value_counts().head(K).index
    df = df[df["itemid"].isin(top_items)].copy()
    item_index = pd.Index(sorted(df["itemid"].unique()))
    item_to_idx = pd.Series(np.arange(len(item_index), dtype=np.int64), index=item_index)
    df["iid"] = item_to_idx.loc[df["itemid"]].values
    return df, item_index, item_to_idx

# === Cell 5: transitions & exposures (REPLACE THIS WHOLE FUNCTION) ===
from scipy.sparse import csr_matrix, dia_matrix

def build_transitions(df, max_dwell_sec=10*60, self_loops=True):
    """
    Build (C, P, T) from a sessionized, iid-indexed events DataFrame.
    - Unit-aware dwell time (ms vs s)
    - Row-stochastic P
    - Optionally add lazy self-loops for rows with no outgoing transitions
    """
    # keep sort
    df = df.sort_values(["session_global", "timestamp"])
    df["iid_next"] = df.groupby("session_global")["iid"].shift(-1)
    df["ts_next"]  = df.groupby("session_global")["timestamp"].shift(-1)
    valid = df["iid_next"].notna()

    # transition counts C
    trans = df.loc[valid, ["iid", "iid_next"]].astype({"iid": np.int64, "iid_next": np.int64})
    N = trans.value_counts().rename("cnt").reset_index()
    N.columns = ["i", "j", "cnt"]
    n_items = int(df["iid"].max()) + 1
    C = csr_matrix((N["cnt"].values, (N["i"].values, N["j"].values)),
                   shape=(n_items, n_items), dtype=np.float64)

    # ---- UNIT-AWARE DWELL (ms vs s) ----
    # infer from your raw timestamps (they are in the same units as ts_next)
    unit = infer_ts_unit(df["timestamp"])           # 'ms' or 's'
    scale = 1e-3 if unit == "ms" else 1.0           # convert to seconds

    # exposure T_i (cap each step)
    dwell = df.loc[valid, ["iid", "timestamp", "ts_next"]].copy()
    dwell["dt"] = ((dwell["ts_next"] - dwell["timestamp"]) * scale)\
                    .clip(lower=1.0, upper=max_dwell_sec)
    Ti = dwell.groupby("iid")["dt"].sum().astype(np.float64)

    T = np.zeros(n_items, dtype=np.float64)
    T[Ti.index.values] = Ti.values
    T[T <= 0] = 1.0

    # ---- ROW-STOCHASTIC P ----
    rowsum = np.asarray(C.sum(axis=1)).ravel()
    row_inv = np.reciprocal(np.maximum(rowsum, 1.0))
    P = dia_matrix((row_inv, 0), shape=(n_items, n_items)).dot(C).tocsr()

    # add lazy self-loops where a row had no outgoing transitions
    if self_loops:
        zero = (rowsum == 0)
        if zero.any():
            P = P.tolil()
            P[zero, zero] = 1.0
            P = P.tocsr()

    return P, C, T

# === Cell 6: stationary pi, Q, reversible Q ===
def stationary_from_P(P, tol=1e-12, maxit=2000):
    """
    Power method for the left stationary distribution: pi^T P = pi^T.
    Uses sparse-friendly matvec: pi_new = (P^T @ pi).
    """
    n = P.shape[0]
    pi = np.ones(n, dtype=np.float64) / n
    for _ in range(maxit):
        # sparse-safe: P.T @ pi returns ndarray
        pi_new = (P.T @ pi)
        pi_new = np.asarray(pi_new, dtype=np.float64).ravel()
        s = pi_new.sum()
        if s <= 0 or not np.isfinite(s):
            # fallback normalization to avoid NaNs/infs
            pi_new = np.ones(n, dtype=np.float64) / n
        else:
            pi_new = pi_new / s
        if np.linalg.norm(pi_new - pi, 1) < tol:
            return pi_new
        pi = pi_new
    return pi  # return the last iterate if not converged

def estimate_Q_from_counts_and_exposure(C, T):
    C = C.tocsr()
    n = C.shape[0]
    rows, cols = C.nonzero()
    data = C.data / T[rows]
    Q_off = csr_matrix((data, (rows, cols)), shape=(n, n))
    out = np.asarray(Q_off.sum(axis=1)).ravel()
    Q = Q_off - dia_matrix((out, 0), shape=(n, n))
    return Q.tocsr()

def make_reversible_Q(Q, pi):
    n = Q.shape[0]
    Pi     = dia_matrix((pi, 0), shape=(n, n))
    Pi_inv = dia_matrix((1.0/np.maximum(pi, 1e-12), 0), shape=(n, n))
    Q_rev = 0.5 * (Q + Pi_inv.dot(Q.T).dot(Pi))
    out = np.asarray(Q_rev.sum(axis=1)).ravel()
    Q_rev = Q_rev - dia_matrix((out, 0), shape=Q_rev.shape)
    return Q_rev.tocsr()

def detailed_balance_gap(Q, pi, sample=20000):
    Q = Q.tocsr()
    r, c = Q.nonzero()
    mask = (r != c)
    r, c = r[mask], c[mask]
    if len(r) == 0:
        return 0.0
    if len(r) > sample:
        idx = np.random.choice(len(r), size=sample, replace=False)
        r, c = r[idx], c[idx]
    vals_ij = np.asarray(Q[r, c]).ravel()  # robust: no .A
    vals_ji = np.asarray(Q[c, r]).ravel()
    return float(np.mean(np.abs(pi[r]*vals_ij - pi[c]*vals_ji)))

# === Cell 7: time buckets & V(t) with ms fix ===
def compute_time_buckets(df, bucket="1D"):
    # Detect unit
    unit = infer_ts_unit(df["timestamp"])
    # Build counts per time bucket × item
    tmp = df[["timestamp", "iid"]].copy()
    tmp["dt"] = pd.to_datetime(tmp["timestamp"], unit=unit, utc=True).dt.tz_convert(None)
    tmp.set_index("dt", inplace=True)
    counts = tmp.groupby([pd.Grouper(freq=bucket), "iid"]).size().rename("cnt")
    frame = counts.reset_index().pivot_table(index="dt", columns="iid", values="cnt", fill_value=0)
    # Row-normalize to get p_t (probability over items in each time bucket)
    row_sum = frame.sum(axis=1).replace(0, np.nan)
    P_rows = frame.div(row_sum, axis=0).fillna(0.0)
    return P_rows

def rolling_popularity_V(P_rows, window=7):
    eps = 1e-12
    roll = P_rows.rolling(window=window, min_periods=1).mean()
    V = -np.log(eps + roll)
    return V

# 1) Load & filter
events = load_events()
print(f"Events loaded: {len(events):,}")
events = filter_events(events, keep_events=("view","addtocart"))

events, item_index, item_to_idx = restrict_topk_items(events, K=K_ITEMS)
print(f"Unique items kept: {len(item_index):,}")

P, C, T = build_transitions(events, max_dwell_sec=MAX_DWELL_SEC)
print(f"P shape={P.shape}, nnz={P.nnz:,}")
print(f"C shape={C.shape}, nnz={C.nnz:,}")

# === Sanity Cell A1: checks immediately after build_transitions ===
import numpy as np
from scipy.sparse import isspmatrix

def check_after_build(P, C, T):
    assert isspmatrix(P) and isspmatrix(C), "P and C must be SciPy sparse matrices"
    # nonnegativity
    assert (P.data >= -1e-12).all(), "P has negative entries"
    # row-stochasticity
    row = np.asarray(P.sum(axis=1)).ravel()
    max_dev = float(np.abs(row - 1.0).max())
    print(f"[P] row-sum max deviation: {max_dev:.3e}")
    assert max_dev < 1e-9, "P rows are not stochastic (did you use the self-loop version?)"
    # self-loop rows actually inserted (rows with zero C-out but diag(P)=1)
    rowsum_C = np.asarray(C.sum(axis=1)).ravel()
    zero_out = (rowsum_C == 0)
    diagP = P.diagonal()
    used_self = int(np.sum(zero_out & (diagP > 0.99)))
    print(f"[P] self-loop rows inserted: {used_self}")
    # exposure positive
    assert np.all(T > 0), "Exposure T has non-positive entries"

check_after_build(P, C, T)

pi = stationary_from_P(P)
Q  = estimate_Q_from_counts_and_exposure(C, T)
Q_rev = make_reversible_Q(Q, pi)
print(f"Q_rev shape={Q_rev.shape}, nnz={Q_rev.nnz:,}")
print("Detailed-balance gap (Q):", detailed_balance_gap(Q, pi))
print("Detailed-balance gap (Q_rev):", detailed_balance_gap(Q_rev, pi))

# === Sanity Cell A2: generators & reversibility checks ===
import numpy as np

def check_after_generators(P, pi, Q, Q_rev, sample=20000):
    # pi well-formed and left-stationary (quick residual)
    assert np.isfinite(pi).all() and (pi >= 0).all()
    assert abs(pi.sum() - 1.0) < 1e-10
    resid = np.linalg.norm(P.T @ pi - pi, 1)
    print(f"[pi] ||P^T pi - pi||_1 = {resid:.3e}")

    # generator row sums ~ 0
    row_Q  = float(np.abs(np.asarray(Q.sum(axis=1)).ravel()).max())
    row_Qr = float(np.abs(np.asarray(Q_rev.sum(axis=1)).ravel()).max())
    print(f"[Q ] max |row sum| = {row_Q:.3e}")
    print(f"[Qr] max |row sum| = {row_Qr:.3e}")
    assert row_Q  < 1e-9 and row_Qr < 1e-9

    # off-diagonals >= 0, diagonals <= 0 (generators)
    Q_off = Q.copy(); Q_off.setdiag(0)
    assert Q_off.data.min() >= -1e-12, "Q has negative off-diagonals"
    assert (Q.diagonal() <= 1e-12).all(), "Q has positive diagonals"

    # detailed balance gap for Q_rev (≈0 if reversible)
    from numpy.random import default_rng
    rng = default_rng(0)
    Qc = Q_rev.tocsr()
    r, c = Qc.nonzero()
    mask = (r != c)
    r, c = r[mask], c[mask]
    if len(r) > sample:
        idx = rng.choice(len(r), size=sample, replace=False)
        r, c = r[idx], c[idx]
    v_ij = np.asarray(Qc[r, c]).ravel()
    v_ji = np.asarray(Qc[c, r]).ravel()
    gap = float(np.mean(np.abs(pi[r] * v_ij - pi[c] * v_ji)))
    print(f"[Qr] detailed-balance gap ≈ {gap:.3e}")

check_after_generators(P, pi, Q, Q_rev)

p_time = compute_time_buckets(events, bucket=TIME_BUCKET)  # uses ms if needed
V_time = rolling_popularity_V(p_time, window=7)
print(f"p_time: {p_time.shape[0]} buckets × {p_time.shape[1]} items")
print(f"V_time: {V_time.shape[0]} buckets × {V_time.shape[1]} items")

import pathlib, pickle
from scipy.sparse import save_npz
import numpy as np
BASE = pathlib.Path('/content/drive/MyDrive/retailrocket_artifacts')
BASE.mkdir(parents=True, exist_ok=True)

# 1) Save each piece separately (robust & reloadable)
save_npz(BASE/'P.npz', P)
save_npz(BASE/'C.npz', C)
np.save(BASE/'T.npy', T)
np.save(BASE/'pi.npy', pi)
save_npz(BASE/'Q.npz', Q)
save_npz(BASE/'Q_rev.npz', Q_rev)

# DataFrames as parquet (compact, fast)
p_time.to_parquet(BASE/'p_time.parquet')
V_time.to_parquet(BASE/'V_time.parquet')

# item_index is a pandas Index of itemids
pd.Series(item_index, name='itemid').to_csv(BASE/'item_index.csv', index=False)

print("Saved all artifacts to:", BASE)

# === Cell 9: save to disk to avoid recompute ===
import pickle, pathlib
OUT = pathlib.Path("/content/retailrocket_artifacts")
OUT.mkdir(parents=True, exist_ok=True)

artifacts = {
    "item_index": item_index,
    "P": P, "C": C, "T": T, "pi": pi, "Q": Q, "Q_rev": Q_rev,
    "p_time": p_time, "V_time": V_time,
}
with open(OUT/"artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved:", OUT/"artifacts.pkl")

# === Sanity: rows of P sum to 1 (up to numerical) ===
row_sumP = np.abs(np.asarray(P.sum(axis=1)).ravel() - 1.0).mean()
print("Mean |row_sum(P)-1|:", row_sumP)

# Q row sums should be 0 (generator)
row_sumQ = np.abs(np.asarray(Q.sum(axis=1)).ravel()).mean()
row_sumQrev = np.abs(np.asarray(Q_rev.sum(axis=1)).ravel()).mean()
print("Mean |row_sum(Q)|:", row_sumQ, " |row_sum(Q_rev)|:", row_sumQrev)

# === Cell 1: load artifacts from Google Drive ===
from google.colab import drive
drive.mount('/content/drive')

import pickle, pathlib, math, gc, os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, load_npz
import matplotlib.pyplot as plt

BASE = pathlib.Path("/content/drive/MyDrive/retailrocket_artifacts")

def load_artifacts(base: pathlib.Path):
    """
    Prefer split files (P.npz, C.npz, etc.) if present.
    Otherwise fall back to a single artifacts.pkl.
    """
    files = {p.name for p in base.glob("*")}
    arts = {}

    if {"P.npz","C.npz","T.npy","pi.npy","Q.npz","Q_rev.npz","p_time.parquet","V_time.parquet","item_index.csv"} <= files:
        # Split format (recommended)
        P = load_npz(base/"P.npz")
        C = load_npz(base/"C.npz")
        T = np.load(base/"T.npy")
        pi = np.load(base/"pi.npy")
        Q = load_npz(base/"Q.npz")
        Q_rev = load_npz(base/"Q_rev.npz")
        p_time = pd.read_parquet(base/"p_time.parquet")
        V_time = pd.read_parquet(base/"V_time.parquet")
        item_index = pd.read_csv(base/"item_index.csv")["itemid"].astype(int).values
        arts = {"P":P,"C":C,"T":T,"pi":pi,"Q":Q,"Q_rev":Q_rev,
                "p_time":p_time,"V_time":V_time,"item_index":item_index}
    else:
        # Fallback: single pickle
        ART = base/"artifacts.pkl"
        if not ART.exists():
            raise FileNotFoundError(f"Could not find split files or {ART}. "
                                    "Double-check the folder or re-save artifacts to Google Drive.")
        with open(ART, "rb") as f:
            arts = pickle.load(f)

        # normalize item_index to a 1D numpy array of ints
        idx = arts["item_index"]
        if hasattr(idx, "values"):
            idx = idx.values
        arts["item_index"] = np.asarray(idx).astype(int)

    return arts

arts = load_artifacts(BASE)

item_index = arts["item_index"]     # np.array of original itemids (len = n_items)
P  = arts["P"]                      # csr row-stochastic (n x n)
C  = arts["C"]                      # csr counts (n x n)
T  = arts["T"]                      # np.array exposure per item (n,)
pi = arts["pi"]                     # stationary distribution from P (n,)
Q  = arts["Q"]                      # csr CTMC generator
Q_rev = arts["Q_rev"]               # csr reversible CTMC generator
p_time = arts["p_time"]             # DataFrame: time buckets x items (iid reindex)
V_time = arts["V_time"]             # DataFrame: time buckets x items (iid reindex)

n_items = P.shape[0]
print("Loaded from:", BASE)
print("Shapes:",
      "P", P.shape, "C", C.shape, "| T", T.shape, "| pi", pi.shape,
      "| p_time", p_time.shape, "| V_time", V_time.shape,
      "| items", len(item_index))

# 2A) Daily total activity (sum over items) — shows seasonality/bursts
daily_total = p_time.sum(axis=1)  # ~1 each day (row-normalized)
plt.figure(figsize=(9,3))
plt.plot(p_time.index, daily_total.values)
plt.title("Sanity: Sum of probabilities per day (should be 1.0)")
plt.xlabel("Date"); plt.ylabel("Sum over items")
plt.tight_layout(); plt.show()

# 2B) Top-20 items by stationary mass (pi)
top_k = 20
top_idx = np.argsort(-pi)[:top_k]
top_items = item_index[top_idx]
top_pi = pi[top_idx]

plt.figure(figsize=(10,4))
x = np.arange(top_k)
plt.bar(x, top_pi)
plt.xticks(x, [str(i) for i in top_items], rotation=70)
plt.title(f"Top {top_k} Items by Stationary Mass (π)")
plt.xlabel("ItemID"); plt.ylabel("π")
plt.tight_layout(); plt.show()

# 2C) Degree distributions from counts graph (in/out)
out_deg = np.asarray((C > 0).sum(axis=1)).ravel()
in_deg  = np.asarray((C > 0).sum(axis=0)).ravel()

plt.figure(figsize=(10,3))
plt.subplot(1,2,1); plt.hist(out_deg, bins=50); plt.title("Out-degree"); plt.xlabel("out-degree"); plt.ylabel("count")
plt.subplot(1,2,2); plt.hist(in_deg,  bins=50); plt.title("In-degree");  plt.xlabel("in-degree")
plt.tight_layout(); plt.show()

# 2D) Exposure time distribution (T)
plt.figure(figsize=(8,3))
plt.hist(T[T>0], bins=50)
plt.title("Exposure time T_i (seconds, capped)")
plt.xlabel("T_i"); plt.ylabel("count")
plt.tight_layout(); plt.show()

# 2E) Popularity vs exposure sanity check
valid = (T > 0)
corr = np.corrcoef(pi[valid], T[valid])[0,1]
print("Corr(pi, T) on valid items:", float(corr))


# Top-10 items (by π) over time
#Cell 3 — Time-series snapshots (unchanged)
k = 10
top10 = top_idx[:k]
cols = [c for c in p_time.columns if c in top10]  # iid indices (0..n_items-1)

plt.figure(figsize=(10,5))
for iid in cols:
    series = p_time[iid]
    plt.plot(series.index, series.values, label=f"iid={iid}")
plt.title("Top-10 items: probability over time p_t(i)")
plt.xlabel("Date"); plt.ylabel("p_t(i)")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout(); plt.show()

# Compare V vs p for one item
iid0 = int(cols[0])
plt.figure(figsize=(10,4))
plt.plot(p_time.index, p_time[iid0].values, label="p_t(i)")
plt.plot(V_time.index, V_time[iid0].values, label="V_t(i)")
plt.title(f"Item iid={iid0}: p_t vs V_t (rolling popularity)")
plt.xlabel("Date"); plt.ylabel("value")
plt.legend(); plt.tight_layout(); plt.show()

import networkx as nx
import numpy as np

def sampled_directed_graph(P, pi, item_index, N=200, m_per_node=5, min_prob=0.0):
    top = np.argsort(-pi)[:N]
    top_set = set(top.tolist())
    P_top = P[top, :]
    G = nx.DiGraph()
    for idx in top:
        G.add_node(int(idx), item=int(item_index[idx]), pi=float(pi[idx]))
    for local_idx, src in enumerate(top):
        row = P_top[local_idx, :]
        if row.nnz == 0:
            continue
        cols = row.indices
        vals = row.data
        order = np.argsort(-vals)
        kept = 0
        for k in order:
            j = cols[k]
            if P[src, j] <= min_prob:
                continue
            if j in top_set:
                G.add_edge(int(src), int(j), w=float(P[src, j]))
                kept += 1
                if kept >= m_per_node:
                    break
    return G

G_top = sampled_directed_graph(P, pi, item_index, N=200, m_per_node=5, min_prob=0.0)
print("Sampled graph:", G_top.number_of_nodes(), "nodes,", G_top.number_of_edges(), "edges")

# Cell 4 — Sampled graph for visualization (unchanged)
plt.figure(figsize=(9,8))
pos = nx.spring_layout(G_top, k=0.15, iterations=50, seed=42)
pis = np.array([G_top.nodes[n]["pi"] for n in G_top.nodes()])
sz = 300 * (pis / pis.max() + 0.05)
nx.draw_networkx_nodes(G_top, pos, node_size=sz, alpha=0.8)
nx.draw_networkx_edges(G_top, pos, alpha=0.2, arrows=False)
lbl_nodes = sorted(G_top.nodes(), key=lambda n: G_top.nodes[n]["pi"], reverse=True)[:15]
labels = {n: str(G_top.nodes[n]["item"]) for n in lbl_nodes}
nx.draw_networkx_labels(G_top, pos, labels=labels, font_size=8)
plt.title("Sampled item→item graph (Top-N by π; edges by P_ij)")
plt.axis("off"); plt.tight_layout(); plt.show()

# Cell 5 — Visualize sampled graph (unchanged)
!pip -q install pyvis
from pyvis.network import Network

def to_pyvis(G, height="700px", notebook=True):
    nt = Network(notebook=notebook, height=height, directed=True)
    node_pis = nx.get_node_attributes(G, "pi")
    max_pi = max(node_pis.values()) if node_pis else 1.0
    for n, data in G.nodes(data=True):
        label = str(data.get("item", n))
        size = 10 + 20 * (data.get("pi", 0.0) / max_pi)
        nt.add_node(n, label=label, title=f"item: {label}\npi: {data.get('pi',0):.6f}", value=size)
    for u, v, edata in G.edges(data=True):
        w = edata.get("w", 0.0)
        nt.add_edge(u, v, title=f"P_ij={w:.6f}")
    nt.toggle_physics(True)
    return nt

nt = to_pyvis(G_top)
nt.show("sampled_graph.html")
print("Rendered: sampled_graph.html — download or open from the Colab file browser.")

from google.colab import files
files.download("sampled_graph.html")

# Cell 6 — Heatmap of a small P block (unchanged)
k = 40
idx = np.argsort(-pi)[:k]
P_block = P[idx, :][:, idx].toarray()

plt.figure(figsize=(6,5))
plt.imshow(P_block, aspect='auto')
plt.colorbar(label='P_ij')
plt.title(f"P submatrix (top {k} by π)")
plt.xlabel("j"); plt.ylabel("i")
plt.tight_layout(); plt.show()

n = n_items
nnz_P = P.nnz
nnz_C = C.nnz

print("Graph summary:")
print(f"- Nodes (items): {n:,}")
print(f"- Directed edges (nonzero transitions in counts C): {nnz_C:,}")
print(f"- Row-stochastic edges (P) nnz: {nnz_P:,}")
print(f"- Mean out-degree (counts>0): {float(np.mean((C > 0).sum(axis=1))):.1f}")
print(f"- Mean in-degree  (counts>0): {float(np.mean((C > 0).sum(axis=0))):.1f}")

# optional SCC snapshot
try:
    from networkx.algorithms.components import strongly_connected_components
    G_counts = nx.from_scipy_sparse_array((C>0).astype(np.int8), create_using=nx.DiGraph)
    scc_sizes = sorted([len(s) for s in strongly_connected_components(G_counts)], reverse=True)[:5]
    print("Top strongly-connected-component sizes (counts graph):", scc_sizes)
except Exception as e:
    print("SCC summary skipped:", e)

def savefig(path):
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()

n_items = P.shape[0]
# Top-20 stationary mass
top_k = 20
top_idx = np.argsort(-pi)[:top_k]
top_items = item_index[top_idx]
top_pi = pi[top_idx]

plt.figure(figsize=(10,4))
x = np.arange(top_k)
plt.bar(x, top_pi)
plt.xticks(x, [str(i) for i in top_items], rotation=70)
plt.title(f"Top {top_k} Items by Stationary Mass (π)")
plt.xlabel("ItemID"); plt.ylabel("π")
#savefig(FIGS/'fig_top20_pi.png')

# Degree histograms
out_deg = np.asarray((C > 0).sum(axis=1)).ravel()
in_deg  = np.asarray((C > 0).sum(axis=0)).ravel()
plt.figure(figsize=(10,3))
plt.subplot(1,2,1); plt.hist(out_deg, bins=50); plt.title("Out-degree"); plt.xlabel("out-degree"); plt.ylabel("count")
plt.subplot(1,2,2); plt.hist(in_deg,  bins=50); plt.title("In-degree");  plt.xlabel("in-degree")
#savefig(FIGS/'fig_degree_hist.png')

# Sampled item→item graph
def sampled_directed_graph(P, pi, item_index, N=200, m_per_node=5, min_prob=0.0):
    top = np.argsort(-pi)[:N]; top_set = set(top.tolist())
    P_top = P[top, :]
    G = nx.DiGraph()
    for idx in top:
        G.add_node(int(idx), item=int(item_index[idx]), pi=float(pi[idx]))
    for li, src in enumerate(top):
        row = P_top[li, :]
        if row.nnz == 0: continue
        cols = row.indices; vals = row.data
        order = np.argsort(-vals)
        kept = 0
        for k in order:
            j = cols[k]
            if P[src, j] <= min_prob: continue
            if j in top_set:
                G.add_edge(int(src), int(j), w=float(P[src, j]))
                kept += 1
                if kept >= m_per_node: break
    return G

G = sampled_directed_graph(P, pi, item_index, N=200, m_per_node=5)
plt.figure(figsize=(9,8))
pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
pis = np.array([G.nodes[n]["pi"] for n in G.nodes()])
sz = 300 * (pis / pis.max() + 0.05)
nx.draw_networkx_nodes(G, pos, node_size=sz, alpha=0.85)
nx.draw_networkx_edges(G, pos, alpha=0.25, arrows=False)
lbl_nodes = sorted(G.nodes(), key=lambda n: G.nodes[n]["pi"], reverse=True)[:15]
labels = {n: str(G.nodes[n]["item"]) for n in lbl_nodes}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
plt.title("Sampled item→item graph (Top-N by π; edges by P_ij)")
plt.axis("off")
#savefig(FIGS/'fig_sampled_graph.png')

# Heatmap of P submatrix (top-k by π)
k = 40
idx = np.argsort(-pi)[:k]
P_block = P[idx, :][:, idx].toarray()
plt.figure(figsize=(6,5))
plt.imshow(P_block, aspect='auto')
plt.colorbar(label='P_ij')
plt.title(f"P submatrix (top {k} by π)")
plt.xlabel("j"); plt.ylabel("i")
savefig(FIGS/'fig_P_block_heatmap.png')

from numpy.linalg import norm
from scipy.sparse.linalg import expm_multiply
# choose subset for manageable expm and noise
M = 300
sub = np.argsort(-pi)[:M]
P_M = P[sub, :][:, sub]
Q_M = Q_rev[sub, :][:, sub]
pi_M = pi[sub] / pi[sub].sum()

p_sub = p_time.loc[:, p_time.columns.isin(sub)].copy()
p_sub = p_sub.reindex(columns=sub)  # ensure column order = sub
V_sub = V_time.loc[p_sub.index, sub].copy()

# basic utilities
def renorm_simplex(x, eps=1e-12):
    x = np.maximum(x, 0.0)
    s = x.sum()
    return x / s if s > eps else np.ones_like(x)/len(x)

def tv_distance(p, q):
    return 0.5 * np.abs(p - q).sum()

def kl_div(p, q, eps=1e-12):
    p = np.maximum(p, eps); q = np.maximum(q, eps)
    return float((p * (np.log(p) - np.log(q))).sum())

# parameters
DELTA_SEC = 24*60*60  # 1 day, Q is per-second if timestamps were seconds; if in ms adjust accordingly
ALPHA = 1.0           # potential tilt strength
NOISE_RANK = 10       # low-rank noise
SIGMA = 0.05          # noise scale
N_ENS = 10            # ensemble size

# precompute a symmetric operator for noise directions
S = 0.5 * (Q_M + Q_M.T)  # symmetric part
# take top-NOISE_RANK eigenvectors of -S (to bias towards diffusive modes)
try:
    import scipy.sparse.linalg as sla
    evals, evecs = sla.eigs(-S, k=min(NOISE_RANK, M-2))
    evecs = np.real(evecs)
except:
    # fallback: random orthonormal directions
    evecs = np.linalg.qr(np.random.randn(M, NOISE_RANK))[0]

dates = p_sub.index.to_list()
Tsteps = len(dates) - 1  # we compare t -> t+1
kl_ctmc, tv_ctmc = [], []
kl_det , tv_det  = [], []
kl_sto , tv_sto  = [], []
is_event = []

from scipy.sparse import csc_matrix
Q_M_csc = csc_matrix(Q_M)

for t in range(Tsteps):
    pt   = p_sub.iloc[t].values
    pt1  = p_sub.iloc[t+1].values
    Vnxt = V_sub.iloc[t+1].values

    # identify event windows by large L1 change
    is_ev = norm(pt1 - pt, 1) > np.quantile(np.abs(p_sub.diff().dropna().values).sum(axis=1), 0.9)
    is_event.append(is_ev)

    # 1) CTMC-only
    y_ctmc = expm_multiply((DELTA_SEC)*Q_M_csc, pt)

    # 2) Drift+potential (deterministic tilt)
    y_det = y_ctmc * np.exp(-ALPHA * Vnxt)
    y_det = renorm_simplex(y_det)

    # 3) Stochastic: add low-rank noise in evecs subspace around y_det
    ens = []
    for _ in range(N_ENS):
        z = np.random.randn(evecs.shape[1])
        noise = (evecs @ z) * SIGMA
        y = y_det + noise
        y = renorm_simplex(y)
        ens.append(y)
    y_sto = np.mean(np.vstack(ens), axis=0)
    y_sto = renorm_simplex(y_sto)

    # metrics
    kl_ctmc.append(kl_div(pt1, y_ctmc)); tv_ctmc.append(tv_distance(pt1, y_ctmc))
    kl_det .append(kl_div(pt1, y_det )); tv_det .append(tv_distance(pt1, y_det ))
    kl_sto .append(kl_div(pt1, y_sto )); tv_sto .append(tv_distance(pt1, y_sto ))

# Plot KL over time (+ event highlight)
tidx = np.arange(Tsteps)
plt.figure(figsize=(10,4))
plt.plot(tidx, kl_ctmc, label='CTMC-only')
plt.plot(tidx, kl_det,  label='Drift + potential (det)')
plt.plot(tidx, kl_sto,  label='Drift + potential + stochastic')
# shade event windows
ev = np.array(is_event, dtype=bool)
if ev.any():
    for x in tidx[ev]:
        plt.axvspan(x-0.5, x+0.5, alpha=0.08, color='red')
plt.title("KL divergence to next-day truth (shaded = event windows)")
plt.xlabel("t → t+1"); plt.ylabel("KL")
plt.legend()
#savefig(FIGS/'fig_kl_over_time.png')

# Pick an example 'event' day to visualize component-wise mass
if ev.any():
    ex = int(tidx[ev][0])
else:
    ex = Tsteps//2

true_next = p_sub.iloc[ex+1].values
pred_ctmc = expm_multiply((DELTA_SEC)*Q_M_csc, p_sub.iloc[ex].values)
pred_det  = renorm_simplex(pred_ctmc * np.exp(-ALPHA * V_sub.iloc[ex+1].values))
# one stochastic realization for visual variety (mean is smoother)
z = np.random.randn(evecs.shape[1])
noise = (evecs @ z) * SIGMA
pred_sto = renorm_simplex(pred_det + noise)

top_show = 20
order = np.argsort(-true_next)[:top_show]
labels = [str(item_index[sub[i]]) for i in order]
plt.figure(figsize=(11,4))
bw = 0.2; x = np.arange(top_show)
plt.bar(x - bw, true_next[order], width=bw, label='True next')
plt.bar(x,       pred_ctmc[order], width=bw, label='CTMC-only')
plt.bar(x + bw,  pred_det[order],  width=bw, label='Deterministic')
# stochastic as line markers to avoid clutter
plt.plot(x + bw*2, pred_sto[order], 'ko', markersize=3, label='Stochastic (one draw)')
plt.xticks(x, labels, rotation=70)
plt.title(f"Example day t→t+1 on top-{top_show} items (ex={ex})")
plt.ylabel("probability mass")
plt.legend(ncol=2, fontsize=8)
savefig(FIGS/'fig_example_day.png')

# Build a small LaTeX table with overall vs event-window averages
def avg(vals, mask=None):
    vals = np.array(vals, dtype=float)
    if mask is None or mask.sum()==0:
        return float(vals.mean())
    return float(vals[mask].mean())

table_tex = r"""
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Overall} & \multicolumn{2}{c}{Event windows} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Model & KL $\downarrow$ & TV $\downarrow$ & KL $\downarrow$ & TV $\downarrow$ \\
\midrule
CTMC-only & %.4f & %.4f & %.4f & %.4f \\
Drift + potential (det) & %.4f & %.4f & %.4f & %.4f \\
Drift + potential + stochastic & %.4f & %.4f & %.4f & %.4f \\
\bottomrule
\end{tabular}
""" % (
    avg(kl_ctmc), avg(tv_ctmc), avg(kl_ctmc, np.array(is_event)), avg(tv_ctmc, np.array(is_event)),
    avg(kl_det ), avg(tv_det ), avg(kl_det , np.array(is_event)), avg(tv_det , np.array(is_event)),
    avg(kl_sto ), avg(tv_sto ), avg(kl_sto , np.array(is_event)), avg(tv_sto , np.array(is_event))
)

# Save as a standalone PDF table via latex if available, otherwise write .tex for Overleaf
tex_path = FIGS/'metrics_table.tex'
tex_path.write_text(table_tex)
print("Wrote LaTeX table to:", tex_path)

# (Optional) If you want a compiled PDF from Colab, you'd need a latex engine.
# Simpler path: upload metrics_table.tex to Overleaf and include it inside a frame, or compile locally.