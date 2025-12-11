# --- Cell R1: lazy self-loop variant of Q_rev ---
from scipy import sparse
from scipy.sparse import dia_matrix

def make_lazy_Qrev(Q_rev, epsilon=0.05):
    # implicit P ≈ I + Δt * Q_rev for small Δt; laziness ≈ convex mix with I
    # Build a lazy generator by shrinking off-diagonals and adjusting diagonal to keep row-sum 0.
    Q = Q_rev.tocsr().copy()
    off = Q - dia_matrix((Q.diagonal(), 0), shape=Q.shape)
    Q_lazy = (1 - epsilon)*off  # shrink transitions
    diag = -np.asarray(Q_lazy.sum(axis=1)).ravel()
    Q_lazy = Q_lazy + dia_matrix((diag,0), shape=Q.shape)
    return Q_lazy.tocsr()

Q_lazy = make_lazy_Qrev(Q_rev, epsilon=0.05)

# Evaluate best config on Q_rev vs Q_lazy with the same V (e.g., V_time_w0)
def eval_Q(Qmat, V_alt, label):
    sub_idx = np.argsort(-pi)[:M_best]
    Q_M = Qmat[sub_idx,:][:,sub_idx]
    # reuse eval_with_V but swapping Q inside: quick inline shim
    return eval_with_V(V_alt)  # if you want, copy eval_with_V body and replace Q_M_c with this Q_M

# Quick comparison (you can reuse P2 by temporarily setting Q_rev = Q_lazy and calling eval_with_V)
print("Baseline Q_rev:"); _=eval_with_V(V_time_w0, "Q_rev")
Q_rev_backup = Q_rev
Q_rev = Q_lazy
print("Lazy Q_rev:   "); _=eval_with_V(V_time_w0, "Q_lazy")
Q_rev = Q_rev_backup

import os

DATA_DIR = "/content/drive/MyDrive/retailrocket"  # same as before

# minimal load
df = pd.read_csv(
    os.path.join(DATA_DIR, "events.csv"),
    usecols=["timestamp","event","itemid"],
    dtype={"timestamp": np.int64, "event": "category", "itemid": np.int64}
)

# match your modeling space (same top-K items you saved in item_index)
item_to_iid = pd.Series(np.arange(len(item_index), dtype=np.int64), index=pd.Index(item_index))
df = df[df["event"].isin(["view","addtocart"])].copy()
df = df[df["itemid"].isin(item_to_iid.index)].copy()
df["iid"] = item_to_iid.loc[df["itemid"]].values

# now compute 12H buckets on the SAME iid indexing
p_time_12h = compute_time_buckets(df, bucket="12H")
V_time_12h = rolling_popularity_V(p_time_12h, window=14)
V_time_12h = V_time_12h.sub(V_time_12h.mean(axis=1), axis=0)

#new R2
p_time_backup, V_time_backup = p_time, V_time

p_time, V_time = p_time_12h, V_time_12h
V_time_w0_12h = V_time.sub(V_time.mean(axis=1), axis=0)

print("12H buckets:")
_ = eval_with_V(V_time_w0_12h, "12H V_pop")

p_time, V_time = p_time_backup, V_time_backup


# --- Cell R3: compare raw Q vs reversible Q_rev ---
Q_M_raw   = Q[np.argsort(-pi)[:M_best], :][:, np.argsort(-pi)[:M_best]]
Q_M_rev   = Q_rev[np.argsort(-pi)[:M_best], :][:, np.argsort(-pi)[:M_best]]

# Minimal eval swapping generator:
def eval_with_Q(QM, label):
    sub_idx = np.argsort(-pi)[:M_best]
    p_sub = p_time.loc[p_time.index[-LAST_days-1:], p_time.columns.isin(sub_idx)].reindex(columns=sub_idx)
    V_sub = V_time.loc[p_sub.index, sub_idx]
    from scipy.sparse import csc_matrix
    Q_M_c = csc_matrix(QM)
    # … copy body of eval_with_V and replace Q_M_c with Q_M_c, leaving V_sub as is …
    # (To keep this short here, you can reuse eval_with_V by temporarily setting a global Q_rev = QM.)

print("Raw Q:")
Q_rev_backup = Q_rev
Q_rev = Q   # reuse eval_with_V machinery
_ = eval_with_V(V_time, "raw Q")
print("Reversible Q_rev:")
Q_rev = Q_rev_backup
_ = eval_with_V(V_time, "Q_rev")
