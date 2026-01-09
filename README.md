# Retailrocket-wasserstein-flows

*Modeling noisy preference dynamics on item graphs with CTMC + Wasserstein drift–diffusion*

This project models how aggregate user attention flows over an e-commerce catalog. From Retailrocket clickstreams we build an item→item graph, estimate a CTMC generator with exposure-aware rates, construct a reversible operator compatible with Wasserstein geometry, and compare deterministic vs stochastic drift–diffusion forecasts of next-day distributions. The repo includes data prep, sparse graph/matrix construction, EDA plots, forecasting code (CTMC / deterministic tilt / low-rank noisy), and evaluation via KL/TV with event-window analysis.

This repo contains the code and experiments for a course project on **graph analysis + matrix computation** using the **Retailrocket** e-commerce dataset.

**Core idea**
Rather than only predicting a user’s *next* item, we model how **aggregate user attention**—a probability distribution over items—**flows over time** on an item→item graph. Concretely, we:

1. Build an **item transition graph** from real clickstream sessions.
2. Estimate a **continuous-time Markov chain (CTMC)** generator from transitions and exposure (dwell) times.
3. Construct a **reversible** generator compatible with **Wasserstein (optimal transport) geometry**.
4. Define a **free energy** combining entropy + popularity-based potential.
5. Compare:

   * a **deterministic** Wasserstein-style gradient-flow step, and
   * a **stochastic** drift–diffusion step with **geometry-aligned** low-rank noise,

on their ability to predict **short-horizon catalog distributions** (p_{t+\Delta}).

This is a **matrix-computation-friendly**, theoretically grounded, and data-driven project connecting:

* Markov chains / CTMCs & sparse operators
* Graphs and Laplacian-style geometry
* Wasserstein gradient flows
* Drift–diffusion (Langevin-like) dynamics
* Real clickstream data

---

## 1) Problem motivation

Most recommendation work focuses on:

* **Next-item ranking** for a specific user/session, or
* **Static similarity/embeddings** (MF, GNNs, etc.).

Here we ask a different question:

> Given the current **catalog-level** attention distribution (p_t) (probability mass over the whole item set),
> can we forecast the **next-day distribution** (p_{t+\Delta}) with a **continuous-time**, **mass-conserving**, and **geometry-aware** model?

Why this matters:

* **Population view**: see which items/categories gain or lose attention and how fast.
* **Interpretability**: rates, energy, and noise all have clear meanings.
* **Linear-algebraic tooling**: everything is sparse-matrix-friendly and explainable.

> **Terms**
> **Item** = one product (a node).
> **Catalog** = the entire set of modeled items (e.g., top-K; (K{=}5{,}000) here).
> **Event windows** = volatile days where (|p_{t+1}-p_t|_1) is unusually large (90th percentile); these stress-test spike tracking.

---

## 2) Conceptual outline (in simple terms)

### 2.1 Item graph (sessions → transitions)

* Keep events: `view` + `addtocart` (primary run).
* **Sessionize** per visitor with a **30-min** inactivity timeout.
* Within a session, if the item changes (i!\to!j), count a transition (C_{ij}{+}{=}1).
* If the item doesn’t change (e.g., `view(A)→addtocart(A)`), treat it as **extra time on A** (exposure), not a new edge.
* Restrict to **top-K** items (e.g., 5k) for tractability.

Outputs:

* **Counts matrix** (C): how often we move (i!\to!j).
* **Exposure** (T_i): total (capped) dwell time on item (i).
* **Row-stochastic** (P) via (P_{ij} = C_{ij}/\sum_j C_{ij}) (with a lazy self-loop fix for zero-out rows).

### 2.2 CTMC generator (Q)

* Off-diagonals: (Q_{ij} = C_{ij}/T_i) for (i\neq j); diagonals (Q_{ii}=-\sum_{j\neq i} Q_{ij}).
* Properties: off-diagonals (\ge0), rows sum to 0.
* Dynamics: (\frac{d}{dt}p_t = p_t Q).

### 2.3 Stationary distribution & reversibilization

* Compute stationary ( \pi ) (long-run attention).
* Build a **reversible** generator:
  [
  Q_{\text{rev}}=\tfrac12\big(Q+\Pi^{-1}Q^\top\Pi\big),\quad \Pi=\mathrm{diag}(\pi)
  ]
  which satisfies detailed balance and aligns with discrete OT/Wasserstein geometry.

### 2.4 Free energy & potentials

* Free energy:
  [
  \mathcal F(p,t)=\sum_i p_i\log p_i + \sum_i V_i(t),p_i
  ]
  where (V_i(t)) captures **attractiveness** signals (we use **rolling popularity**; optional: price/discount).

---

## 3) Models we compare

Let (\widehat p^{\text{ctmc}} = p_t,\mathrm{expm}(\Delta Q_{\text{rev}})).

1. **CTMC-only (no potential)**
   Structure-aware smoothing along the item graph.

2. **Deterministic (free-energy step)**
   Tilt toward the next day’s potential:
   [
   \widehat p^{\text{det}} \propto \widehat p^{\text{ctmc}}\cdot \exp(-\alpha V_{t+\Delta})
   ]
   then renormalize. Intuition: “gently follow what’s likely to be popular tomorrow.”

3. **Stochastic (low-rank fluctuations)**
   Add noise along a few **diffusive graph directions** (top eigenspace of the symmetric part of (Q_{\text{rev}})):
   [
   \widehat p^{\text{sto}}=\mathbb E\big[\mathrm{renorm}\big(\widehat p^{\text{det}}+\sigma,U z\big)\big],\quad z\sim\mathcal N(0,I)
   ]
   Intuition: **catalog shocks** (promos/banner placement) often move mass along **category-level modes**; low-rank noise helps track spikes.

**Event windows** (volatile days) are flagged by large (|p_{t+1}-p_t|_1).
We report metrics **overall** and **restricted to event windows**.

---

## 4) Results (Retailrocket, K=5k; daily buckets)

**Hyper-parameter sweep (subset (M{=}150), last 60 days)**
Grid: (\alpha\in{0.5,1.0}), (\sigma\in{0,0.03}), rank (=5), ensemble (=3).

**Best config:** (\alpha=0.5), (\sigma=0.03), rank (=5), (n_{\text{ens}}=3)

* **KL overall:** **0.0860**
* **KL (events):** 0.1507
* **TV overall:** 0.4889
* **TV (events):** 0.4891

**Ablation (same split)**

* **CTMC-only:** worst KL (over-smooths spikes).
* **Deterministic (tilt):** improved KL vs CTMC; follows baseline popularity.
* **Stochastic:** **best KL** overall and in event windows; small but consistent TV gain.
* **No potential (α=0, same noise):** similar to stochastic but slightly worse on KL—shows the tilt helps.

**Significance (Wilcoxon, sto vs det)**

* KL: (p=2.8\times10^{-11}), median (\Delta)KL = **−0.0495**
* TV: (p=2.6\times10^{-11}), median (\Delta)TV = **−0.0022**
* Events only (KL): (p=0.031) → improvement persists on volatile days.

> **Interpretation**
> Tilt = **trend** (toward baseline popularity).
> Low-rank noise = **spikes** (moves mass along meaningful graph directions).
> Together they improve short-horizon catalog forecasting, especially when the catalog “gusts.”

---

## 5) Visualization (EDA & graph)

The notebooks produce presentation-ready figures:

* **Top-K stationary mass** (long-run hubs).
* **Degree histograms** (sparse, hub-heavy).
* **Exposure time distribution** (T_i).
* **Top-10 items over time** (p_t(i)).
* **Sampled item→item subgraph** (NetworkX + pyvis).
* **Heatmap of (P) submatrix** (blocks & hub stripes).

> See `figures/` outputs or generate via the notebooks.

---

## 6) Stage-aware extension (optional)

Primary run: **nodes = items**.
Extension: **nodes = (item, stage)** with stage ∈ {view, addtocart, transaction}.
This treats `view(A)→addtocart(A)` as a genuine transition and enables **funnel dynamics** with the same CTMC/Wasserstein machinery.

---

## 7) Quickstart / Reproducibility

### Data & environment

* Retailrocket CSVs under `data/retailrocket/` (or your Google Drive path).
* Python ≥ 3.10; install deps:

  ```bash
  pip install numpy pandas scipy networkx pyvis matplotlib tqdm pyarrow
  ```

### Colab workflow (recommended)

1. Mount Drive and set paths:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   DATA_DIR = "/content/drive/MyDrive/retailrocket"              # raw CSVs
   ART_DIR  = "/content/drive/MyDrive/retailrocket_artifacts"    # saved artifacts
   ```
2. Run notebooks in order:

   * `01_data_wrangling.ipynb` → events filtering, sessionization, top-K.
   * `02_graph_and_matrices.ipynb` → build `C,P,T,pi,Q,Q_rev` (saves to `ART_DIR`).

     * **Gotcha fixes included**

       * Timestamps: detect **ms vs s** for dwell consistency.
       * **Row-stochastic P**: add **lazy self-loops** for zero-out rows (Option A).
   * `03_visualizations.ipynb` → EDA/graphs.
   * `04_models_and_metrics.ipynb` → CTMC/deterministic/stochastic forecasts, KL/TV, event windows, Wilcoxon.
   * `05_stage_aware_extension.ipynb` (optional) → (item,stage) graph.

Artifacts persist in `retailrocket_artifacts/` (split files: `P.npz`, `C.npz`, `T.npy`, `pi.npy`, `Q.npz`, `Q_rev.npz`, `p_time.parquet`, `V_time.parquet`, `item_index.csv`), so you don’t have to recompute.

---

## 8) Repository structure (suggested)

```
.
├─ data/
│  └─ retailrocket/                 # raw CSVs (not tracked)
├─ artifacts/
│  └─ retailrocket_artifacts/       # saved P, C, T, pi, Q, Q_rev, p_time, V_time, etc.
├─ figures/                         # generated plots/tables
├─ notebooks/
│  ├─ 01_data_wrangling.ipynb
│  ├─ 02_graph_and_matrices.ipynb
│  ├─ 03_visualizations.ipynb
│  ├─ 04_models_and_metrics.ipynb
│  └─ 05_stage_aware_extension.ipynb
├─ src/
│  ├─ sessionize.py
│  ├─ build_graph.py
│  ├─ ctmc.py
│  ├─ potentials.py
│  ├─ simulate.py
│  ├─ eval.py
│  └─ viz.py
├─ requirements.txt
├─ README.md
└─ LICENSE
```

---

## 9) Practical notes & gotchas

* **Timestamp units**: Retailrocket events are **ms**; make dwell (T_i) unit-aware (convert ms→s) so (Q) is meaningful.
* **Zero-out rows in (C)**: items with no outgoing transitions break row-stochastic (P). We use **lazy self-loops** to fix: if a row sum is zero, set (P_{ii}=1).
* **Event windows**: defined by the **90th percentile** of day-to-day (\ell_1) change; tune as needed.
* **TV scale quirk**: an early CTMC plotting cell reported TV on a different absolute scale; **KL** comparisons are the primary headline (ordering is consistent).

---

## 10) Extension ideas

* **Price/discount potential**: blend a discount-based (V^{\text{price}}) with popularity (V^{\text{pop}}):
  (V = (1-w),V^{\text{pop}} + w,V^{\text{price}}), (w\in[0,1]).
* **Adaptive noise**: increase (\sigma) when (|p_t-p_{t-1}|_1) is large (anticipate events).
* **Different aggregation**: compare daily vs 12-hour buckets.
* **Topology variants**: compare raw (Q) vs (Q_{\text{rev}}) to show reversibilization stability.

---

## 11) License / citation

Add your preferred license and citation line here.

> If you use this code or ideas, please cite this repo and the Retailrocket dataset.

---

**Contact / Questions**
Open an issue or PR with questions, suggestions, or improvements.
