# Retailrocket-wasserstein-flows
_Modeling noisy preference dynamics on item graphs with CTMC + Wasserstein drift–diffusion_

This repo contains the code and experiments for a course project on **graph analysis + matrix computation** using the **Retailrocket** e-commerce dataset.

**Core idea:**  
Instead of only predicting the *next* item (as in standard recommender systems), we model how **aggregate user attention** (a probability distribution over items) **flows over time** on an item–item graph. We:

1. Build an **item transition graph** from real clickstream sessions.
2. Estimate a **continuous-time Markov chain (CTMC)** generator from transitions and exposure times.
3. Construct a **reversible** generator compatible with **Wasserstein (optimal transport) geometry**.
4. Define a **free energy** combining entropy + popularity-based potential.
5. Compare:
   - a **deterministic Wasserstein gradient flow** model, and  
   - a **stochastic drift–diffusion** model with geometry-aligned noise,
   
   on their ability to predict **short-horizon item distributions**.

This is meant as a **matrix-computation-friendly**, theoretically grounded, and data-driven project that connects:
- Markov chains / CTMCs  
- Graphs & sparse matrices  
- Wasserstein gradient flows  
- Drift–diffusion / Langevin dynamics  
- Real clickstream data

---

## 1. Problem motivation

Most recommender work focuses on:

- **Next-item ranking for a specific user**, or
- **Static similarity / embeddings** (MF, GNNs, etc.).

Here we ask a different question:

> Given the current global distribution of user attention over items \(p_t\),  
> can we model and forecast how this **distribution** will evolve to \(p_{t+\Delta}\),  
> using a **continuous-time**, **mass-conserving**, and **geometry-aware** model?

Why this is interesting:

- It provides a **population-level view**: which items gain/lose attention mass and how fast.
- It’s **interpretable**: every part (rates, energy, noise) has a clear meaning.
- It leverages **graph + matrix tools** (sparse operators, eigen-decompositions) in a non-trivial, real-world setting.

---

## 2. Conceptual outline (in simple terms)

### 2.1 Item graph

From the Retailrocket logs:

- We filter to `view` + `addtocart` (primary run).
- We **sessionize** per user with a **30-minute inactivity timeout**.
- Inside each session, we look at consecutive events:
  - If the item changes \(i \to j\), we record an **item→item transition**.
  - If the item stays the same (e.g., `view(A) → addtocart(A)`), we **do not** add an edge; we treat it as extra **time spent on A**.

We restrict to **top-K items** to keep matrices tractable.

This yields:

- **Counts matrix** `C`: `C[i,j]` = # of times we saw \(i\to j\).
- A directed, weighted item→item graph.

### 2.2 Exposure time (sojourn time)

For each item \(i\), we estimate how long users collectively **stay** at \(i\):

- For each within-session pair `(event_t, event_{t+1})` at items `(i, j)`:
  - Add dwell time \(\Delta t = \min(\text{timestamp}_{t+1} - \text{timestamp}_t,\; \text{cap})\) to item \(i\).
- No dwell is added after the last event in a session.
- Same-item consecutive events contribute to **exposure**, not transitions.

We store this as vector **`T`**, where `T[i]` is total exposure time at item \(i\).

> Intuition: `C[i,j]` says *how often* we jump from `i` to `j`,  
> `T[i]` says *how long* we spend at `i`.  
> `C/T` gives **continuous-time jump rates**.

### 2.3 CTMC generator \(Q\)

We estimate a **continuous-time Markov chain (CTMC)**:

- For \(i\neq j\): \(Q_{ij} = C_{ij} / T_i\)
- \(Q_{ii} = -\sum_{j\neq i} Q_{ij}\)

Properties:

- Off-diagonals ≥ 0
- Row sums = 0

This defines the **infinitesimal dynamics**:

\[
\frac{d}{dt} p_t = p_t Q.
\]

If you simulate this CTMC, you get one candidate model of how attention flows.

### 2.4 Stationary distribution and “long-run attention” \( \pi \)

We compute the **stationary distribution** \( \pi \) from `P` (or equivalently from `Q`):

- \( \pi^\top P = \pi^\top \), \( \sum_i \pi_i = 1 \)

Interpretation:

- \( \pi_i \) = long-run **share of attention** on item \(i\)
- This is similar to PageRank without teleportation: a principled “importance” score.

We use \( \pi \) both:
- as a baseline “equilibrium attention profile”, and
- to build a **reversible** generator.

### 2.5 Reversible generator \(Q_\text{rev}\)

To align with discrete Wasserstein gradient-flow theory, we form:

\[
Q_{\text{rev}} = \tfrac12 \big( Q + \Pi^{-1} Q^\top \Pi \big),\quad \Pi = \text{diag}(\pi),
\]

which satisfies **detailed balance**:

\[
\pi_i Q_{\text{rev},ij} = \pi_j Q_{\text{rev},ji}.
\]

This gives us a **geometry-compatible** generator we can use as the backbone of drift and diffusion.

---

## 3. Free energy, drift, and noise

We model the evolving **distribution of attention** \(p_t\) on items.

### 3.1 Free energy

Define a **free energy functional**:

\[
\mathcal F(p, t) = \sum_i p_i \log p_i \;+\; \sum_i V_i(t)\, p_i.
\]

- First term: **entropy** (prefers spread-out distributions).
- Second term: **potential** \(V_i(t)\), built from **rolling popularity** etc.
  - Example: \(V_i(t) \approx -\log(\text{recent frequency of item } i)\).

### 3.2 Deterministic Wasserstein drift

We think of the dynamics as a **gradient flow of \(\mathcal F\)** under a Wasserstein-type geometry on the graph:

\[
\dot p_t = K\, \nabla_p \mathcal F(p_t, t),
\]

where \(K\) is an operator derived from the graph / \(Q_{\text{rev}}\).

In practice:

- We use \(Q_{\text{rev}}\) or a Laplacian-like matrix as the **drift operator**.
- We simulate:

  \[
  \dot p_t = p_t Q_{\text{rev}} + \text{(correction from } V(t)\text{)},
  \]
  
  with a small time step and projection back to the simplex.

This gives a **smooth**, interpretable evolution: attention flows along likely edges and towards attractive items.

### 3.3 Stochastic drift–diffusion (fluctuations)

To capture **spikes, bursts, and volatility**, we add **noise**:

\[
dp_t = \underbrace{K\,\nabla_p \mathcal F(p_t, t)}_{\text{drift}}\,dt
     + \underbrace{\sigma\, B\, dW_t}_{\text{diffusion}},
\]

- \(dW_t\): Brownian increments (random shock).
- \(B\): a **low-rank** matrix built from leading eigenvectors of a symmetric operator (e.g. from \(Q_{\text{rev}}\)), so noise lives along **meaningful graph directions**.
- \(\sigma\): noise strength.

We simulate this with **Euler–Maruyama** and project back into the probability simplex.

**Interpretation:**  
The deterministic part explains **trend**; the stochastic part explains **random but structured fluctuations**, especially around events (promotions, new items, etc.).

---

## 4. What we actually compute & compare

For each time bucket \(t\) (e.g. daily):

- `p_time[t]` = empirical distribution over items at day \(t\).
- `V_time[t]` = potential for day \(t\).

We compare several one-step-ahead predictors for \(p_{t+\Delta}\):

1. **CTMC baseline:**  
   \(\hat p_{t+\Delta} = p_t \, e^{\Delta Q_{\text{rev}}}\)
2. **Deterministic drift (energy-based):**  
   integrate \(\dot p = K \nabla \mathcal F\) using `V_time`.
3. **Stochastic drift–diffusion:**  
   simulate the SDE multiple times, average, compare.

**Metrics:**

- **KL divergence** \( \mathrm{KL}(p_{t+\Delta} \,\|\, \hat p_{t+\Delta}) \)
- **Total variation (TV)** distance \( \frac12 \|p_{t+\Delta} - \hat p_{t+\Delta}\|_1 \)
- **Top-k mass error** (how much probability mass we mis-allocate among the most important items)

We also look at:

- Normal days vs **event windows** (bursty days)  
  → check if the stochastic model better explains spikes.

---

## 5. Visualization

The repo includes code/notebooks to produce:

- **Top-K stationary mass plot**:
  - shows “long-run attention” hubs (high-\(\pi\) items).
- **In/out degree histograms**:
  - show sparsity + hub structure of the item graph.
- **Exposure time distribution**:
  - sanity check for CTMC estimation.
- **Top-10 items’ time series**:
  - visualize attention bursts and trends.
- **Sampled item→item graph**:
  - top-\(\pi\) nodes with strongest edges (NetworkX + pyvis).
- **Heatmap of a \(P\) submatrix**:
  - reveals community/block structure among important items.

These figures are designed to be **presentation-ready** to explain both:
1. The data/graph structure.
2. Why a drift–diffusion model is reasonable.

---

## 6. Stage-aware extension (optional)

Primary run: **nodes = items**.

Extension: **nodes = (item, stage)**, where stage ∈ {view, addtocart, transaction}.

- Now `view(A) → addtocart(A)` **is** a transition between states.
- You can rebuild `C`, `P`, `T`, `Q`, `Q_rev` on this extended state space.
- This lets you analyze **funnel dynamics** (how attention flows through stages) with the same tools.

---

## 7. Repository structure

Suggested layout (adapt as your code evolves):

```text
.
├─ data/
│  └─ retailrocket/                 # raw CSVs (not tracked)
├─ artifacts/
│  └─ retailrocket_artifacts/       # saved P, C, T, pi, Q, Q_rev, p_time, V_time, etc.
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
