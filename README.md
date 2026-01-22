# Retailrocket-wasserstein-flows

*Modeling noisy preference dynamics on item graphs with CTMC + Wasserstein drift–diffusion*

This project models how **aggregate user attention** flows over an e-commerce catalog.  
From Retailrocket clickstreams we:

- build an **item→item graph**;
- estimate a **continuous-time Markov chain (CTMC)** generator with exposure-aware jump rates;
- construct a **reversible** generator compatible with Wasserstein geometry;
- define a **free energy** combining entropy and popularity-based potentials; and
- compare **CTMC-only**, **deterministic drift**, and **stochastic drift–diffusion** forecasts of **next-day catalog distributions**.

The repo contains:

- data prep and sessionization,
- sparse graph & matrix construction,
- EDA plots,
- forecasting code (CTMC / deterministic tilt / low-rank noisy),
- evaluation via KL/TV including **event-window** analysis, and
- simple robustness & feature experiments.

This is a course project in **graph analysis + matrix computation** using the **Retailrocket** e-commerce dataset.

---

## 1) Problem: catalog-level attention flows

Most recommendation methods focus on:

- **next-item prediction** for a particular user/session, or
- **static similarity/embeddings** (MF, GNNs, etc.).

Here we ask a different question:

> Given today’s **catalog-level** attention distribution $`p_t`$ (probability mass over items),
> can we forecast the **next-day distribution** $`p_{t+\Delta}`$ (with $`\Delta = 1`$ day)
> using a **continuous-time**, **mass-conserving**, and **geometry-aware** model?

Why this matters:

- **Population view**: see which items/categories gain or lose attention and how fast.
- **Interpretability**: rates, “energy”, and noise have clear meanings.
- **Linear-algebra friendly**: everything is sparse-matrix based.

> **Terms**  
> **Item**: one product (a node).  
> **Catalog**: the modeled item set (e.g., top-$`K`$, with $`K=5{,}000`$ here).  
> **Event windows**: days where $`\lVert p_{t+1} - p_t \rVert_1`$ exceeds the 90th percentile; used to stress-test models on **volatile** periods.

---

## 2) Conceptual outline

### 2.1 From clickstreams to an item graph

Raw data: `events.csv` with  
`(timestamp, visitorid, event ∈ {view, addtocart}, itemid)`.

1. **Sessionization**

   - Sort by `(visitorid, timestamp)`.
   - Start a new session if inactivity $`> 30`$ minutes.
   - Within each session, obtain an ordered sequence of items  
     $`i_1 \to i_2 \to \cdots \to i_L`$.

2. **Counts & exposure**

   For each consecutive **change of item** $`i\to j`$ inside a session:

   - increment a transition count $`C_{ij}`$;
   - if the item stays the same (e.g., `view(A) → addtocart(A)`), treat this as **extra dwell time** on $`i`$.

3. **Top-$`K`$ restriction**

   - Keep the $`K`$ most frequently interacted items (here $`K=5000`$);
   - Reindex them as nodes $`i \in \{1,\dots,K\}`$.

Result: a sparse directed item graph with weighted adjacency matrix $`C = (C_{ij})`$.

### 2.2 Discrete random walk: transition matrix $`P`$

Row-stochastic transition matrix:
```math
P_{ij} =
\begin{cases}
\dfrac{C_{ij}}{\sum_{j'} C_{ij'}}, & \text{if } \sum_{j'} C_{ij'} > 0, \\
0, & \text{otherwise}.
\end{cases}
```

- From item $`i`$, the probability of jumping to neighbor $`j`$ in one step is $`P_{ij}`$.
- We add **lazy self-loops** for rows with zero outgoing counts so that every row of $`P`$ sums to 1.

### 2.3 CTMC generator $`Q`$

We estimate a **continuous-time** generator from transition counts and total exposure time:

- Dwell time between two events in a session:
  ```math
  \Delta t = \text{timestamp}_{t+1} - \text{timestamp}_t,
  ```
  capped at 10 minutes and converted to seconds.
- Aggregate per item:
  ```math
  T_i = \sum_{\text{visits to } i} \Delta t.
  ```

Then define the generator $`Q`$:
```math
Q_{ij} = \frac{C_{ij}}{T_i} \quad (i\ne j),
\qquad
Q_{ii} = -\sum_{j\ne i} Q_{ij}.
```

Properties:

- off-diagonals $`Q_{ij} \ge 0`$;
- each row of $`Q`$ sums to 0;
- the CTMC dynamics satisfy
  ```math
  \frac{d}{dt} p_t = p_t Q.
  ```

$`Q`$ is a **graph Laplacian** for a continuous-time random walk on the directed item graph.

### 2.4 Stationary distribution and reversible $`Q_{\text{rev}}`$

We approximate a stationary distribution $`\pi`$ such that
```math
\pi^\top P = \pi^\top, \qquad \sum_i \pi_i = 1.
```

Interpretation: $`\pi_i`$ is a **long-run attention weight** or centrality score for item $`i`$.

To make the dynamics compatible with discrete Wasserstein geometry, we use a **reversible symmetrization**:
```math
Q_{\text{rev}}
  = \frac12 \left( Q + \Pi^{-1} Q^\top \Pi \right),
\qquad
\Pi = \mathrm{diag}(\pi).
```

- $`Q_{\text{rev}}`$ is a generator of a **reversible** Markov process with stationary distribution $`\pi`$.
- The symmetric part
  ```math
  S = \frac12 \left( Q_{\text{rev}} + Q_{\text{rev}}^\top \right)
  ```
  is negative semidefinite and plays the role of a (weighted) graph Laplacian in the Onsager/Wasserstein picture.

### 2.5 Daily distributions and popularity potential

We aggregate events into **daily buckets**:

- For each day $`t`$ and item $`i`$, let $`n_t(i)`$ be the event count.
- Define the daily attention distribution
  ```math
  p_t(i) = \frac{n_t(i)}{\sum_j n_t(j)}.
  ```

We smooth popularity with a rolling window of length $`w`$ (here $`w=7`$):
```math
\bar p_t(i) = \frac1w \sum_{s=t-w+1}^t p_s(i).
```

Define a **popularity potential** (energy landscape):
```math
V_t(i) = -\log\big(\varepsilon + \bar p_t(i)\big),
```
where $`\varepsilon > 0`$ is small for numerical stability.

- Recently popular items: $`\bar p_t(i)`$ large $`\Rightarrow V_t(i)`$ small (low-energy wells).
- Cold items: $`\bar p_t(i)`$ small $`\Rightarrow V_t(i)`$ large (high energy).

### 2.6 Free energy

We use a simple **free-energy functional** combining entropy and potential:
```math
\mathcal{F}(p, t) = \sum_i p_i \log p_i + \sum_i V_t(i)\, p_i.
```

Later we track how $`\mathcal{F}`$ changes under our update rules.

---

## 3) Forecasting models

We want to predict **next-day catalog distributions**:
given $`p_t`$, forecast $`p_{t+\Delta}`$ with $`\Delta = 1`$ day.

We restrict to the top-$`M`$ items by $`\pi`$ (here typically $`M = 150`$) and work with the $`M\times M`$ submatrix of $`Q_{\text{rev}}`$ and the corresponding slices of $`p_t, V_t`$.

Denote the CTMC one-step evolution (applied to a row vector) by
```math
\hat p^{\text{ctmc}} = p_t \exp(\Delta Q_{\text{rev}}),
```
evaluated via sparse `expm_multiply`.

### Model 1: CTMC-only baseline

- **Dynamics only**, no potential:
  - Predict using $`\hat p^{\text{ctmc}}`$ directly.
- Intuition: a **structure-aware diffusion** along the item graph that smooths attention.

### Model 2: Deterministic drift with potential (trend)

We apply a **Boltzmann tilt** toward the next day’s potential:
```math
\tilde p_i
  = \hat p^{\text{ctmc}}_i \exp\big(-\alpha\, V_{t+\Delta}(i)\big),
\qquad
\hat p^{\text{det}}_i
  = \frac{\tilde p_i}{\sum_j \tilde p_j},
```
where $`\alpha \ge 0`$ controls how strongly we follow the popularity landscape.

- Intuition: “gently follow what is expected to be popular tomorrow.”

### Model 3: Stochastic low-rank drift–diffusion (fluctuations)

Take the symmetric part
```math
S = \frac12\left(Q_{\text{rev}} + Q_{\text{rev}}^\top\right),
```
compute the top $`r`$ eigenvectors of $`-S`$, and stack them as columns in $`U \in \mathbb{R}^{M\times r}`$.

For each forecast:

1. Start from $`\hat p^{\text{det}}`$.
2. Sample $`z \sim \mathcal{N}(0, I_r)`$.
3. Add **graph-aware noise** and renormalize:
   ```math
   \hat p^{\text{sto}} = \mathrm{renorm}\big(\hat p^{\text{det}} + \sigma U z\big).
   ```
4. Average several such draws (ensemble of size $`n_{\text{ens}}`$).

- Intuition: low-rank noise moves probability along **smooth diffusion modes** of the graph, modeling catalog-level shocks (promotions, shifts between related items).

### Event windows and metrics

We evaluate predictions $`q_{t+\Delta}`$ against the true $`p_{t+\Delta}`$ using:

- **Kullback–Leibler divergence**  
  ```math
  \mathrm{KL}(p\|q) = \sum_i p_i \log\frac{p_i}{q_i},
  ```
- **total variation**  
  ```math
  \mathrm{TV}(p,q) = \frac12 \sum_i |p_i - q_i|.
  ```

We report metrics:

- **overall** (all days), and
- **event windows** (days where $`\lVert p_{t+1} - p_t\rVert_1`$ is in the top 10%).

Event windows highlight how well models track **spikes and partial reversals**.

---

## 4) Results on Retailrocket (daily buckets, $`K=5000`$)

### 4.1 Hyper-parameter sweep

We swept a small grid over $`(\alpha,\sigma)`$ while fixing:

- $`M = 150`$,
- last-days window = 60,
- noise rank $`r = 5`$,
- ensemble size $`n_{\text{ens}} = 3`$.

Grid:
- $`\alpha \in \{0.5, 1.0\}`$ (potential strength),
- $`\sigma \in \{0.00, 0.03\}`$ (stochastic fluctuation level).

Summary (overall vs event windows):

| cfg | $`M`$ | last_days | $`\alpha`$ | $`\sigma`$ | rank | $`n_{\text{ens}}`$ | KL\_overall | TV\_overall | KL\_events | TV\_events |
|----:|----:|----------:|---------:|---------:|-----:|-----------------:|------------:|------------:|-----------:|-----------:|
| 0 | 150 | 60 | 0.5 | 0.00 | 5 | 3 | 0.136178 | 0.491053 | 0.210264 | 0.491561 |
| 1 | 150 | 60 | 0.5 | 0.03 | 5 | 3 | **0.086032** | **0.488883** | **0.150696** | **0.489127** |
| 2 | 150 | 60 | 1.0 | 0.00 | 5 | 3 | 0.141490 | 0.491134 | 0.218928 | 0.491872 |
| 3 | 150 | 60 | 1.0 | 0.03 | 5 | 3 | 0.087050 | 0.488844 | 0.159091 | 0.489277 |

**Best config:**  
$`\alpha = 0.5`$, $`\sigma = 0.03`$, rank $`r=5`$, $`n_{\text{ens}}=3`$ (cfg 1).

- Provides the lowest KL overall and competitive KL on event windows.
- TV differences across configs are small but consistent.

### 4.2 Model comparison and ablations

Using the best $`(\alpha,\sigma)`$, we re-evaluated four variants:

| Model | KL\_all | TV\_all | KL\_ev | TV\_ev |
|-------|--------:|--------:|-------:|-------:|
| **CTMC-only** | 0.231329 | 0.019054 | 0.307608 | 0.025330 |
| **Deterministic** (tilt, $`\sigma=0`$) | 0.136178 | 0.491053 | 0.210264 | 0.491561 |
| **Stochastic** ($`\alpha=0.5,\sigma=0.03`$) | **0.087401** | 0.488920 | **0.149298** | 0.488901 |
| **No potential** ($`\alpha=0,\sigma=0.03`$) | 0.087514 | **0.488915** | 0.147759 | **0.488874** |

Takeaways:

- **CTMC-only**: worst KL — diffusion alone **over-smooths spikes**.  
  (TV was computed on a different absolute scale in this early cell, so we treat KL as the main headline here.)
- **Deterministic tilt**: big KL improvement over CTMC; it tracks baseline popularity trends.
- **Stochastic drift–diffusion**: best overall KL and competitive TV; gives clear gains vs deterministic on both all days and event windows.
- **No potential (same noise)**: close to stochastic but slightly worse in KL — the popularity **tilt genuinely helps**.

### 4.3 Statistical significance (stochastic vs deterministic)

We compared the stochastic model to the deterministic tilt **day by day**, forming paired differences

- $`\Delta \mathrm{KL} = \mathrm{KL}_{\text{sto}} - \mathrm{KL}_{\text{det}}`$,
- $`\Delta \mathrm{TV} = \mathrm{TV}_{\text{sto}} - \mathrm{TV}_{\text{det}}`$.

Wilcoxon signed-rank tests:

| Test | statistic | $`p`$-value | median difference |
|------|-----------|-----------|-------------------|
| KL (all days) | 3.0 | $`2.79\times 10^{-11}`$ | $`\mathrm{median}\,\Delta\mathrm{KL} = -0.0495`$ |
| TV (all days) | 1.5 | $`2.58\times 10^{-11}`$ | $`\mathrm{median}\,\Delta\mathrm{TV} = -0.0022`$ |
| KL (event days) | — | $`0.0312`$ | $`\mathrm{median}\,\Delta\mathrm{KL} = -0.0533`$ |

Interpretation:

- The stochastic model **consistently** improves KL and TV (negative medians, extremely small $p$-values).
- The improvement is still visible when restricted to **event windows** (bursty days).

### 4.4 Onsager view and energy diagnostics

On the chosen subset ($`M=150`$) we looked at **energy-like diagnostics** under the deterministic step:

- **Free-energy change** per day:
  ```math
  \Delta \mathcal{F}_t
    = \mathcal{F}(\hat p^{\text{det}}_{t+1}, t+1)
      - \mathcal{F}(p_t, t).
  ```
- **Dirichlet energy** using the symmetric part $`S`$:
  ```math
  \mathcal{E}(p) = -p^\top S p \ge 0,
  \qquad
  \Delta \mathcal{E}_t = \mathcal{E}(\hat p^{\text{det}}_{t+1}) - \mathcal{E}(p_t).
  ```

Empirical summary:

- Median $`\Delta \mathcal{F} \approx 3.75`$  
  (fraction of negative changes: **0%**).
- Median $`\Delta \mathcal{E} \approx 0`$  
  (fraction of negative changes: **23.73%**).

Interpretation:

- In an ideal gradient-flow picture, we’d expect **non-increasing** free energy and Dirichlet energy.
- Here, our heuristic tilt does **not** produce a clean monotone descent; the model is more of a **phenomenological drift–diffusion** than an exact Wasserstein gradient flow.

### 4.5 Reversibility and spectral sanity

For the chosen data configuration:

- **Detailed-balance gap** on the subset:
  ```math
  \max_{i,j} |\pi_i Q_{\text{rev},ij} - \pi_j Q_{\text{rev},ji}|
    \approx 2.89\times 10^{-19},
  ```
  i.e. **numerically reversible**.
- The largest real parts of eigenvalues of $`Q_{\text{rev}}`$ (subset) satisfy
  $`\mathrm{Re}(\lambda_k) \le 0`$ up to numerical noise.

This supports the view of $`Q_{\text{rev}}`$ as a stable, reversible diffusion operator suitable for the Onsager/Wasserstein connection.

### 4.6 Practical feature experiment: potential blending (price slot)

Motivation:

- In real e-commerce, transitions depend not only on history but also on **item attributes** (price, category, brand, etc.).

Idea:

- Construct an additional potential $`V^{\text{price}}(i)`$ and blend it with popularity:
  ```math
  V^{(w)}_t(i)
  = (1-w)\,V^{\text{pop}}_t(i)
    + w\,V^{\text{price}}(i),
  \qquad w\in[0,1].
  ```

What actually happened:

- In the Retailrocket properties table, we did **not** find a reliable price column, so we used a **neutral** feature:
  ```math
  V^{\text{price}}(i) \equiv 0,
  ```
  and treated $`w`$ as a **sensitivity parameter** for rescaling the potential.

Results (using the same stochastic pipeline, varying only $`V^{(w)}`$):

| mix weight $`w`$ | KL\_all | TV\_all | KL\_ev | TV\_ev |
|----------------|--------:|--------:|-------:|-------:|
| 0.0 (baseline, centered) | **0.0804** | **0.4886** | 0.1519 | 0.4890 |
| 0.2 | 0.0855 | 0.4889 | 0.1516 | 0.4890 |
| 0.4 | 0.0847 | 0.4889 | **0.1486** | **0.4889** |
| 0.6 | 0.0897 | 0.4891 | 0.1832 | 0.4904 |

Takeaways:

- Mild mixing ($`w \approx 0.4`$) **slightly improves event-window KL**, but large $`w`$ hurts.
- Practical message: **potential shaping can help**, but over-weighting side information (especially if noisy or neutral) can **degrade spike prediction**.
- A real attribute signal (true price / discount / category) would be needed for a definitive test.

### 4.7 Robustness checks: generator choice & time resolution

We ran a few lightweight robustness experiments:

- **R1: Baseline $`Q_{\text{rev}}`$ vs “lazy” $`Q_{\text{rev}}`$**

  - Lazy variant adds self-loops (slows dynamics).
  - Both reduce KL relative to a deterministic baseline; lazy $`Q_{\text{rev}}`$ slightly shifts the trade-off:
    - marginally weaker overall KL improvement,
    - sometimes better behavior in event windows (more stable under spikes).

- **R2: 12-hour buckets**

  - Using 12H time buckets increases sparsity and noise.
  - We observed smaller KL gains overall and **worse event-window behavior** compared to daily aggregation.
  - Trade-off: higher time resolution vs robustness.

- **R3: Raw $`Q`$ vs reversible $`Q_{\text{rev}}`$**

  - Reversibilization improves stability and event behavior:
    - $`Q_{\text{rev}}`$ gives slightly better (more negative) KL differences on average and on event windows.
  - This supports the modeling choice of working with a **reversible generator** when connecting to Wasserstein/Onsager theory.

---

## 5) Visualization & EDA

The notebooks generate presentation-ready plots, including:

- top-$`K`$ **stationary mass** (long-run hubs),
- in-/out-degree **histograms** (sparsity + hub structure),
- **exposure time** distributions $`T_i`$,
- top-10 items’ $`p_t(i)`$ trajectories over time,
- a sampled **item→item subgraph** (NetworkX + pyvis),
- **heatmap of $`P`$** restricted to top-$`k`$ items (block structure + hub stripes).

Outputs are stored under `figures/`.

---

## 6) Practical interpretation

What the results say in real-world terms:

- **Catalog-level attention is a flow**: sessions induce a directed item graph; diffusion-style models can forecast where **aggregate attention** will move tomorrow.
- **Adding trends and noise helps**:
  - the popularity-based potential captures **slow trends**;
  - low-rank stochasticity captures **bursty shocks** (promotions, external news, sudden popularity).
- For operations:
  - better forecasting of **which items gain attention tomorrow**,
  - improved handling of **spiky days**,
  - an interpretable graph view of how attention moves across the catalog.

---

## 7) Quickstart / reproducibility

### Data & environment

- Retailrocket CSVs under `data/retailrocket/` (or your Google Drive path).
- Python ≥ 3.10; install dependencies:

  ```bash
  pip install numpy pandas scipy networkx pyvis matplotlib tqdm pyarrow










### Colab workflow

1. Mount Drive and set paths:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   DATA_DIR = "/content/drive/MyDrive/retailrocket"           # raw CSVs
   ART_DIR  = "/content/drive/MyDrive/retailrocket_artifacts" # saved artifacts
   ```

2. Run notebooks in order:

   * `01_data_wrangling.ipynb`
     → events filtering, sessionization, top-$K$.
   * `02_graph_and_matrices.ipynb`
     → builds `C, P, T, pi, Q, Q_rev` and saves to `ART_DIR`.

     **Gotchas handled:**

     * timestamp units (ms vs s) for dwell time,
     * row-stochastic $P$ via **lazy self-loops** for zero-out rows.
   * `03_visualizations.ipynb`
     → EDA and graph plots.
   * `04_models_and_metrics.ipynb`
     → CTMC/deterministic/stochastic forecasts, KL/TV, event windows, Wilcoxon tests, energy diagnostics.
   * `05_feature_and_robustness.ipynb` (optional)
     → potential blending (price slot) and robustness checks (lazy $Q$, 12H buckets, raw $Q$ vs $Q_{\text{rev}}$).

Artifacts:

* stored in `retailrocket_artifacts/`:

  * `P.npz`, `C.npz`, `T.npy`, `pi.npy`,
  * `Q.npz`, `Q_rev.npz`,
  * `p_time.parquet`, `V_time.parquet`,
  * `item_index.csv`.

You can re-run experiments without rebuilding everything from scratch.

---

## 8) Repository structure (suggested)

```text
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
│  └─ 05_feature_and_robustness.ipynb
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

## 9) Limitations & future work

Current limitations:

* **Feature limitation**: the public Retailrocket properties lacked a reliable price field, so our “price potential” was effectively neutral.
* **Data/graph sparsity**: top-$K$ truncation plus filtering to `{view, addtocart}` yields a very sparse transition graph; long-tail items and rare transitions are underrepresented.
* **Temporal resolution trade-off**:

  * 12H buckets increase sparsity and noise and can hurt event-window performance.
  * Daily aggregation was more stable in our tests.

Future directions:

* **Stage-aware graph**
  Nodes as $(\text{item}, \text{stage})$ with stage $\in{\text{view}, \text{addtocart}, \text{purchase}}$, capturing funnel dynamics.
* **Richer potentials**
  Incorporate item metadata (category, brand), price/discounts, and promotion signals into $V_t$.
* **Adaptive noise**
  Let $\sigma$ depend on recent catalog volatility (e.g., increase when $\lVert p_t - p_{t-1} \rVert_1$ is large).
* **Systematic robustness**
  Explore different $K$, session gaps, dwell caps, and multiple random seeds.

---

## 10) License / citation

If you use this code or ideas, please cite this repository and the Retailrocket dataset.

---
