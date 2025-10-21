# Retailrocket-wasserstein-flows
# Noisy Preference Flows
_Modeling clickstream dynamics with Wasserstein drift–diffusion on graphs_

**tl;dr** We build a continuous-time, geometry-aware model of how *user attention mass* moves between items in real e-commerce sessions. Using the Retailrocket dataset, we estimate a reversible CTMC over items, define a free energy with a popularity-based potential, and compare **deterministic gradient flow** vs a **stochastic (fluctuation) model** for short-horizon distribution prediction. We also provide clear graph + matrix visualizations to make the dynamics tangible.

---

##  Highlights
- **Real data:** Retailrocket RecSys dataset (click/cart/purchase events).
- **Graph view:** items = nodes; within-session item→item transitions = directed edges.
- **Matrices:** counts `C`, row-stochastic `P`, exposure times `T`, CTMC generator `Q` (and reversible `Q_rev`), time-bucketed distributions `p_time`, rolling potential `V_time`.
- **Models:** 
  - Deterministic: \(\dot p = K\,\nabla \mathcal F(p)\), \(\mathcal F(p) = \sum p_i \log p_i + \sum V_i(t)\,p_i\)
  - Stochastic: \(dp = K\,\nabla \mathcal F(p)\,dt + \sigma\,B\,dW_t\) (low-rank noise aligned with the Onsager structure)
- **Metrics:** KL/TV distance and Top-k mass error for short-horizon prediction; event-window analysis for volatility.

---

##  Project structure
noisy-preference-flows/
├─ notebooks/
│ ├─ 01_data_wrangling.ipynb
│ ├─ 02_graph_and_matrices.ipynb
│ ├─ 03_visualizations.ipynb
│ ├─ 04_models_and_metrics.ipynb
│ └─ 05_stage_aware_extension.ipynb # optional (item,stage) nodes
├─ src/
│ ├─ io_utils.py
│ ├─ sessionize.py
│ ├─ build_graph.py
│ ├─ ctmc.py
│ ├─ potentials.py
│ ├─ simulate.py # ODE/SDE predictors
│ └─ eval.py
├─ figures/ # exported plots for the README / report
├─ requirements.txt
├─ README.md
└─ LICENSE
