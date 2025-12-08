# === Cell W1: Dirichlet energy and Free energy diagnostics ===
import numpy as np
from scipy.sparse import csc_matrix

def dirichlet_energy(Q_rev, p):
    """E(p) = - p^T S p with S = 0.5 (Q_rev + Q_rev^T)"""
    S = 0.5 * (Q_rev + Q_rev.T)
    Sp = S @ p
    return float(- p @ Sp)

def free_energy(p, V, eps=1e-12):
    """F(p) = sum_i p_i log p_i + sum_i V_i p_i"""
    p = np.maximum(p, eps)
    return float(np.sum(p * np.log(p)) + np.sum(V * p))

# === W2 (revised): free-energy & Dirichlet-energy traces on the chosen subset ===
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply

# --- Assumes H5 has defined: Q_M, Q_M_csc, p_sub, V_sub, alpha_b, DELTA_SEC.
# If Q_M_csc is missing for any reason, rebuild it:
Q_M_csc = Q_M_csc if 'Q_M_csc' in globals() else csc_matrix(Q_M)

def free_energy(p, V, eps=1e-12):
    """F(p;V) = Σ p_i log p_i + Σ V_i p_i  (no α inside; α only affects the tilt step)."""
    p = np.maximum(np.asarray(p, float), eps)
    V = np.asarray(V, float)
    return float((p * np.log(p)).sum() + (p * V).sum())

def dirichlet_energy(Q, p):
    """E(p) = - pᵀ S p with S = ½(Q+Qᵀ) (≥0)."""
    S = 0.5 * (Q + Q.T)   # sparse
    p = np.asarray(p, float)
    return float(- p @ (S @ p))

dF, dE = [], []
Tsteps = len(p_sub.index) - 1

for t in range(Tsteps):
    pt   = p_sub.iloc[t].to_numpy(float)
    Vt   = V_sub.iloc[t].to_numpy(float)
    Vnxt = V_sub.iloc[t+1].to_numpy(float)

    # 1) evolve by reversible CTMC for one day
    y_ctmc = expm_multiply(DELTA_SEC * Q_M_csc, pt)

    # 2) deterministic “free-energy” tilt by next-day potential, then renormalize
    y_det  = renorm_simplex(y_ctmc * np.exp(-alpha_b * Vnxt))

    # Record changes in free energy and Dirichlet energy
    dF.append(free_energy(y_det, Vnxt) - free_energy(pt, Vt))
    dE.append(dirichlet_energy(Q_M, y_det) - dirichlet_energy(Q_M, pt))

plt.figure(figsize=(10,3.5))
plt.plot(dF, label="ΔF (deterministic step)")
plt.axhline(0, color='k', lw=0.8)
plt.title("Free-energy change per day (more negative ⇒ better alignment with V)")
plt.xlabel("t → t+1"); plt.ylabel("ΔF")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,3.5))
plt.plot(dE, color='tab:orange', label="ΔE (Dirichlet)")
plt.axhline(0, color='k', lw=0.8)
plt.title("Dirichlet-energy change per day")
plt.xlabel("t → t+1"); plt.ylabel("ΔE")
plt.legend(); plt.tight_layout(); plt.show()

print(f"Median ΔF: {np.median(dF):.4f} | fraction negative: {(np.array(dF)<0).mean():.2%}")
print(f"Median ΔE: {np.median(dE):.4f} | fraction negative: {(np.array(dE)<0).mean():.2%}")

# === W3 (revised): course-ties & spectral sanity on the chosen subset ===
import numpy as np
import scipy.sparse.linalg as sla

# π restricted to the chosen subgraph (renormalized)
pi_sub = pi[sub]
pi_sub = pi_sub / pi_sub.sum()

gap = detailed_balance_gap(Q_M, pi_sub)  # uses your earlier helper
print(f"Detailed-balance gap on subset: {gap:.3e}  (≈0 ⇒ numerically reversible).")

# Spectral sanity: generator eigenvalues should have Re(λ) ≤ 0
k_eval = min(4, max(Q_M.shape[0]-2, 1))
lam = sla.eigs(Q_M, k=k_eval, which='LR', return_eigenvectors=False)
print("Largest real parts of eig(Q_M):", np.sort(np.real(lam))[::-1])

print("Onsager view: S = ½(Q+Qᵀ) is negative semidefinite; Dirichlet E(p) = -pᵀ S p ≥ 0.")
