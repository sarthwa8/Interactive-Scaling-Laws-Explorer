import numpy as np

# --- Constants from the paper's tables and figures ---
# From Table 5
ALPHA_N = 0.076 # [cite: 885]
NC = 8.8e13 # [cite: 885]

ALPHA_D = 0.095 # [cite: 885]
DC = 5.4e13 # [cite: 885]

# From Table 6 (for compute-efficient frontier)
# We use the empirically fitted exponents from Figure 14
P_N = 0.73 # [cite: 888]
N_E = 1.3e9 # [cite: 888]
P_S = 0.03 # [cite: 888]
S_E = 5.4e3 # [cite: 888]

# From Figure 13
ALPHA_C_MIN = 0.050 # [cite: 703, 885]
C_C_MIN = 2.3e8 # [cite: 703] (Note: Paper has 3.1e8 in table but 2.3e8 in figure, we use figure)


# --- Functions to calculate theoretical values ---

def calculate_loss_vs_n(n_params, alpha_n_val=ALPHA_N):
    """Calculates theoretical loss given N parameters."""
    return (NC / n_params)**alpha_n_val

def calculate_loss_vs_d(d_tokens, alpha_d_val=ALPHA_D):
    """Calculates theoretical loss given D tokens."""
    return (DC / d_tokens)**alpha_d_val

def calculate_overfitting_loss(n, d):
    """Calculates loss based on both N and D using Equation 1.5."""
    # [cite_start]Equation 1.5: L(N,D) = [(Nc/N)^(alpha_N/alpha_D) + Dc/D]^alpha_D [cite: 204]
    term_n = (NC / n)**(ALPHA_N / ALPHA_D) # [cite: 204]
    term_d = DC / d # [cite: 204]
    
    status = "Model capacity is the bottleneck."
    if term_d > term_n:
        status = "Dataset size is the bottleneck (overfitting)."
        
    loss = (term_n + term_d)**ALPHA_D # [cite: 204]
    return loss, status

def calculate_optimal_model_size(c_min):
    """Calculates optimal model size N for a given compute budget C_min."""
    # [cite_start]From Figure 14: N_opt = (1.3 * 10^9) * C_min^0.73 [cite: 728]
    return N_E * (c_min**P_N)

def calculate_optimal_steps(c_min):
    """Calculates optimal training steps S_min for a given compute budget C_min."""
    # [cite_start]From Figure 14: S_min = (5.4 * 10^3) * C_min^0.03 [cite: 733]
    return S_E * (c_min**P_S)

def calculate_best_loss_from_compute(c_min):
    """Calculates the best possible loss L for a given compute budget C_min."""
    # [cite_start]From Figure 13: L(C_min) = (C_c_min / C_min)^alpha_C_min [cite: 703]
    return (C_C_MIN / c_min)**ALPHA_C_MIN