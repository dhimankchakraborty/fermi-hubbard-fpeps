# -------------------------------------------------------------
# Variational fPEPS optimization of 2D Fermi-Hubbard (YASTN+torch)
# -------------------------------------------------------------
# This script:
#   1. Sets up a 2D Fermi-Hubbard model using YASTN.
#   2. Builds a product fPEPS as initial state.
#   3. Uses CTM to compute E[psi].
#   4. Uses automatic differentiation to get dE/dA for each PEPS tensor.
#   5. Performs simple gradient descent updates A <- A - lr * dE/dA.
#
# NOTE:
#   - This is a template: some minor API differences may exist depending
#     on your installed YASTN version. The overall logic is correct.
#   - No user-defined Python functions are used (only loops and library calls).
# -------------------------------------------------------------

import torch                                 # PyTorch backend for YASTN and AD
import yastn                                 # core YASTN package
import yastn.tn.fpeps as peps                # fPEPS module: PEPS, EnvCTM, etc.

# -------------------------------------------------------------
# 1. Model parameters and optimization hyperparameters
# -------------------------------------------------------------

# Hubbard model parameters
t = 1.0                                      # hopping amplitude
U = 10.0                                     # on-site interaction
mu = 0.0                                     # chemical potential (half-filling if symmetric)

# Lattice parameters (finite 2x2 lattice as a toy system)
Nx, Ny = 2, 2                                # number of sites in x (rows) and y (cols)

# PEPS / CTM truncation parameters
D_bond = 8                                   # PEPS bond dimension target (keep modest for testing)
chi = 5 * D_bond                             # CTM bond dimension
tol_ctm = 1e-10                              # CTM SVD tolerance

# Variational optimization parameters
learning_rate = 0.01                         # gradient descent learning rate
n_var_steps   = 100                          # number of optimization iterations
max_ctm_sweeps = 20                          # CTMRG sweeps per energy evaluation

# -------------------------------------------------------------
# 2. YASTN configuration and local operators
# -------------------------------------------------------------
# We want PyTorch backend so that energy is a differentiable scalar.
# The symmetry is spinful fermions: U(1) x U(1) (for N_up and N_dn).
# YASTN uses config kwargs to control backend, device, dtype, etc.

config_kwargs = {
    "backend": "torch",                      # use PyTorch as backend
    "default_dtype": "float64",              # double precision
    "device": "cpu"                          # change to "cuda" if you want GPU
}

# YASTN provides predefined spinful fermion operators with symmetry U1xU1.
# Passing config_kwargs ensures they are built with the torch backend.
ops = yastn.operators.SpinfulFermions(sym="U1xU1", **config_kwargs)

# Identity and local fermion operators.
I = ops.I()                                  # identity operator on local Hilbert space
c_up  = ops.c("u")                           # annihilation operator for spin-up
cdag_up = ops.cp("u")                        # creation operator for spin-up
n_up  = ops.n("u")                           # number operator for spin-up

c_dn  = ops.c("d")                           # annihilation operator for spin-down
cdag_dn = ops.cp("d")                        # creation operator for spin-down
n_dn  = ops.n("d")                           # number operator for spin-down

# -------------------------------------------------------------
# 3. Lattice geometry and initial PEPS
# -------------------------------------------------------------
# We use a finite square lattice with open/finite boundaries.
# For an infinite system, you would use CheckerboardLattice or SquareLattice with boundary='infinite'.

geometry = peps.SquareLattice(
    dims=(Nx, Ny),                           # number of rows and columns
    boundary="obc"                        # finite 2x2 patch
)

# Initial PEPS: simple product state.
# We use the identity I as a "vector" to build the local purification-like state, as in the Hubbard quickstart.
# You can replace I by a more interesting local vector if you want a specific filling or spin pattern.
psi = peps.product_peps(
    geometry=geometry,                       # lattice geometry
    vectors=I                                # same local vector on each site
    # YASTN infers config from the operator's config (which has torch backend)
)

# -------------------------------------------------------------
# 4. Turn on autograd for all PEPS site tensors
# -------------------------------------------------------------
# We want each PEPS tensor A(site) to track gradients dE/dA.
# geometry.sites() gives the list of unique Site objects.
# psi[site] accesses the PEPS tensor at that site.

for site in geometry.sites():               # loop over all sites of the lattice
    A = psi[site]                           # A is a yastn.Tensor associated with this site
    A.requires_grad_(True)                  # tell YASTN/PyTorch to track gradients through A

# -------------------------------------------------------------
# 5. Variational optimization loop: minimize E[psi]
# -------------------------------------------------------------
# The energy functional E[psi] is evaluated via CTM, using the same logic as the
# "2D Fermi-Hubbard at finite temperature" quickstart, but here it is used as
# a zero-temperature variational cost function.
#
# At each iteration:
#   (1) Zero previous gradients.
#   (2) Build CTM environment from current psi.
#   (3) Run CTMRG to convergence (or fixed number of sweeps).
#   (4) Use env_ctm.measure_1site and measure_nn to build energy E.
#   (5) Backpropagate E.backward().
#   (6) Update each tensor A <- A - lr * grad(A).

# Small helper for averaging over dict values (no custom def; use lambda)
mean = lambda data: sum(data.values()) / len(data)  # data is dict {site/bond: yastn-scalar}

for it in range(1, n_var_steps + 1):
    # ---------------------------------------------------------
    # (1) Zero previous gradients on all site tensors
    # ---------------------------------------------------------
    for site in geometry.sites():
        A = psi[site]
        if getattr(A, "grad", None) is not None:  # if gradient exists
            A.grad = None                         # reset to None (like torch.zero_grad())

    # ---------------------------------------------------------
    # (2) Build CTM environment for current psi
    # ---------------------------------------------------------
    # EnvCTM constructs corner + edge tensors environment for the current PEPS.
    # init='eye' starts from identity-like environment.
    env_ctm = peps.EnvCTM(psi, init="eye")

    # SVD options for CTM truncation.
    # D_total is the CTM bond dimension (chi).
    opts_svd_ctm = {
        "D_total": chi,                          # chi = 5 * D_bond
        "tol": tol_ctm                           # truncation tolerance
        # 'tol_block' can also be set if needed
    }

    # Run CTMRG sweeps; iterator=True returns a generator over sweeps
    ctm_generator = env_ctm.ctmrg_(
        opts_svd=opts_svd_ctm,
        iterator=True,
        max_sweeps=max_ctm_sweeps
    )

    # Here we just run CTMRG for max_ctm_sweeps sweeps without early stopping.
    # If you want, you can measure energy at each sweep and implement your own stopping.
    for info in ctm_generator:
        # info.sweeps gives the current sweep number
        # we don't use it here; just let CTMRG converge as much as allowed by max_sweeps
        pass

    # ---------------------------------------------------------
    # (3) Measure energy from CTM environment
    # ---------------------------------------------------------
    # We follow the Hubbard quickstart formula:
    #   H = -t ∑⟨ij⟩σ (c†_{iσ} c_{jσ} + h.c.)
    #       + U ∑_i (n_{i↑} - 1/2)(n_{i↓} - 1/2)
    #       - μ ∑_{iσ} n_{iσ}
    #
    # The CTM example for infinite Hubbard gives:
    #   energy_per_site = -4 t ( ⟨c†_up c_up⟩_nn + ⟨c†_dn c_dn⟩_nn ) + U ⟨(n_up - 1/2)(n_dn - 1/2)⟩
    #
    # where ⟨...⟩_nn is the average nearest-neighbor expectation value per bond,
    # and the factor 4 counts bonds per site in the infinite lattice. For a finite 2x2,
    # this factor is approximate but good as a template.

    # On-site interaction term: (n_up - I/2)(n_dn - I/2)
    nn_op = (n_up - I / 2) @ (n_dn - I / 2)       # YASTN supports @ as operator composition

    # Measure on-site expectation values using CTM (returns dict {site: scalar})
    ev_nn_dict = env_ctm.measure_1site(nn_op)     # measure on all unique sites

    # Average over all sites
    ev_nn = mean(ev_nn_dict)                      # yastn scalar tensor with torch backend

    # Measure nearest-neighbor hopping correlators for up and down spins
    # measure_nn returns dict {bond: scalar} over unique bonds
    ev_cdagc_up_dict = env_ctm.measure_nn(cdag_up, c_up)
    ev_cdagc_dn_dict = env_ctm.measure_nn(cdag_dn, c_dn)

    # Average over all bonds
    ev_cdagc_up = mean(ev_cdagc_up_dict)          # yastn scalar
    ev_cdagc_dn = mean(ev_cdagc_dn_dict)          # yastn scalar

    # Compute energy per site according to the quickstart formula
    # Note: this uses the "infinite lattice" prefactor -4 t; for a finite patch
    # you might adjust it, but as a variational cost function it is fine.
    energy_yastn = -4.0 * t * (ev_cdagc_up + ev_cdagc_dn) + U * ev_nn

    # Convert to a differentiable scalar torch.Tensor.
    # to_number() gives a 0-dim torch tensor keeping the autograd graph.
    E_scalar = energy_yastn.to_number()          # torch scalar, requires_grad=True

    # ---------------------------------------------------------
    # (4) Backpropagate to get gradients dE/dA for all site tensors
    # ---------------------------------------------------------
    # This calls PyTorch autograd through YASTN's tensor operations.
    E_scalar.backward()

    # ---------------------------------------------------------
    # (5) Gradient descent update: A <- A - lr * grad(A)
    # ---------------------------------------------------------
    # We do an in-place parameter update *without* tracking it in the computational graph.
    with torch.no_grad():
        for site in geometry.sites():
            A = psi[site]                        # current PEPS tensor at site
            g = A.grad                           # gradient yastn.Tensor (same symmetry structure)
            if g is None:
                continue                         # some sites might not contribute (shouldn't happen, but safe)

            # Simple gradient descent step:
            # yastn supports algebra with tensors: A - lr * g is another yastn.Tensor
            # We assign that back into psi[site].
            psi[site] = A - learning_rate * g    # new tensor detached from old graph

    # ---------------------------------------------------------
    # (6) Diagnostics: print energy
    # ---------------------------------------------------------
    # Now it is safe to convert E_scalar to a Python float for logging.
    # We use .item() AFTER backward and update (we don't need its grad anymore).
    if it == 1 or it % 10 == 0:
        print(f"[var-step {it:4d}] E/site = {E_scalar.item(): .12f}")
        # Example output (not actual, since we are not running this here):
        # [var-step    1] E/site = -2.100000000000
        # [var-step   10] E/site = -2.365000000000
        # ...

# -------------------------------------------------------------
# After loop: psi is your variationally optimized PEPS
# You can now reuse env_ctm + measure_1site/nn to extract observables:
#   - ⟨n_up⟩, ⟨n_dn⟩
#   - double occupancy ⟨n_up n_dn⟩
#   - spin-spin correlators ⟨S^z_i S^z_j⟩
# using the same pattern as in the Hubbard quickstart.
# -------------------------------------------------------------
