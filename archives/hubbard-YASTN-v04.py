#!/usr/bin/env python3
"""
Finite 2D Fermi-Hubbard model with fPEPS (YASTN)
-------------------------------------------------
This script sets up:
  - Imports
  - Model parameters (t, U, mu)
  - Spinful fermion operators (U(1) x U(1))
  - Finite 2D square lattice geometry
  - Skeleton for a Néel-like initial product PEPS: ↑, ↓, ↑, ↓, ...

NO time-evolution or Hamiltonian gates yet.
"""

# =========================
# 1. Imports and config
# =========================

# Core YASTN package: symmetric tensors + TN algorithms
import yastn

# fPEPS module: lattice geometries, PEPS container, product_peps, evolution, environments, ...
import yastn.tn.fpeps as fpeps

# Spinful fermion operator factory:
# provides creation/annihilation/number operators etc. for the Hubbard model
import yastn.operators as op_mod


# =========================
# 2. Model parameters
# =========================

# H = -t Σ_{<ij>,σ} (c†_{iσ} c_{jσ} + h.c.)
#     + U Σ_i (n_{i↑} - 1/2)(n_{i↓} - 1/2)
#     - μ Σ_{i,σ} n_{iσ}
#
# We use the particle-hole symmetric form, so at half-filling μ = 0

t = 1.0      # hopping amplitude (sets energy scale)
U = 10.0     # on-site interaction; choose something "strong" like 4–8
mu = 0.0     # half filling in the particle-hole symmetric form

# Finite lattice size: Lx x Ly sites
Lx = 2
Ly = 2


# =========================
# 3. Spinful fermion operators
# =========================

# We set up spinful fermions with U(1) x U(1) symmetry:
#   - one U(1) for N_up
#   - one U(1) for N_dn
#
# This gives the local Hilbert space |0>, |↑>, |↓>, |↑↓>
# with appropriate charges stored inside the tensor structure.

ops = op_mod.SpinfulFermions(sym="U1xU1")
# 'ops' now knows:
#   - identity I
#   - creation/annihilation: c('u'), cp('u'), c('d'), cp('d')
#   - number operators: n('u'), n('d')
#   - and internal info about local legs and charges

# Convenience handles for local operators (we'll need them later for the Hamiltonian)
I = ops.I()                            # local identity (possibly on physical⊗ancilla in purification contexts)
c_up, cdag_up, n_up = ops.c('u'), ops.cp('u'), ops.n('u')
c_dn, cdag_dn, n_dn = ops.c('d'), ops.cp('d'), ops.n('d')

print("Local operator setup completed.")
print("Type of I:", type(I))          # e.g. <class 'yastn.tensor.Tensor'>
print("Shape of I:", I.get_shape())   # example: something like (d_phys, d_phys) depending on context


# =========================
# 4. Lattice geometry: finite 2D square lattice
# =========================

# We build a finite Lx x Ly square lattice with open boundary conditions ("finite").
# This geometry object knows:
#   - which sites exist
#   - how they’re indexed (x, y)
#   - which nearest-neighbour bonds exist

geometry = fpeps.SquareLattice(dims=(Lx, Ly), boundary="obc")

print("\nLattice geometry created.")
print("Number of sites:", len(list(geometry.sites())))  # e.g. 16 for 4x4
print("Sites (x, y):", list(geometry.sites()))          # e.g. [(0,0), (1,0), ..., (3,3)]
print("Number of unique bonds:", len(list(geometry.bonds())))


# =========================
# 5. Initial Néel-like product state: ↑, ↓, ↑, ↓, ...
# =========================
#
# We want an initial state that looks like
#
#   ↑  ↓  ↑  ↓
#   ↓  ↑  ↓  ↑
#   ↑  ↓  ↑  ↓
#   ↓  ↑  ↓  ↑
#
# i.e. n_up + n_dn = 1 on every site (half-filling),
# with staggered spin pattern to bias towards AFM order.
#
# In PEPS language, this means:
#   - For each lattice site (x, y), we assign a one-site tensor |↑> or |↓>
#   - Then we call product_peps(geometry, vectors=...) which attaches trivial
#     bond-dimension-1 virtual legs and constructs the full PEPS.
#
# The only subtlety is: how to get the one-site tensors for |↑> and |↓> from YASTN?
#
# The SpinfulFermions operator factory "ops" encodes the local Hilbert space and charges.
# It typically provides helper methods to get local basis states, but these are not
# explicitly documented in the quickstart snippet. A robust way to proceed when you run
# this script is:
#
#   1. Inspect 'ops' in your Python REPL:
#         >>> dir(ops)
#      and look for something like:
#         vec_empty / vec_up / vec_dn / vec_doubly_occ
#      or similar constructors for basis kets.
#
#   2. Once you have the correct names, define:
#         state_up = ops.vec_up()
#         state_dn = ops.vec_dn()
#
#   3. Use those as vectors in product_peps.
#
# Since I cannot introspect your local installed version of YASTN directly from here,
# I'll write the code assuming you fill in 'state_up' and 'state_dn' with the correct
# basis constructors.
#
# *** IMPORTANT: You MUST adjust the two TODO lines below according to the actual
#                YASTN SpinfulFermions API on your machine. ***

# === TODO: define the one-site basis states |↑> and |↓> ===

# Example pattern of what you will likely do after you inspect dir(ops):
#   state_empty = ops.vec_empty()      # |0>
#   state_up    = ops.vec_up()         # |↑>
#   state_dn    = ops.vec_dn()         # |↓>
#   state_dbl   = ops.vec_doubly_occ() # |↑↓>
#
# For now, put placeholders so the script structure is clear;
# replace them with the real calls in your environment.

state_up = ops.vec_n((1, 0))   # TODO: replace with something like: ops.vec_up()
state_dn = ops.vec_n((0, 1))   # TODO: replace with something like: ops.vec_dn()

if state_up is None or state_dn is None:
    print("\n[WARNING] You still need to define 'state_up' and 'state_dn' from 'ops'.")
    print("          Run this script once, inspect 'dir(ops)', and plug in the correct")
    print("          constructor methods for |↑> and |↓> basis states.")
else:
    print("\nOne-site basis states defined:")
    print("  |↑> shape:", state_up.get_shape())
    print("  |↓> shape:", state_dn.get_shape())

# === Build the dictionary of local vectors for the Néel pattern ===

vectors = {}  # maps (x, y) -> one-site vector tensor

for (x, y) in geometry.sites():
    # Simple checkerboard rule: (x + y) even -> up, odd -> down
    if (x + y) % 2 == 0:
        vectors[(x, y)] = state_up
    else:
        vectors[(x, y)] = state_dn

# At this point, 'vectors' is a complete map from sites to local states
# (provided you filled state_up/state_dn properly).


# === Create the product PEPS ===

# product_peps(geometry, vectors) -> Peps
#   - If 'vectors' is a single Tensor, it's copied to each site.
#   - If 'vectors' is a dict[(x, y) -> Tensor], each site gets its own entry.
#   - Virtual legs of dimension 1 with zero charge are automatically added.
#
# This PEPS has bond dimension D = 1 and is an unentangled Néel product state.

psi = None
if state_up is not None and state_dn is not None:
    psi = fpeps.product_peps(geometry=geometry, vectors=vectors)
    print("\nProduct PEPS created.")
    print("PEPS type:", type(psi))
    print("Unique sites in PEPS:", list(psi.sites()))
    # Example: you can inspect one tensor:
    site0 = next(iter(psi.sites()))
    print(f"Example tensor at site {site0}: ndim =", psi[site0].ndim)  # should be 5 for PEPS with physical leg
else:
    print("\n[INFO] Skipping product_peps construction until state_up/state_dn are defined.")
    psi = None


# =========================
# 6. Hubbard Hamiltonian gates (hopping + interaction)
# =========================
#
# Now we build the *imaginary-time* evolution gates for the Hubbard Hamiltonian,
# using the fPEPS gate helpers provided by YASTN.
#
# The quickstart for the infinite Hubbard model does exactly this: :contentReference[oaicite:1]{index=1}
#
#   g_hop_u = peps.gates.gate_nn_hopping(t, db/2, I, c_up, cdag_up)
#   g_hop_d = peps.gates.gate_nn_hopping(t, db/2, I, c_dn, cdag_dn)
#   g_loc   = peps.gates.gate_local_Coulomb(mu, mu, U, db/2, I, n_up, n_dn)
#   gates   = peps.gates.distribute(geometry, gates_nn=[g_hop_u, g_hop_d],
#                                   gates_local=g_loc)
#
# Here:
#   - gate_nn_hopping  builds nearest-neighbour two-site gates exp(-Δτ H_kin),
#   - gate_local_Coulomb builds one-site gates exp(-Δτ H_onsite),
#   - distribute() places those gates on all bonds/sites of *your* geometry,
#     and (by default) symmetrizes them for a 2nd-order Trotter scheme.
#
# We'll copy this structure but interpret Δτ as imaginary-time step for
# ground-state projection instead of β/2 for finite temperature.

# --- Imaginary-time Trotter step setup ---

dtau = 0.05  # imaginary-time step (you will tune this later)
n_steps = 40  # number of Trotter steps for projection (also tunable)

# In a standard 2nd-order Trotter:
#   e^{-dtau H} ≈ e^{-dtau/2 H_loc} e^{-dtau H_kin} e^{-dtau/2 H_loc}
# and if we symmetrize over bond orderings as well, the library chooses
# a pattern where each gate takes effectively half-step 'dtau / 2'.
#
# So we follow the docs and pass dtau / 2 into gate constructors,
# then let fpeps.gates.distribute() handle the symmetric ordering.

# --- Build nearest-neighbour hopping gates for ↑ and ↓ spins ---

# gate_nn_hopping(
#     t,          # hopping amplitude appearing in H
#     dt_eff,     # effective time-step (here dtau / 2)
#     I,          # local identity
#     c, cdag     # annihilation and creation operators
# )
#
# It internally builds exp(-dt_eff * H_kin,σ) on a bond.
# Note: sign conventions are handled *inside* this helper; follow the example.

g_hop_up = fpeps.gates.gate_nn_hopping(t, dtau / 2, I, c_up, cdag_up)
g_hop_dn = fpeps.gates.gate_nn_hopping(t, dtau / 2, I, c_dn, cdag_dn)

# --- Build on-site Coulomb + chemical potential gate ---

# gate_local_Coulomb(
#     mu_up, mu_dn,  # spin-dependent chemical potentials
#     U,             # Hubbard interaction
#     dt_eff,        # effective time-step (here dtau / 2)
#     I,             # identity
#     n_up, n_dn     # number operators
# )
#
# It represents exp(-dt_eff * H_loc(i)), with:
#   H_loc(i) = U (n_up - 1/2)(n_dn - 1/2) - mu_up n_up - mu_dn n_dn
#
# For half filling with particle-hole symmetric form, mu_up = mu_dn = 0.

g_loc = fpeps.gates.gate_local_Coulomb(mu, mu, U, dtau / 2, I, n_up, n_dn)

print("\nHubbard gates built:")
print("  g_hop_up type:", type(g_hop_up))
print("  g_hop_dn type:", type(g_hop_dn))
print("  g_loc    type:", type(g_loc))

# --- Distribute gates over the finite lattice geometry ---

# distribute(
#     geometry,
#     gates_nn=[...],   # list of NN gate templates (applied to each bond)
#     gates_local=...   # local gate template (applied to each site)
# )
#
# For a given geometry (finite or infinite), this returns a list of gate
# objects each equipped with:
#   - which sites/bond they act on,
#   - the actual tensor for the gate,
#   - internal ordering for 2nd-order Trotterization.

gates = fpeps.gates.distribute(
    geometry,
    gates_nn=[g_hop_up, g_hop_dn],   # hopping for both spins
    gates_local=g_loc                # on-site interaction + μ term
)

print(f"\nTotal number of gates in one Trotter layer: {len(gates)}")

# At this point:
#   - 'psi' is an initial Néel product fPEPS (once state_up/state_dn are filled in),
#   - 'gates' is the full set of imaginary-time evolution gates corresponding to
#     your 2D finite Hubbard model at half filling.
#
# The next step (later) will be:
#   - choose an environment (e.g. env = fpeps.EnvNTU(psi, which='NN')),
#   - run a loop calling fpeps.evolution_step_(env, gates, opts_svd=...),
#   - and then use CTM to measure energy and correlators.


# =========================
# 7. Imaginary-time evolution with NTU
# =========================
#
# Goal:
#   - Use Neighborhood Tensor Update (EnvNTU) as environment.
#   - Repeatedly apply imaginary-time gates `gates` to project onto
#     the ground state of the 2D Hubbard model.
#   - Control bond dimension with opts_svd_ntu.
#
# Prerequisites (defined above this block):
#   psi    : initial PEPS (Néel product state)
#   gates  : list of Gate objects from fpeps.gates.distribute(...)
#   dtau   : imaginary-time step
#   n_steps: number of Trotter steps

# Safety check: we only evolve if psi exists (i.e. state_up/state_dn were defined).
if psi is None:
    print("\n[ERROR] psi is None. Define 'state_up' and 'state_dn' correctly, "
          "build the product_peps, and then run the evolution.")
else:
    # -------------------------
    # 7.1 NTU environment init
    # -------------------------
    #
    # EnvNTU(psi, which='NN') builds a local environment around each bond.
    #   which='NN'    -> smallest cluster (fastest, least accurate)
    #   which='NN+'   -> larger cluster
    #   which='NN++'  -> even larger, more accurate, more expensive
    #
    # For a first run, 'NN' or 'NN+' is a sensible choice.
    
    env_ntu = fpeps.EnvNTU(psi, which="NN")  # choose "NN+" later if you want
    
    # -------------------------
    # 7.2 SVD truncation options
    # -------------------------
    #
    # evolution_step_ needs opts_svd that control the *target bond dimension*.
    # These options are passed to yastn.linalg.svd_with_truncation().
    #
    # Typical keys:
    #   "D_total" : maximum total bond dimension after truncation
    #   "tol"     : tolerance for discarded weight (optional)
    #   "cutoff"  : smallest singular value to keep (optional)
    #
    # For a tiny system like 4x4, you can start with D_total = 4 or 6
    # and later experiment.

    D_target = 16  # <- choose your PEPS bond dimension here

    opts_svd_ntu = {
        "D_total": D_target,   # target bond dimension for all virtual bonds
        # optional:
        # "tol": 1e-8,         # discard if weight is below this (if you want)
        # "cutoff": 0.0,       # minimal singular value to keep
    }

    print(f"\n[NTU] Initializing imaginary-time evolution with D_total = {D_target}")
    print(f"[NTU] Using cluster type: 'NN'")
    print(f"[NTU] Time step dtau = {dtau}, number of steps = {n_steps}")

    # -------------------------
    # 7.3 Imaginary-time loop
    # -------------------------
    #
    # evolution_step_(env, gates, opts_svd=opts_svd_ntu) does:
    #   - applies the list of gates in a 2nd-order Trotter pattern
    #   - after each non-local (two-site) gate, truncates back to D_total
    #   - uses env_ntu.bond_metric(...) to define the metric for truncation
    #
    # It returns a list of "Evolution_out" NamedTuples (one per gate application),
    # containing diagnostics like truncation error, metric quality, etc.
    #
    # We store all of these in 'infoss' to later estimate accumulated truncation error.

    infoss = []  # list to store evolution diagnostics at each step

    for step in range(n_steps):
        # Perform a single imaginary-time step
        infos = fpeps.evolution_step_(env_ntu, gates, opts_svd=opts_svd_ntu)
        infoss.append(infos)

        # Optional: quick monitoring of truncation error on the fly
        # We can take the mean truncation error over all gates in this step:
        mean_err = sum(info.truncation_error for info in infos) / len(infos)
        print(f"[NTU] Step {step+1}/{n_steps}  "
              f"mean truncation error ≈ {mean_err:.3e}")

    # -------------------------
    # 7.4 Accumulated truncation error
    # -------------------------
    #
    # The docs provide helper fpeps.accumulated_truncation_error(infoss)
    # to accumulate truncation errors over all steps as a rough estimate
    # of the total evolution error. :contentReference[oaicite:1]{index=1}

    Delta = fpeps.accumulated_truncation_error(infoss, statistics="mean")
    print(f"\n[NTU] Estimated accumulated truncation error Δ ≈ {Delta:.3e}")

    # -------------------------
    # 7.5 Getting the evolved state
    # -------------------------
    #
    # The PEPS is stored inside env_ntu and is updated in place.
    # Usually, env_ntu.psi refers to the same object as the original `psi`,
    # but to be explicit we can rebind:

    psi = env_ntu.psi

    print("\n[NTU] Imaginary-time evolution finished.")
    print("      'psi' now holds the evolved fPEPS (approximate ground state).")


# =========================
# 8. CTM environment and observables
# =========================
#
# At this point:
#   - psi is the *evolved* PEPS (approximate ground state),
#   - geometry is the finite SquareLattice,
#   - c_up, cdag_up, c_dn, cdag_dn, n_up, n_dn, I, t, U are defined.
#
# Now we:
#   - Build a CTM environment for psi.
#   - Run CTMRG until convergence.
#   - Measure energy per site and local observables.

if psi is None:
    print("\n[ERROR] psi is None; CTM environment cannot be built.")
else:
    # -------------------------
    # 8.1 Initialize CTM
    # -------------------------
    #
    # EnvCTM(psi, init='eye') builds a corner-transfer-matrix environment
    # around the *double layer* of psi.
    #
    # 'init=\"eye\"' starts from bond-dimension-1 "identity" corners and edges.

    env_ctm = fpeps.EnvCTM(psi, init="eye")

    # CTMRG has its own bond dimension χ (environment bond dimension).
    # Often one chooses χ ~ 3–5 times the PEPS bond dimension D.
    chi = 5 * D_target

    opts_svd_ctm = {
        "D_total": chi,    # environment bond dimension χ
        "tol": 1e-10,      # strict truncation tolerance
    }

    # Helper function for averaging over dict values (sites or bonds)
    def mean_dict_values(d):
        vals = list(d.values())
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\n[CTM] Initializing CTMRG with χ = {chi}")
    print("[CTM] Running CTMRG sweeps until energy converges...")

    # -------------------------
    # 8.2 CTMRG loop with energy-based convergence
    # -------------------------
    #
    # ctmrg_(iterator=True) returns a generator; after each sweep we can:
    #   - compute energy per site
    #   - decide whether to stop based on its change.

    ctm_iter = env_ctm.ctmrg_(
        opts_svd=opts_svd_ctm,
        iterator=True,     # yield after each sweep
        max_sweeps=50,     # hard cap on sweeps
        # you *can* also pass corner_tol=... but here we do energy-based stopping
    )

    energy_old = 0.0
    tol_energy = 1e-7

    # We’ll also compute some geometry data for kinetic-energy normalization.
    sites_list = list(geometry.sites())
    bonds_list = list(geometry.bonds())
    num_sites = len(sites_list)
    num_bonds = len(bonds_list)

    # For a general finite graph:
    #   N_bonds = N_sites * z_avg / 2  -> z_avg = 2 * N_bonds / N_sites
    # This z_eff accounts for boundary sites having fewer neighbours than 4.
    z_eff = 2.0 * num_bonds / num_sites

    print(f"[CTM] Number of sites: {num_sites}, number of bonds: {num_bonds}")
    print(f"[CTM] Effective coordination z_eff = {z_eff:.3f}")

    for info in ctm_iter:
        # 1-site local interaction contribution:
        #
        # (n_up - I/2)(n_dn - I/2)
        on_site_op = (n_up - I / 2) @ (n_dn - I / 2)
        ev_loc = env_ctm.measure_1site(on_site_op)  # {site: value}
        ev_loc_mean = mean_dict_values(ev_loc)      # average over all sites

        # NN kinetic contribution: <c†_iσ c_jσ> over bonds, for each spin.
        ev_cdagc_up = env_ctm.measure_nn(cdag_up, c_up)  # {bond: value}
        ev_cdagc_dn = env_ctm.measure_nn(cdag_dn, c_dn)  # {bond: value}
        ev_cdagc_up_mean = mean_dict_values(ev_cdagc_up)
        ev_cdagc_dn_mean = mean_dict_values(ev_cdagc_dn)

        # Kinetic energy per site:
        #   H_kin = -t Σ_<ij>,σ (c†_iσ c_jσ + h.c.)
        # If the state is real and homogeneous on average, we can take:
        #   E_kin/site ≈ - z_eff * t * (ev_cdagc_up_mean + ev_cdagc_dn_mean)
        # where z_eff = 2 * N_bonds / N_sites (effective coordination).
        E_kin_site = - z_eff * t * (ev_cdagc_up_mean + ev_cdagc_dn_mean)

        # On-site interaction per site:
        #   H_loc = U (n_up - 1/2)(n_dn - 1/2)
        # measure_1site already gave us the *site average* of that operator.
        E_loc_site = U * ev_loc_mean

        # Total energy per site:
        energy = E_kin_site + E_loc_site

        print(f"[CTM] Sweep {info.sweeps:2d} | "
              f"E/site = {energy: .10f} | "
              f"E_kin/site = {E_kin_site: .10f} | "
              f"E_loc/site = {E_loc_site: .10f}")

        # Energy-based convergence check:
        if abs(energy - energy_old) < tol_energy:
            print(f"[CTM] Converged: ΔE < {tol_energy}")
            break

        energy_old = energy

    print("\n[CTM] Final estimated ground-state energy per site:", energy)


    # -------------------------
    # 8.3 Local densities and double occupancy
    # -------------------------
    #
    # Check half filling: <n_up> ≈ <n_dn> ≈ 0.5.

    ev_n_up = env_ctm.measure_1site(n_up)  # {site: value}
    ev_n_dn = env_ctm.measure_1site(n_dn)  # {site: value}
    n_up_mean = mean_dict_values(ev_n_up)
    n_dn_mean = mean_dict_values(ev_n_dn)

    print(f"[OBS] <n_up> ≈ {n_up_mean: .10f}")
    print(f"[OBS] <n_dn> ≈ {n_dn_mean: .10f}")
    print(f"[OBS] Total filling <n_up + n_dn> ≈ {n_up_mean + n_dn_mean: .10f}")

    # Double occupancy D = <n_up n_dn>
    double_occ = env_ctm.measure_1site(n_up @ n_dn)
    double_occ_mean = mean_dict_values(double_occ)
    print(f"[OBS] <n_up n_dn> (double occupancy) ≈ {double_occ_mean: .10f}")

    # -------------------------
    # 8.4 Spin-spin nearest-neighbor correlator
    # -------------------------
    #
    # Sz = 1/2 (n_up - n_dn); compute <Sz_i Sz_j> on NN bonds and average.

    Sz = 0.5 * (n_up - n_dn)
    ev_SzSz = env_ctm.measure_nn(Sz, Sz)   # {bond: value}
    SzSz_mean = mean_dict_values(ev_SzSz)
    print(f"[OBS] <Sz_i Sz_j>_NN ≈ {SzSz_mean: .10f}")

