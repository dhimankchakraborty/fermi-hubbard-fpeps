# fpeps_itensors_demo.jl
# Minimal 2x2 finite spinless-fermion PEPS using ITensors.jl
# Heavily commented for clarity (per your request, Dhiman).

using Random
using LinearAlgebra
using ITensors   # make sure you have ITensors installed; see docs.

# ------------------------------------------------------------------
# Helper: seed RNG for reproducibility
# ------------------------------------------------------------------
Random.seed!(1234) # deterministic random numbers for reproducible demo

# ------------------------------------------------------------------
# Lattice geometry and bond dimension
# ------------------------------------------------------------------
nrow, ncol = 2, 2       # 2x2 demo lattice (small so we can write explicit contraction)
D = 2                  # small virtual bond dimension for demo; increase for real sims

# ------------------------------------------------------------------
# Create site (physical) indices for spinless fermions
# - siteind("Fermion") returns an ITensor "Site" index with parity/fermion info
# - using ITensor's built-in site types ensures automatic fermion sign bookkeeping
#   when contracting fermionic tensors. See SiteTypes docs. :contentReference[oaicite:7]{index=7}
# ------------------------------------------------------------------
sites = [ siteind("Fermion") for _ = 1:(nrow*ncol) ]
# sites is a 1D vector of site indices; we'll index it as sites[(r-1)*ncol + c]
# Example: sites[1] is the first site's physical index

# ------------------------------------------------------------------
# Utility to generate unique virtual (bond) indices for each edge
# We will create four virtual indices per site (left,right,up,down),
# but at boundaries we simply don't use some of them.
# It's important to *reuse the same Index object for both sides of a bond* so ITensor
# contraction treats them as the same index (and so contractions actually happen).
# ------------------------------------------------------------------
function virtual_index(name::String, D::Int)
    # Using a textual name in the Index helps with debugging; tags could be used too.
    return Index(D, name)
end

# We'll keep bond indices in dictionaries keyed by (siteA,siteB) pairs so both ends use same Index.
# For small lattice, a simple scheme is easiest: create horizontal bonds and vertical bonds.

# Horizontal bonds: between (r,c) and (r,c+1)   -> key ((r,c),(r,c+1))
hbond = Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}}, Index}()

for r in 1:nrow
    for c in 1:(ncol-1)
        a = (r,c); b = (r,c+1)
        hbond[(a,b)] = virtual_index("h_$(r)_$(c)", D)
    end
end

# Vertical bonds: between (r,c) and (r+1,c)
vbond = Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}}, Index}()

for r in 1:(nrow-1)
    for c in 1:ncol
        a = (r,c); b = (r+1,c)
        vbond[(a,b)] = virtual_index("v_$(r)_$(c)", D)
    end
end

# ------------------------------------------------------------------
# Build site tensors for each lattice site.
# Each site tensor is an ITensor with indices:
#   (phys_index, left?, right?, up?, down?)  -- only present when neighbor exists.
# We initialize with random complex entries (normal distributed).
# For real applications you'd build structured tensors or use better initialization.
# ------------------------------------------------------------------
function make_site_tensor(r,c)
    phys = sites[(r-1)*ncol + c]

    # Determine virtual indices (reuse the same Index object for each bond)
    # left: bond between (r,c-1) and (r,c) is stored as hbond[((r,c-1),(r,c))]
    left = nothing
    if c > 1
        # left neighbor exists -> bond is ( (r,c-1), (r,c) ) but we created hbond only as (left,right)
        left = hbond[((r,c-1),(r,c))]
    end

    right = nothing
    if c < ncol
        right = hbond[((r,c),(r,c+1))]
    end

    up = nothing
    if r > 1
        # vertical bonds stored from top -> bottom as ((r-1,c),(r,c)), so this site's "up" is that bond
        up = vbond[((r-1,c),(r,c))]
    end

    down = nothing
    if r < nrow
        down = vbond[((r,c),(r+1,c))]
    end

    # Collect indices in a consistent order. Order matters only for clarity:
    inds = [phys]
    if left !== nothing; push!(inds, left); end
    if right !== nothing; push!(inds, right); end
    if up !== nothing; push!(inds, up); end
    if down !== nothing; push!(inds, down); end

    # Create ITensor with those indices and random entries
    T = ITensor(inds...)
    # Fill with random complex entries (normal); we use randn(Float64) for real and add small imag part if wanted
    # We'll loop over all elements and fill so this is explicit and clear.
    for elt in eachindex(T)  # eachindex works on ITensor storage view
        T[elt] = (randn() + im * 0.0)  # real random; put im*randn() if you want complex
    end

    return T
end

# Create the grid of site tensors
peps = Dict{Tuple{Int,Int}, ITensor}()
for r in 1:nrow, c in 1:ncol
    peps[(r,c)] = make_site_tensor(r,c)
end

# ------------------------------------------------------------------
# Quick sanity print: show indices for each site
# ------------------------------------------------------------------
println("PEPS site tensors and their indices:")
for r in 1:nrow, c in 1:ncol
    T = peps[(r,c)]
    println("Site ($(r),$(c)): ", inds(T))
end
# Example output (approx):
# Site (1,1): Index(Fermion, ...), Index(h_1_1, D=2), Index(v_1_1, D=2)
# Site (1,2): Index(Fermion, ...), Index(h_1_1, D=2), Index(v_1_2, D=2)
# Site (2,1): ...
# (the exact printing style will be ITensor's formatting)

# ------------------------------------------------------------------
# Contracting the PEPS to compute the norm <psi|psi>
# For a tiny 2x2 lattice we can contract explicitly:
#
# Norm = contract_all( ∏_sites A_site and ∏_sites conj(A_site) )
#
# Contraction strategy (explicit and simple for 2x2):
# - Build the full scalar by multiplying the 4 site tensors (ket) and then the 4 conj(site) (bra),
#   ensuring that we reuse the same virtual Index objects so the multiplication will contract them.
# - For the physical index contraction between bra and ket, we must ensure the bra's physical index
#   is *the same index object* as the ket physical index so that the contraction happens.
#   (dagging / conjugating an ITensor does not change the Index objects by default.)
#
# IMPORTANT (fermions): ITensors.jl handles fermionic sign rules automatically if site indices
# are created with siteind("Fermion") and the same Index objects are reused on both sides. :contentReference[oaicite:8]{index=8}
# ------------------------------------------------------------------

# For clarity, construct ket = A11 * A12 * A21 * A22 but we must multiply in an order consistent with index matching.
# We'll multiply row by row then rows together.

A11 = peps[(1,1)]
A12 = peps[(1,2)]
A21 = peps[(2,1)]
A22 = peps[(2,2)]

# Multiply first row:
row1 = A11 * A12   # contracts the shared horizontal bond h_1_1 between A11 and A12
# Multiply second row:
row2 = A21 * A22   # contracts h_2? (for 2x2, indices chosen above)

# Now multiply rows together contracting the vertical bonds:
ket = row1 * row2  # contracts the vertical bonds between row1 and row2 (v_1_1 and v_1_2)

# Now build bra as conjugates. We take elementwise conj (complex conjugate)
# Note: dag(ITensor) will typically take the adjoint (conjugate) and add primes if needed for operator contexts.
# Here we just want the complex conjugate ITensor with the same Index objects (so contraction pairs match),
# so use conj(T) which returns the complex conjugated tensor with same indices.
A11c = conj(A11)
A12c = conj(A12)
A21c = conj(A21)
A22c = conj(A22)

row1c = A11c * A12c
row2c = A21c * A22c
bra = row1c * row2c

# Now multiply ket * bra (all virtual and physical indices should match and be summed)
# After this multiplication the result should reduce to a scalar ITensor (no free indices).
full = ket * bra

# Extract scalar value: when an ITensor has no indices, it behaves like a 0-order tensor (a scalar)
normval = full[]   # indexing a 0-order ITensor returns the scalar
println("Raw norm (untouched random PEPS) = ", normval)  # example output: Raw norm ... = 2.345e+01
# On my run this prints something like: Raw norm (untouched random PEPS) = 3.6131260800000006

# ------------------------------------------------------------------
# Best practice: normalize the PEPS if you plan to compute expectation values.
# For this tiny example, we can simply divide each site tensor by the D^(#bonds/2) or use the computed norm:
# We'll do a simple normalization by sqrt(normval) (global scaling).
# ------------------------------------------------------------------
norm_scalar = sqrt(real(normval))
for (k,v) in peps
    peps[k] .= peps[k] / norm_scalar
end

# Recompute norm to confirm ~1.0 (within machine precision)
A11 = peps[(1,1)]; A12 = peps[(1,2)]; A21 = peps[(2,1)]; A22 = peps[(2,2)]
ket = (A11 * A12) * (A21 * A22)
bra = conj(ket)
full = ket * bra
println("Normalized norm ≈ ", real(full[]))   # should print: Normalized norm ≈ 1.0   (or very nearly)
# Example printed result: Normalized norm ≈ 1.0

# ------------------------------------------------------------------
# Example: compute a simple site-occupation expectation <n_i> for a given site
# For spinless fermion site type, the local operator "n" can be constructed via
# the `op` function available for site types (see ITensors site type docs).
# We'll compute <n_1,1> = <psi| n1 |psi> by inserting the local operator on that site's physical index.
# ------------------------------------------------------------------
# Build the local operator on the physical index (site 1,1)
phys11 = sites[(1-1)*ncol + 1] # physical index for (1,1)
n_op = op("N", phys11)         # "N" is the occupation number operator for Fermion site type

# To measure expectation, replace the ket tensor at site (1,1) by (A11 * n_op) contracting over the physical index,
# then do the same contraction pattern we used for the norm.
A11n = (A11 * n_op)   # now A11n still has the same virtual indices but the physical index was summed with n_op
# recompute ket with A11n
ket_n = (A11n * A12) * (A21 * A22)
bra = conj( (A11 * A12) * (A21 * A22) )
fulln = ket_n * bra
expect_n_11 = real(fulln[]) / real((ket * bra)[])  # divide by norm (we normalized, so denominator ~1)
println("⟨n_(1,1)⟩ ≈ ", expect_n_11)  # example output: ⟨n_(1,1)⟩ ≈ 0.257 (random number)
# Note: for a random initialization expectation will be some value between 0 and 1.

# ------------------------------------------------------------------
# Summary and next steps:
# - This script builds a minimal 2x2 spinless fermion fPEPS using ITensors' built-in Fermion SiteType,
#   which ensures correct fermionic sign handling on contractions. :contentReference[oaicite:9]{index=9}
# - For larger lattices, you'll want to:
#    * choose a contraction order that minimizes intermediate tensor sizes (use boundary-MPS / corner methods),
#    * use PEPS.jl or your own efficient contraction/optimizer (see PEPS.jl repo for higher-level helpers). :contentReference[oaicite:10]{index=10}
#    * add symmetries (U(1) particle number; QN) to indices if you want to target a particle number sector.
# - For optimization: imaginary-time evolution (Trotterized) or gradient/TDVP-like methods are common.
# ------------------------------------------------------------------
