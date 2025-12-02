import torch
import numpy as np
import yastn
import yastn.tn.fpeps as fpeps
import yastn.operators as op_mod
from tqdm import tqdm

# Settings
NUM_THREADS = 4
torch.set_num_threads(NUM_THREADS)

L_x, L_y = 2, 2
t = 1.0
U = 10.0
D_target = 4
chi = 20  # Bond dimension for CTM environment
epochs = 50
learning_rate = 0.1

# 1. Operators & Symmetry
ops = op_mod.SpinfulFermions(sym="U1xU1", backend='torch', default_dtype='float64')

c_up, cdag_up, n_up = ops.c('u'), ops.cp('u'), ops.n('u')
c_dn, cdag_dn, n_dn = ops.c('d'), ops.cp('d'), ops.n('d')

# 2. Initialization (AFM Product State)
geometry = fpeps.SquareLattice(dims=(L_x, L_y), boundary="obc")
state_up = ops.vec_n((1, 0))
state_dn = ops.vec_n((0, 1))

vectors = {}
for (x, y) in geometry.sites():
    vectors[(x, y)] = state_up if (x + y) % 2 == 0 else state_dn

# Create PEPS
psi = fpeps.product_peps(geometry=geometry, vectors=vectors)

# 3. Optimizer Setup
# Extract raw data blocks from yastn tensors for PyTorch optimizer
params = []
seen_ids = set() # Avoid duplicate parameters

for site in geometry.sites():
    tensor = psi[site]
    blocks_iter = [tensor.data]

    for block in blocks_iter:
        if id(block) not in seen_ids:
            block.requires_grad_(True)
            params.append(block)
            seen_ids.add(id(block))

optimizer = torch.optim.Adam(params, lr=learning_rate)

# 4. Optimization Loop
print(f"Starting AD Optimization (L={L_x}x{L_y}, U={U})")
print("-" * 60)

# Initialize Environment once (warm start)
env_ctm = fpeps.EnvCTM(psi, init="eye")
opts_svd_ctm = {"D_total": chi, "tol": 1e-6}

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # A. Run CTM (Forward Pass)
    # We run a fixed number of sweeps to build the differentiable graph
    # Note: iterator=False runs the loop internally
    env_ctm.ctmrg_(opts_svd=opts_svd_ctm, max_sweeps=10, iterator=False)
    
    # B. Measure Energy Terms
    # 1. Hopping (Kinetic) Energy: <c^dag_i c_j> + h.c.
    # measure_nn returns dict: {bond: value}
    nn_up = env_ctm.measure_nn(cdag_up, c_up)
    nn_dn = env_ctm.measure_nn(cdag_dn, c_dn)
    
    # Sum over all bonds.
    # Note: measure_nn usually gives one direction. We add h.c. (complex conj)
    # Since diagonal elements are real, for hopping: val + val.conj()
    E_kin = 0.0
    for val in nn_up.values():
        E_kin += -t * (val + val.conj())
    for val in nn_dn.values():
        E_kin += -t * (val + val.conj())
        
    # 2. Interaction (Potential) Energy: U * <n_up n_dn>
    # measure_1site returns dict: {site: value}
    nn_int = env_ctm.measure_1site(n_up @ n_dn)
    E_pot = sum(U * val for val in nn_int.values())
    
    # Total Energy (Loss)
    loss = E_kin + E_pot
    
    # C. Backpropagation
    loss.backward()
    
    # D. Update
    optimizer.step()
    
    print(f"Epoch {epoch+1:03d} | Total Energy: {loss.item():.6f}")

# 5. Final Results
print("-" * 60)
print("Final Measurement (High Precision)")

with torch.no_grad():
    # Run CTM until convergence for accurate observables
    env_ctm.ctmrg_(opts_svd={"D_total": chi, "tol": 1e-10}, max_sweeps=50, iterator=False)
    
    # Measure Observables
    ev_n_up = env_ctm.measure_1site(n_up)
    ev_n_dn = env_ctm.measure_1site(n_dn)
    ev_docc = env_ctm.measure_1site(n_up @ n_dn)

    # Averages
    def get_mean(d): return sum(d.values()).item() / len(d)
    
    print(f"Energy         : {loss.item():.6f}")
    print(f"<n_up>         : {get_mean(ev_n_up):.6f}")
    print(f"<n_dn>         : {get_mean(ev_n_dn):.6f}")
    print(f"Double Occ.    : {get_mean(ev_docc):.6f}")