# import torch
# import torch.optim as optim
from scipy.sparse.linalg import eigsh
import numpy as np
import yastn
import yastn.tn.fpeps as fpeps
import yastn.operators as op_mod
from tqdm import tqdm
from functions_fpeps import *
from functions_ed import *
# import matplotlib.pyplot as plt


NUM_THREADS = 8 
torch.set_num_threads(NUM_THREADS)




L = 2
L_x, L_y = L, L
t = 1.0
U = 10.0
mu = 0.0
learning_rate = 1
n_var_steps = 10
ntu_steps = 20
D_target = 4
chi = 5 * D_target
output_no = 8
max_ctm_sweeps = 20
tol_ctm = 1e-10
dtau = 0.05



print('-'*80)
H_sparse_shifted = build_hamiltonian_shifted(L_x, L_y, (L_x * L_y) // 2, (L_x * L_y) // 2, t, U, use_pbc=False)
eigenvalues_shifted, _ = eigsh(H_sparse_shifted, k=1, which='SA')
ed_energy_per_site = eigenvalues_shifted[0] / (L_x * L_y)
print(f"Exact Ground State Energy per Site: {ed_energy_per_site}")
print('-'*80)


ops = yastn.operators.SpinfulFermions(sym="U1xU1", backend='torch')

I = ops.I()
c_up, cdag_up, n_up = ops.c('u'), ops.cp('u'), ops.n('u')
c_dn, cdag_dn, n_dn = ops.c('d'), ops.cp('d'), ops.n('d')

state_up = ops.vec_n((1, 0))
state_dn = ops.vec_n((0, 1))

geometry = fpeps.SquareLattice(dims=(L_x, L_y), boundary="obc")

vectors = {}
for (x, y) in geometry.sites():
    if (x + y) % 2 == 0:
        vectors[(x, y)] = state_up
    else:
        vectors[(x, y)] = state_dn

psi = fpeps.product_peps(geometry=geometry, vectors=vectors)

g_hop_up = fpeps.gates.gate_nn_hopping(t, dtau / 2, I, c_up, cdag_up)
g_hop_dn = fpeps.gates.gate_nn_hopping(t, dtau / 2, I, c_dn, cdag_dn)
g_loc = fpeps.gates.gate_local_Coulomb(mu, mu, U, dtau / 2, I, n_up, n_dn)

gates = fpeps.gates.distribute(
    geometry,
    gates_nn=[g_hop_up, g_hop_dn],
    gates_local=g_loc
)

env_ntu = fpeps.EnvNTU(psi, which="NN")
opts_svd_ntu = {
    "D_total": D_target,
}

infoss = []
for step in tqdm(range(ntu_steps)):
    infos = fpeps.evolution_step_(env_ntu, gates, opts_svd=opts_svd_ntu)
    infoss.append(infos)

opts_svd_ctm = {
    "D_total": chi,
    "tol": tol_ctm
}

energy_yastn = compute_energy_ctm(psi, opts_svd_ctm, max_ctm_sweeps, I, n_up, n_dn, cdag_up, cdag_dn, c_up, c_dn, U, t)
print(energy_yastn)

for site in geometry.sites():
    psi[site].requires_grad_(True)



energy_yastn = compute_energy_ctm(psi, opts_svd_ctm, max_ctm_sweeps, I, n_up, n_dn, cdag_up, cdag_dn, c_up, c_dn, U, t)

# print('-' * 80)
# print(f"Energy after {ntu_steps} steps of simple update (CTM eval): {energy_yastn.item()}")
# print(f"Error: {abs((energy_yastn.item() - ed_energy_per_site) * 100 / ed_energy_per_site)} %")
# print('-' * 80)

for it in range(1, n_var_steps + 1):

    # for site in geometry.sites():
    #     if psi[site].grad is not None:
    #         psi[site].grad.zero_()
    
    energy_yastn = compute_energy_ctm(psi, opts_svd_ctm, max_ctm_sweeps, I, n_up, n_dn, cdag_up, cdag_dn, c_up, c_dn, U, t)

    energy_yastn.backward()

    with torch.no_grad():
        for site in geometry.sites():
            g = psi[site].grad()
            if g is None:
                continue
            psi[site] = psi[site] - learning_rate * g
    
    for site in geometry.sites():
        psi[site].requires_grad_(True)

    print('-' * 80)
    print(f"Energy after {it} steps of variational update: {energy_yastn.item()}")
    # print(f"Error: {abs((energy_yastn.item() - ed_energy_per_site) * 100 / ed_energy_per_site)} %")
    print('-' * 80)