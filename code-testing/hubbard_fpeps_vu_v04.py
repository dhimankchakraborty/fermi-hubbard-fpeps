import torch
import torch.optim as optim
from scipy.sparse.linalg import eigsh
import numpy as np
import yastn
import yastn.tn.fpeps as fpeps
import yastn.operators as op_mod
from tqdm import tqdm
from functions_fpeps import *
from functions_ed import *
import matplotlib.pyplot as plt


NUM_THREADS = 8 
torch.set_num_threads(NUM_THREADS)




L_x, L_y = 2, 2
t = 1.0
U = 10.0
mu = 0.0
learning_rate = 0.01
n_var_steps = 2
ntu_steps = 10
D_target = 4
chi = 5 * D_target
output_no = 8
max_ctm_sweeps = 20
tol_ctm = 1e-10
dtau = 0.05


config_kwargs = {
    "backend": "torch",
    "default_dtype": "float64",
    "device": "cpu"
}


print('-'*80)
H_sparse_shifted = build_hamiltonian_shifted(L_x, L_y, (L_x * L_y) // 2, (L_x * L_y) // 2, t, U, use_pbc=False)
eigenvalues_shifted, _ = eigsh(H_sparse_shifted, k=1, which='SA')
ed_energy_per_site = eigenvalues_shifted[0] / (L_x * L_y)
print(f"Exact Ground State Energy per Site: {ed_energy_per_site}")
print('-'*80)


ops = yastn.operators.SpinfulFermions(sym="U1xU1", **config_kwargs)

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

env_ntu = fpeps.EnvNTU(psi, which="NN++")
opts_svd_ntu = {
    "D_total": D_target,
}

infoss = []
for step in tqdm(range(ntu_steps)):
    infos = fpeps.evolution_step_(env_ntu, gates, opts_svd=opts_svd_ntu)
    infoss.append(infos)

for site in geometry.sites():
    psi[site].requires_grad_(True)

opts_svd_ctm = {
    "D_total": chi,
    "tol": tol_ctm
}
env_ctm = fpeps.EnvCTM(psi, init="eye")
ctm_generator = env_ctm.ctmrg_(opts_svd=opts_svd_ctm, iterator=True, max_sweeps=max_ctm_sweeps)
nn_op = (n_up - I / 2) @ (n_dn - I / 2)
ev_nn_dict = env_ctm.measure_1site(nn_op)
ev_cdagc_up_dict = env_ctm.measure_nn(cdag_up, c_up)
ev_cdagc_dn_dict = env_ctm.measure_nn(cdag_dn, c_dn)
ev_nn_values = list(ev_nn_dict.values())
ev_nn = torch.stack(ev_nn_values).mean()
ev_cdagc_up_values = list(ev_cdagc_up_dict.values())
ev_cdagc_up = torch.stack(ev_cdagc_up_values).mean()
ev_cdagc_dn_values = list(ev_cdagc_dn_dict.values())
ev_cdagc_dn = torch.stack(ev_cdagc_dn_values).mean()
energy_yastn = -2.0 * t * (ev_cdagc_up + ev_cdagc_dn) + U * ev_nn

print('-'*80)
print(f"Energy after {ntu_steps} steps of simple update: {energy_yastn.data}")
print(f"Error: {abs((energy_yastn.data - ed_energy_per_site) * 100 / ed_energy_per_site)} %")
print('-'*80)


for it in range(1, n_var_steps + 1):

    energy_yastn.backward()

    # with torch.no_grad():
    for site in geometry.sites():
        g = psi[site].grad()
        if g.data is None:
            continue

        psi[site] = psi[site] - learning_rate * g
    
    # env_ctm = fpeps.EnvCTM(psi, init="eye")
    # ctm_generator = env_ctm.ctmrg_(opts_svd=opts_svd_ctm, iterator=True, max_sweeps=max_ctm_sweeps)
    nn_op = (n_up - I / 2) @ (n_dn - I / 2)
    ev_nn_dict = env_ctm.measure_1site(nn_op)
    ev_cdagc_up_dict = env_ctm.measure_nn(cdag_up, c_up)
    ev_cdagc_dn_dict = env_ctm.measure_nn(cdag_dn, c_dn)
    ev_nn_values = list(ev_nn_dict.values())
    ev_nn = torch.stack(ev_nn_values).mean()
    ev_cdagc_up_values = list(ev_cdagc_up_dict.values())
    ev_cdagc_up = torch.stack(ev_cdagc_up_values).mean()
    ev_cdagc_dn_values = list(ev_cdagc_dn_dict.values())
    ev_cdagc_dn = torch.stack(ev_cdagc_dn_values).mean()
    energy_yastn = -2.0 * t * (ev_cdagc_up + ev_cdagc_dn) + U * ev_nn

    print('-'*80)
    print(f"Energy after {it} steps of variational update: {energy_yastn.data}")
    print(f"Error: {abs((energy_yastn.data - ed_energy_per_site) * 100 / ed_energy_per_site)} %")
    print('-'*80)