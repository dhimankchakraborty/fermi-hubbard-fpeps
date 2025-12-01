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
n_var_steps   = 1
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

for site in geometry.sites():
    psi[site].requires_grad_(True)

# # Gradient Check (Working Correctly)
# for site in geometry.sites():
#     if not psi[site].requires_grad:
#         print(f"Warning: Site {site} is not tracking gradients!")
#     else:
#         print(f"Site {site} is tracking gradients!")

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

# print(psi)
# print('-'*80)

infoss = []
for step in tqdm(range(10)):
    infos = fpeps.evolution_step_(env_ntu, gates, opts_svd=opts_svd_ntu)
    infoss.append(infos)

# print(psi)

# # Gradient Check (Working Correctly)
# for site in geometry.sites():
#     if not psi[site].requires_grad:
#         print(f"Warning: Site {site} is not tracking gradients!")
#     else:
#         print(f"Site {site} is tracking gradients!")

for site in geometry.sites():
    print(psi[site])


# for it in range(1, n_var_steps + 1):

#     env_ctm = fpeps.EnvCTM(psi, init="eye")

#     # # Gradient Check Done Here (Working Correctly)

#     opts_svd_ctm = {
#         "D_total": chi,
#         "tol": tol_ctm
#         # 'tol_block' can also be set if needed
#     }

#     ctm_generator = env_ctm.ctmrg_(opts_svd=opts_svd_ctm, iterator=True, max_sweeps=max_ctm_sweeps)

#     for info in ctm_generator:
#         pass  # This executes the loop steps one by one

#     # # Gradient Check Done Here (Working Correctly)

#     nn_op = (n_up - I / 2) @ (n_dn - I / 2)
#     ev_nn_dict = env_ctm.measure_1site(nn_op)

#     # for key in ev_nn_dict:
#     #     print(f"{key} : {ev_nn_dict[key]}")

#     # print(ev_nn_dict)

#     ev_cdagc_up_dict = env_ctm.measure_nn(cdag_up, c_up)
#     ev_cdagc_dn_dict = env_ctm.measure_nn(cdag_dn, c_dn)

#     # print(ev_cdagc_up_dict)
#     # print(ev_cdagc_dn_dict)
#     # # Gradient Check Done Here (Working Correctly)

#     ev_nn_values = list(ev_nn_dict.values())
#     ev_nn = torch.stack(ev_nn_values).mean()

#     ev_cdagc_up_values = list(ev_cdagc_up_dict.values())
#     ev_cdagc_up = torch.stack(ev_cdagc_up_values).mean()

#     ev_cdagc_dn_values = list(ev_cdagc_dn_dict.values())
#     ev_cdagc_dn = torch.stack(ev_cdagc_dn_values).mean()

#     print(ev_nn)
#     print(ev_cdagc_dn)
#     print(ev_cdagc_up)

#     energy_yastn = -2.0 * t * (ev_cdagc_up + ev_cdagc_dn) + U * ev_nn

#     energy_yastn.backward()

#     with torch.no_grad():
#         for site in geometry.sites():
#             A = psi[site]
#             g = psi[site].grad()
#             # print(g)
#             if g.data is None:
#                 continue                         # some sites might not contribute (shouldn't happen, but safe)

#             # print(type(learning_rate))
#             # print(type(g.data))
#             psi[site] = psi[site] - learning_rate * g
    
#     print(psi)




