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
n_var_steps   = 10
D_target = 4
chi = 5 * D_target
output_no = 8
max_ctm_sweeps = 20
tol_ctm = 1e-10


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
    A = psi[site]
    A.requires_grad_(True)


for it in range(1, n_var_steps + 1):

#     # for site in geometry.sites():
#     #     A = psi[site]
#     #     if getattr(A, "grad", None) is not None:
#     #         A.grad = None # type: ignore
    
    env_ctm = fpeps.EnvCTM(psi, init="eye")

    opts_svd_ctm = {
        "D_total": chi,
        "tol": tol_ctm
        # 'tol_block' can also be set if needed
    }

    ctm_generator = env_ctm.ctmrg_(opts_svd=opts_svd_ctm, iterator=True, max_sweeps=max_ctm_sweeps)

    nn_op = (n_up - I / 2) @ (n_dn - I / 2)
    ev_nn_dict = env_ctm.measure_1site(nn_op)
    ev_nn = mean_dict_values(ev_nn_dict)
    ev_cdagc_up_dict = env_ctm.measure_nn(cdag_up, c_up)
    ev_cdagc_dn_dict = env_ctm.measure_nn(cdag_dn, c_dn)

    ev_cdagc_up = mean_dict_values(ev_cdagc_up_dict)
    ev_cdagc_dn = mean_dict_values(ev_cdagc_dn_dict)

    energy_yastn = -2.0 * t * (ev_cdagc_up + ev_cdagc_dn) + U * ev_nn

    energy_yastn.requires_grad_(True)
    # print(energy_yastn)
    energy_yastn.backward()

    # torch.no_grad()

    with torch.no_grad():
        for site in geometry.sites():
            A = psi[site]
            g = psi[site].grad()
            # print(g)
            if g.data is None:
                continue                         # some sites might not contribute (shouldn't happen, but safe)

            # print(type(learning_rate))
            # print(type(g.data))
            psi[site] = psi[site] - learning_rate * g
    
    print(psi)
    

    if it == 1 or it % 1 == 0:
        print(f"[var-step {it:4d}] E/site = {energy_yastn.data: .12f}")

