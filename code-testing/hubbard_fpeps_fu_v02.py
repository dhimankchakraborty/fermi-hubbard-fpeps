import torch
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
dtau = 0.05
n_steps = 10
D_target = 8
chi = 5 * D_target
output_no = 4


ops = op_mod.SpinfulFermions(
    sym="U1xU1",
    backend='torch',
    default_dtype='float64'
)

I = ops.I()
c_up,  cdag_up,  n_up  = ops.c('u'),  ops.cp('u'),  ops.n('u')
c_dn,  cdag_dn,  n_dn  = ops.c('d'),  ops.cp('d'),  ops.n('d')

state_up = ops.vec_n((1, 0))
state_dn = ops.vec_n((0, 1)) 

geometry = fpeps.SquareLattice(
    dims=(L_x, L_y),
    boundary="obc"
)

vectors = {}
for (x, y) in geometry.sites():
    if (x + y) % 2 == 0:
        vectors[(x, y)] = state_up
    else:
        vectors[(x, y)] = state_dn

psi = fpeps.product_peps(
    geometry=geometry,
    vectors=vectors
)

g_hop_up = fpeps.gates.gate_nn_hopping(t, dtau / 2, I, c_up, cdag_up)
g_hop_dn = fpeps.gates.gate_nn_hopping(t, dtau / 2, I, c_dn, cdag_dn)
g_loc = fpeps.gates.gate_local_Coulomb(mu, mu, U, dtau / 2, I, n_up, n_dn)

gates = fpeps.gates.distribute(
    geometry,
    gates_nn=[g_hop_up, g_hop_dn],
    gates_local=g_loc
)

opts_svd_full = {
    "D_total": D_target
}

opts_svd_ctm = {
    "D_total": chi,
    "tol": 1e-10,
}

env_ctm = fpeps.EnvCTM(psi,init="eye")
env_ctm.opts_svd = opts_svd_ctm

energy_arr = np.zeros((n_steps))
infoss = []


for step in tqdm(range(n_steps)):
    infos = fpeps.evolution_step_(
        env_ctm,            # FULL-UPDATE ENVIRONMENT (CTM)
        gates,
        opts_svd=opts_svd_full,
        method='NN',       # default; 'NN' splits into NN gates if using longer-range MPO gates
        fix_metric=1        # project negative eigenvalues of metric -> more stable truncation
        # opts_post_truncation could be used to update CTM after each bond,
        # but here we keep the CTM fixed during this time step, as in Lubasch FU.
    )