import torch
from scipy.sparse.linalg import eigsh
import numpy as np
import yastn
import yastn.tn.fpeps as fpeps
import yastn.operators as op_mod
from tqdm import tqdm
from functions_fpeps import *
from functions_ed import *
# import matplotlib.pyplot as plt




def mean_dict_values(d):
    vals = list(d.values())
    return sum(vals) / len(vals) # if vals else 0.0


def hubbard_fpeps_torch(geometry, ops):

    state_up = ops.vec_n((1, 0))
    state_dn = ops.vec_n((0, 1))

    vectors = {}
    for (x, y) in geometry.sites():
        if (x + y) % 2 == 0:
            vectors[(x, y)] = state_up
        else:
            vectors[(x, y)] = state_dn


    psi = fpeps.product_peps(geometry=geometry, vectors=vectors)

    return psi


def hubbard_gates(geometry, t, dtau, mu, U, ops):

    I = ops.I()
    c_up, cdag_up, n_up = ops.c('u'), ops.cp('u'), ops.n('u')
    c_dn, cdag_dn, n_dn = ops.c('d'), ops.cp('d'), ops.n('d')

    g_hop_up = fpeps.gates.gate_nn_hopping(t, dtau / 2, I, c_up, cdag_up)
    g_hop_dn = fpeps.gates.gate_nn_hopping(t, dtau / 2, I, c_dn, cdag_dn)
    g_loc = fpeps.gates.gate_local_Coulomb(mu, mu, U, dtau / 2, I, n_up, n_dn)

    gates = fpeps.gates.distribute(
        geometry,
        gates_nn=[g_hop_up, g_hop_dn],
        gates_local=g_loc
    )

    return gates


def run_simple_updates(psi, gates, D_target, n_steps, which="NN"):
    env_ntu = fpeps.EnvNTU(psi, which=which)
    opts_svd_ntu = {
        "D_total": D_target,
        # "tol": 1e-8,
        # "cutoff": 0.0,
    }

    infoss = []
    for step in tqdm(range(n_steps)):
        infos = fpeps.evolution_step_(env_ntu, gates, opts_svd=opts_svd_ntu)
        infoss.append(infos)
    
    return env_ntu.psi


def measure_energy_ctm(psi, chi, max_sweeps, geometry, ops, U, t):
    env_ctm = fpeps.EnvCTM(psi, init="eye")
    opts_svd_ctm = {
        "D_total": chi,
        "tol": 1e-10,
    }

    ctm_iter = env_ctm.ctmrg_(
        opts_svd=opts_svd_ctm,
        iterator=True,
        max_sweeps=max_sweeps,
    )

    energy_old = 0.0
    tol_energy = 1e-7
    sites_list = list(geometry.sites())
    bonds_list = list(geometry.bonds())
    num_sites = len(sites_list)
    num_bonds = len(bonds_list)
    z_eff = 2.0 * num_bonds / num_sites

    I = ops.I()
    c_up, cdag_up, n_up = ops.c('u'), ops.cp('u'), ops.n('u')
    c_dn, cdag_dn, n_dn = ops.c('d'), ops.cp('d'), ops.n('d')


    for info in ctm_iter:
        on_site_op = (n_up - I / 2) @ (n_dn - I / 2)
        ev_loc = env_ctm.measure_1site(on_site_op)
        ev_loc_mean = mean_dict_values(ev_loc)

        ev_cdagc_up = env_ctm.measure_nn(cdag_up, c_up)
        ev_cdagc_dn = env_ctm.measure_nn(cdag_dn, c_dn)
        ev_cdagc_up_mean = mean_dict_values(ev_cdagc_up)
        ev_cdagc_dn_mean = mean_dict_values(ev_cdagc_dn)

        E_kin_site = - z_eff * t * (ev_cdagc_up_mean + ev_cdagc_dn_mean)
        E_loc_site = U * ev_loc_mean
        energy = E_kin_site + E_loc_site

        # if abs(energy - energy_old) < tol_energy:
        #     break

        
    
    return energy # type: ignore