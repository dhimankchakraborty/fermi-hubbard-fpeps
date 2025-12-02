import yastn.tn.fpeps as fpeps
import yastn.operators as op_mod
from functions_fpeps import *




L_x, L_y = 2, 2
t = 1.0
U = 10.0
mu = 0.0
dtau = 0.05
n_steps = 10
D_target = 4       # Target bond dimension
chi = 5 * D_target
max_sweeps = 10
backend = 'torch'
n_var_steps = 10
learning_rate = 1


geometry = fpeps.SquareLattice(dims=(L_x, L_y), boundary="obc")

ops = op_mod.SpinfulFermions(sym="U1xU1", backend=backend)

psi = hubbard_fpeps_torch(geometry, ops)
gates = hubbard_gates(geometry, t, dtau, mu, U, ops)
psi = run_simple_updates(psi, gates, D_target, n_steps, which="NN")

for site in geometry.sites():
    psi[site].requires_grad_(True) # type: ignore

energy = measure_energy_ctm(psi, chi, max_sweeps, geometry, ops, U, t)

for it in range(1, n_var_steps + 1):

    energy.backward()

    with torch.no_grad():
        for site in geometry.sites():
            g = psi[site].grad() # type: ignore
            if g is None:
                continue
            psi[site] = psi[site] - learning_rate * g
    
    for site in geometry.sites():
        psi[site].requires_grad_(True) # type: ignore
    
    energy = measure_energy_ctm(psi, chi, max_sweeps, geometry, ops, U, t)

    # print('-' * 80)
    # print(f"Energy after {it} steps of variational update: {energy}")
    # # print(f"Error: {abs((energy_yastn.item() - ed_energy_per_site) * 100 / ed_energy_per_site)} %")
    # print('-' * 80)
    print(energy)