import numpy as np
from scipy.sparse.linalg import eigsh
import time
from functions_ed import *




L_x, L_y = 2, 2
N_sites = L_x * L_y
N_up, N_down = 2, 2 # Half-filling
t = 1.0
U = 10 

print(f"\nBuilding Hamiltonian for U = {U}")

start_time_standard = time.time()
H_sparse_standard = build_hamiltonian_standard(L_x, L_y, N_up, N_down, t, U, use_pbc=False)
end_time_standard = time.time()
eigenvalues_standard, _ = eigsh(H_sparse_standard, k=1, which='SA')

print("Standard Model")
print(f"Total Ground State Energy: {eigenvalues_standard[0]}")
print(f"Ground State Energy per Site: {eigenvalues_standard[0] / (L_x * L_y)}")
print(f"Time Taken: {end_time_standard - start_time_standard}")
print()
print()


start_time_shifted = time.time()
H_sparse_shifted = build_hamiltonian_shifted(L_x, L_y, N_up, N_down, t, U, use_pbc=False)
end_time_shifted = time.time()
eigenvalues_shifted, _ = eigsh(H_sparse_shifted, k=1, which='SA')

print("Shifted Model")
print(f"Ground State Energy: {eigenvalues_shifted[0]}")
print(f"Ground State Energy per Site: {eigenvalues_shifted[0] / (L_x * L_y)}")
print(f"Time Taken: {end_time_shifted - start_time_shifted}")
