# Fermi-Hubbard Model Simulation using fPEPS

This repository contains a Python implementation for simulating the **Fermi-Hubbard model** on 2D lattices using **Fermionic Projected Entangled Pair States (fPEPS)**. It also includes **Exact Diagonalization (ED)** codes for benchmarking small systems.

## 📌 Project Overview

The main goal of this project is to explore the ground state properties of strongly correlated fermionic systems governed by the Hubbard Hamiltonian. The code utilizes tensor network methods (fPEPS) to efficiently represent the many-body wavefunction in two dimensions, overcoming the exponential scaling of Hilbert space encountered in exact diagonalization.

### 📂 Repository Structure

- **`hubbard_fpeps.py`**: The main executable script for running the fPEPS simulation. It likely sets up the lattice, Hamiltonian parameters, and initiates the variational optimization or imaginary time evolution.
- **`functions_fpeps.py`**: Contains the core library of functions for the fPEPS algorithm. This includes tensor contraction routines, fermionic swap gate implementations, and environment tensor updates.
- **`functions_ed.py`**: Helper functions for performing Exact Diagonalization. Useful for verifying fPEPS results on small lattice sizes (e.g., 2x2 or 2x3).
- **`exact_diagonalization/`**: A directory dedicated to ED scripts, results, or comparison benchmarks.
- **`archives/`**: Contains older versions or deprecated code.

## ⚛️ Theoretical Background

### The Fermi-Hubbard Model
The system is modeled by the Fermi-Hubbard Hamiltonian, which describes fermions on a lattice interacting via on-site repulsion. The Hamiltonian is given by:

$$
\hat{H} = -t \sum_{\langle i,j \rangle, \sigma} \left( \hat{c}_{i\sigma}^\dagger \hat{c}_{j\sigma} + \text{h.c.} \right) + U \sum_i \hat{n}_{i\uparrow} \hat{n}_{i\downarrow} - \mu \sum_{i,\sigma} \hat{n}_{i\sigma}
$$

Where:
- $\hat{c}_{i\sigma}^\dagger$ and $\hat{c}_{i\sigma}$ are the creation and annihilation operators for a fermion at site $i$ with spin $\sigma \in \{\uparrow, \downarrow\}$.
- $\hat{n}_{i\sigma} = \hat{c}_{i\sigma}^\dagger \hat{c}_{i\sigma}$ is the number operator.
- $t$ is the hopping amplitude (kinetic energy).
- $U$ is the on-site Coulomb repulsion strength.
- $\mu$ is the chemical potential controlling the filling.
- $\langle i,j \rangle$ denotes summation over nearest-neighbor pairs.

### Fermionic PEPS (fPEPS)
Projected Entangled Pair States (PEPS) generalize Matrix Product States (MPS) to 2D. For fermions, standard PEPS must be augmented to handle the anti-commutation relations of fermionic operators:
$$\{ \hat{c}_i, \hat{c}_j^\dagger \} = \delta_{ij}$$

This implementation likely uses **fermionic swap gates** or **parity-preserving tensors** to ensure that the contraction of the tensor network respects the global fermionic sign structure. The ansatz for the wavefunction $|\Psi\rangle$ is formed by contracting local tensors $A^{[i]}$ on a 2D grid:

$$|\Psi\rangle = \sum_{\vec{s}} \text{tTr}(A^{[1]s_1} A^{[2]s_2} \dots ) |\vec{s}\rangle$$

where $\text{tTr}$ denotes the tensor trace over auxiliary bond indices.

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.x installed along with the essential scientific computing libraries.
```bash
pip install numpy scipy