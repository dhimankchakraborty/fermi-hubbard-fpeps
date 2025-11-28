import numpy as np
import scipy.sparse as sparse
from itertools import combinations




def count_set_bits(n):
    """Counts the number of set bits (1s) in an integer."""
    count = 0
    while n > 0:
        n &= (n - 1)
        count += 1
    return count



def get_fermi_sign(state, i, j):
    """
    Calculates the fermionic sign for hopping from j to i.
    Assumes i < j.
    """
    mask = (1 << j) - (1 << (i + 1))
    bits_between = count_set_bits(state & mask)
    return (-1) ** bits_between



def generate_basis(N_sites, N_elec):
    """
    Generates the basis states for a given number of sites and electrons.
    A state is represented as an integer.
    Returns:
        - basis: A list of the basis state integers.
        - state_to_index: A dictionary mapping the integer state to its index.
    """
    basis = []
    state_to_index = {}

    # Iterate over all combinations of N_elec electrons in N_sites
    for sites in combinations(range(N_sites), N_elec):
        state = 0
        for site in sites:
            state |= (1 << site)

        state_to_index[state] = len(basis)
        basis.append(state)

    return np.array(basis, dtype=np.int64), state_to_index



def get_neighbors(L_x, L_y, use_pbc=False):
    """
    Generates a list of nearest-neighbor pairs.
    Supports open (use_pbc=False) or periodic (use_pbc=True) boundary conditions.
    """
    N_sites = L_x * L_y
    neighbors = []
    for i in range(N_sites):
        x, y = i % L_x, i // L_x

        if use_pbc:
            # Periodic Boundary Conditions
            j_right = ((x + 1) % L_x) + y * L_x
            neighbors.append((i, j_right))

            j_down = x + ((y + 1) % L_y) * L_x
            neighbors.append((i, j_down))
        else:
            # print('Hello')
            # Open Boundary Conditions
            # Right neighbor (only if not on the right edge)
            if (x + 1) < L_x:
                j_right = (x + 1) + y * L_x
                neighbors.append((i, j_right))

            # Down neighbor (only if not on the bottom edge)
            if (y + 1) < L_y:
                j_down = x + (y + 1) * L_x
                neighbors.append((i, j_down))

    return neighbors



def build_hamiltonian_standard(L_x, L_y, N_up, N_down, t, U, use_pbc=False):
    """
    Builds the sparse Hamiltonian matrix for the Hubbard model.
    """
    N_sites = L_x * L_y

    # 1. Generate basis for up and down spins
    # print(f"Generating basis for {N_up} up spins...")
    basis_up, map_up = generate_basis(N_sites, N_up)
    # print(f"Generating basis for {N_down} down spins...")
    basis_down, map_down = generate_basis(N_sites, N_down)

    D_up = len(basis_up)
    D_down = len(basis_down)
    D_total = D_up * D_down

    # if D_total > 200000: # Safety check
    #     print(f"Warning: Hilbert space dimension is {D_total}. This may be slow.")

    # print(f"Total Hilbert space dimension D = {D_up} x {D_down} = {D_total}")

    # Use lists to build the sparse matrix in COO format
    row_indices = []
    col_indices = []
    matrix_data = []

    # Get neighbor list based on boundary condition
    neighbor_pairs = get_neighbors(L_x, L_y, use_pbc=use_pbc)

    # 2. Iterate over all basis states |n> = |up_n> |down_n>
    for n_idx_up, state_up in enumerate(basis_up):
        for n_idx_down, state_down in enumerate(basis_down):

            n_idx_total = n_idx_up * D_down + n_idx_down

            # --- U-term (Diagonal) ---
            # H_nn = <n|H|n>
            double_occupancy = count_set_bits(state_up & state_down)
            H_nn = U * double_occupancy

            # Add diagonal element
            row_indices.append(n_idx_total)
            col_indices.append(n_idx_total)
            matrix_data.append(H_nn)

            # --- t-term (Off-diagonal) ---
            # H_mn = <m|H|n>
            # We apply H_t to |n> and see which |m> states it connects to.
            # H_t = -t * sum_{<i,j>, s} (c_i_s^+ c_j_s + c_j_s^+ c_i_s)

            for i, j in neighbor_pairs:
                # --- Hopping up-spin ---

                # Try c_i^+ c_j (hop from j to i)
                if (state_up & (1 << j)) and not (state_up & (1 << i)):
                    new_state_up = state_up ^ (1 << j) ^ (1 << i)
                    m_idx_up = map_up[new_state_up]
                    m_idx_total = m_idx_up * D_down + n_idx_down

                    sign = get_fermi_sign(state_up, min(i, j), max(i, j))

                    row_indices.append(m_idx_total)
                    col_indices.append(n_idx_total)
                    matrix_data.append(-t * sign)

                # Try c_j^+ c_i (hop from i to j)
                if (state_up & (1 << i)) and not (state_up & (1 << j)):
                    new_state_up = state_up ^ (1 << i) ^ (1 << j)
                    m_idx_up = map_up[new_state_up]
                    m_idx_total = m_idx_up * D_down + n_idx_down

                    sign = get_fermi_sign(state_up, min(i, j), max(i, j))

                    row_indices.append(m_idx_total)
                    col_indices.append(n_idx_total)
                    matrix_data.append(-t * sign)

                # --- Hopping down-spin ---

                # Try c_i^+ c_j (hop from j to i)
                if (state_down & (1 << j)) and not (state_down & (1 << i)):
                    new_state_down = state_down ^ (1 << j) ^ (1 << i)
                    m_idx_down = map_down[new_state_down]
                    m_idx_total = n_idx_up * D_down + m_idx_down

                    sign = get_fermi_sign(state_down, min(i, j), max(i, j))

                    row_indices.append(m_idx_total)
                    col_indices.append(n_idx_total)
                    matrix_data.append(-t * sign)

                # Try c_j^+ c_i (hop from i to j)
                if (state_down & (1 << i)) and not (state_down & (1 << j)):
                    new_state_down = state_down ^ (1 << i) ^ (1 << j)
                    m_idx_down = map_down[new_state_down]
                    m_idx_total = n_idx_up * D_down + m_idx_down

                    sign = get_fermi_sign(state_down, min(i, j), max(i, j))

                    row_indices.append(m_idx_total)
                    col_indices.append(n_idx_total)
                    matrix_data.append(-t * sign)

    # 3. Create the sparse matrix
    # print("Building sparse matrix...")
    H = sparse.csc_matrix((matrix_data, (row_indices, col_indices)),
                          shape=(D_total, D_total), dtype=np.float64)

    return H



def build_hamiltonian_shifted(L_x, L_y, N_up, N_down, t, U, use_pbc=False):
    """
    Builds the sparse Hamiltonian matrix for the *shifted* Hubbard model:

        H = -t sum_<i,j>,s (c^dagger_{i,s} c_{j,s} + h.c.)
            + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2)

    Note:
        In a fixed (N_up, N_down) sector, the interaction can be written as
            U * [ double_occupancy
                  - 0.5 * (N_up + N_down)
                  + 0.25 * N_sites ]
        where:
            double_occupancy = sum_i n_{i,up} n_{i,down}
            N_sites = L_x * L_y

        The last two terms are *constant shifts* in this sector.
        For half-filling on 2x2 (N_up = N_down = 2, N_sites = 4) the
        constant shift is: -U * N_sites / 4 = -U.
    """
    N_sites = L_x * L_y

    # 1. Generate basis for up and down spins
    basis_up, map_up = generate_basis(N_sites, N_up)
    basis_down, map_down = generate_basis(N_sites, N_down)

    D_up = len(basis_up)
    D_down = len(basis_down)
    D_total = D_up * D_down

    # Use lists to build the sparse matrix in COO format
    row_indices = []
    col_indices = []
    matrix_data = []

    # Get neighbor list based on boundary condition
    neighbor_pairs = get_neighbors(L_x, L_y, use_pbc=use_pbc)

    # 2. Iterate over all basis states |n> = |up_n> |down_n>
    for n_idx_up, state_up in enumerate(basis_up):
        for n_idx_down, state_down in enumerate(basis_down):

            # Combined basis index = tensor product index
            n_idx_total = n_idx_up * D_down + n_idx_down

            # -------------------------------------------------
            # --- Diagonal term: shifted Hubbard interaction ---
            # -------------------------------------------------
            #
            # double_occupancy = sum_i n_{i,up} n_{i,down}
            #
            # In this fixed-(N_up, N_down) sector:
            #
            #   sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2)
            #     = sum_i n_{i,up} n_{i,down}
            #       - 1/2 * (sum_i n_{i,up} + sum_i n_{i,down})
            #       + 1/4 * N_sites
            #
            #   = double_occupancy
            #     - 0.5 * (N_up + N_down)
            #     + 0.25 * N_sites
            #
            # Because in this sector:
            #   sum_i n_{i,up}   = N_up
            #   sum_i n_{i,down} = N_down
            #
            # So we can compute the diagonal element as:
            #
            #   H_nn = U * [
            #               double_occupancy
            #               - 0.5 * (N_up + N_down)
            #               + 0.25 * N_sites
            #             ]
            #
            # Note the last two pieces are constant across all basis states.

            double_occupancy = count_set_bits(state_up & state_down)  # integer: #sites with both spins

            H_nn = U * (
                double_occupancy
                - 0.5 * (N_up + N_down)
                + 0.25 * N_sites
            )

            # Add diagonal element
            row_indices.append(n_idx_total)
            col_indices.append(n_idx_total)
            matrix_data.append(H_nn)

            # ------------------------------------------
            # --- Off-diagonal hopping term (kinetic) ---
            # ------------------------------------------
            #
            # H_t = -t * sum_<i,j>,s (c_i,s^+ c_j,s + c_j,s^+ c_i,s)
            #
            # Same as your original implementation.

            for i, j in neighbor_pairs:
                # --- Hopping up-spin ---

                # Try c_i^+ c_j (hop from j to i)
                if (state_up & (1 << j)) and not (state_up & (1 << i)):
                    new_state_up = state_up ^ (1 << j) ^ (1 << i)
                    m_idx_up = map_up[new_state_up]
                    m_idx_total = m_idx_up * D_down + n_idx_down

                    sign = get_fermi_sign(state_up, min(i, j), max(i, j))

                    row_indices.append(m_idx_total)
                    col_indices.append(n_idx_total)
                    matrix_data.append(-t * sign)

                # Try c_j^+ c_i (hop from i to j)
                if (state_up & (1 << i)) and not (state_up & (1 << j)):
                    new_state_up = state_up ^ (1 << i) ^ (1 << j)
                    m_idx_up = map_up[new_state_up]
                    m_idx_total = m_idx_up * D_down + n_idx_down

                    sign = get_fermi_sign(state_up, min(i, j), max(i, j))

                    row_indices.append(m_idx_total)
                    col_indices.append(n_idx_total)
                    matrix_data.append(-t * sign)

                # --- Hopping down-spin ---

                # Try c_i^+ c_j (hop from j to i)
                if (state_down & (1 << j)) and not (state_down & (1 << i)):
                    new_state_down = state_down ^ (1 << j) ^ (1 << i)
                    m_idx_down = map_down[new_state_down]
                    m_idx_total = n_idx_up * D_down + m_idx_down

                    sign = get_fermi_sign(state_down, min(i, j), max(i, j))

                    row_indices.append(m_idx_total)
                    col_indices.append(n_idx_total)
                    matrix_data.append(-t * sign)

                # Try c_j^+ c_i (hop from i to j)
                if (state_down & (1 << i)) and not (state_down & (1 << j)):
                    new_state_down = state_down ^ (1 << i) ^ (1 << j)
                    m_idx_down = map_down[new_state_down]
                    m_idx_total = n_idx_up * D_down + m_idx_down

                    sign = get_fermi_sign(state_down, min(i, j), max(i, j))

                    row_indices.append(m_idx_total)
                    col_indices.append(n_idx_total)
                    matrix_data.append(-t * sign)

    # 3. Create the sparse matrix
    H = sparse.csc_matrix(
        (matrix_data, (row_indices, col_indices)),
        shape=(D_total, D_total),
        dtype=np.float64
    )

    return H

