import numpy as np
import qutip as qt
import scipy.sparse
import tqdm.notebook

import QMB as qmb


def constr(st, r, size):
    return (1 - qmb.occ(st, (r - 1) % size, size)) * (1 - qmb.occ(st, (r + 1) % size, size))


def pxp(st, r, size):
    j = size - r - 1  # compute lattice site in reversed bit configuration (cf QuSpin convention for mapping from bits to sites)
    b = 1;
    b <<= j  # compute a "mask" integer b which is 1 on site j and zero elsewhere
    st ^= b
    return st


def pxp_hamiltonian(basis, basis_ind, size):
    """
    PXP Hamiltonian
    :param basis: dictionary with mapped basis indices
    :param basis_ind: dictionary inverse of basis dictionary
    :param size: system size
    :return: Hamiltonian as a qutip object
    """
    H = 0
    for r in tqdm.notebook.tqdm(range(size)):
        scan = []
        for st in range(len(basis)):

            if constr(basis[st], r, size) == 1:
                scan.append([basis_ind[pxp(basis[st], r, size)], st, 1])

        elms = np.array(scan)

        row = np.array(elms.T[0])
        col = np.array(elms.T[1])
        data = np.array(elms.T[2])
        H = H + scipy.sparse.csr_matrix((data, (row, col)), shape=(len(basis), len(basis)))

    return qt.Qobj(H)


def sz_neel(basis, size):
    row = np.arange(0, len(basis))
    col = np.arange(0, len(basis))
    neel = 2 * np.array([np.ones(int(size / 2)), np.zeros(int(size / 2))]).T.flatten() - 1
    data = np.array([np.sum(neel * (2 * qmb.ind2state(basis[r], size) - 1)) for r in range(len(basis))])

    return qt.Qobj(scipy.sparse.csr_matrix((data, (row, col)), shape=(len(basis), len(basis))))
