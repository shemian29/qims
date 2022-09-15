import numpy as np
import qutip as qt
import scipy.sparse
import tqdm.notebook

import qims as qims

#adding test comment
eta = 0.636

def constr(st, r, size):
    return (1 - qims.occ(st, (r - 1) % size, size)) * (1 - qims.occ(st, (r + 1) % size, size))


def pxp(st, r, size):
    j = size - r - 1
    b = 1
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
    h = 0
    for r in tqdm.notebook.tqdm(range(size)):
        scan = []
        for st in range(len(basis)):

            if constr(basis[st], r, size) == 1:
                scan.append([basis_ind[pxp(basis[st], r, size)], st, 1])

        elms = np.array(scan)

        row = elms.T[0]
        col = elms.T[1]
        data = elms.T[2]
        h = h + scipy.sparse.csr_matrix((data, (row, col)), shape=(len(basis), len(basis)))

    return qt.Qobj(h)


def sp(st, r, size):
    j = size - r - 1
    b = 1
    b <<= j  # compute a "mask" integer b which is 1 on site j and zero elsewhere
    st ^= b
    return st


def pxp_operators(basis, basis_ind, size, prms):
    """
    PXP Hamiltonian
    :param basis: dictionary with mapped basis indices
    :param basis_ind: dictionary inverse of basis dictionary
    :param size: system size
    :return: Hamiltonian as a qutip object
    """
    SP = 0
    sp = {}
    sz = {}
    prj = {}
    # for r in tqdm.notebook.tqdm(range(size)):
    for r in range(size):
        scan = []
        scanz = []
        for st in range(len(basis)):
            # print()
            scanz.append([st, st, 2 * qims.occ(basis[st], r, size) - 1])
            if constr(basis[st], r, size) == 1:



                if (r % 2)==0 and 0 == qims.occ(basis[st], r, size):
                    # print(r, qims.ind2state(st, size))
                    scan.append([basis_ind[pxp(basis[st], r, size)], st, 1])
                elif (r % 2)==1 and 1 == qims.occ(basis[st], r, size):
                    scan.append([basis_ind[pxp(basis[st], r, size)], st, 1])
                else:
                    scan.append([basis_ind[pxp(basis[st], r, size)], st, 0])

        elms = np.array(scan)

        row = np.array(elms.T[0])
        col = np.array(elms.T[1])
        data = np.array(elms.T[2])

        sp[r] = qt.Qobj(scipy.sparse.csr_matrix((data, (row, col)), shape=(len(basis), len(basis))))
        SP = SP + scipy.sparse.csr_matrix((data, (row, col)), shape=(len(basis), len(basis)))


        elms = np.array(scanz)

        row = np.array(elms.T[0])
        col = np.array(elms.T[1])
        data = np.array(elms.T[2])
        sz[r] = qt.Qobj(scipy.sparse.csr_matrix((data, (row, col)), shape=(len(basis), len(basis))))
        prj[r] = (qt.identity(len(basis))-sz[r])/2
    Nx = size
    OP = {}

    sm = 0
    for r in range(Nx):
        sm = sm + prj[(r - 1) % Nx] * ((sp[r])) * prj[(r + 1) % Nx] * prj[(r + 2) % Nx]

    OP[0] = sm #+ sm.dag()

    sm = 0
    for r in range(Nx):
        if r % 2 == 0:
            sm = sm + prj[(r - 1) % Nx] * ((sp[r])) * prj[(r + 1) % Nx] * sz[(r + 2) % Nx] * prj[(r + 3) % Nx]
        else:
            sm = sm + prj[(r - 1) % Nx] * ((sp[r].dag())) * prj[(r + 1) % Nx] * sz[(r + 2) % Nx] * prj[(r + 3) % Nx]

    OP[1] = sm #+ sm.dag()

    sm = 0
    for r in range(Nx):
        if r % 2 == 0:
            sm = sm + prj[(r - 1) % Nx] * ((sp[r])) * prj[(r + 1) % Nx] * prj[(r + 2) % Nx] * prj[(r + 3) % Nx]
        else:
            sm = sm + prj[(r - 1) % Nx] * ((sp[r].dag())) * prj[(r + 1) % Nx] * prj[(r + 2) % Nx] * prj[(r + 3) % Nx]

    OP[2] = sm #+ sm.dag()

    sm = 0
    for r in range(Nx):
        sm = sm + prj[(r - 2) % Nx] * prj[(r - 1) % Nx] * ((sp[r])) * prj[(r + 1) % Nx] * prj[(r + 2) % Nx]

    OP[3] = sm #+ sm.dag()

    sm = 0
    for r in range(Nx):
        if r % 2 == 0:
            sm = sm + prj[(r - 2) % Nx] * prj[(r - 1) % Nx] * ((sp[r])) * prj[(r + 1) % Nx] * sz[(r + 2) % Nx] * prj[
                (r + 3) % Nx]
        else:
            sm = sm + prj[(r - 2) % Nx] * prj[(r - 1) % Nx] * ((sp[r].dag())) * prj[(r + 1) % Nx] * sz[(r + 2) % Nx] * \
                 prj[(r + 3) % Nx]

    OP[4] = sm #+ sm.dag()

    sm = 0
    for r in range(Nx):
        if r % 2 == 0:
            sm = sm + prj[(r - 2) % Nx] * prj[(r - 1) % Nx] * ((sp[r])) * prj[(r + 1) % Nx] * prj[(r + 2) % Nx] * prj[
                (r + 3) % Nx]
        else:
            sm = sm + prj[(r - 2) % Nx] * prj[(r - 1) % Nx] * ((sp[r].dag())) * prj[(r + 1) % Nx] * prj[(r + 2) % Nx] * \
                 prj[(r + 3) % Nx]

    OP[5] = sm #+ sm.dag()

    sm = 0
    for r in range(Nx):
        if r % 2 == 0:
            sm = sm + prj[(r - 1) % Nx] * ((sp[r])) * prj[(r + 1) % Nx] * sz[(r + 2) % Nx] * prj[(r + 3) % Nx] * prj[
                (r + 4) % Nx]
        else:
            sm = sm + prj[(r - 1) % Nx] * ((sp[r].dag())) * prj[(r + 1) % Nx] * sz[(r + 2) % Nx] * prj[(r + 3) % Nx] * \
                 prj[(r + 4) % Nx]

    OP[6] = sm #+ sm.dag()

    sm = 0
    for r in range(Nx):
        if r % 2 == 0:
            sm = sm + prj[(r - 1) % Nx] * ((sp[r])) * prj[(r + 1) % Nx] * prj[(r + 2) % Nx] * prj[(r + 3) % Nx] * prj[
                (r + 4) % Nx]
        else:
            sm = sm + prj[(r - 1) % Nx] * ((sp[r].dag())) * prj[(r + 1) % Nx] * prj[(r + 2) % Nx] * prj[(r + 3) % Nx] * \
                 prj[(r + 4) % Nx]

    OP[7] = sm #+ sm.dag()

    sm = 0
    for r in range(Nx):
        if r % 2 == 0:
            sm = sm + prj[(r - 2) % Nx] * prj[(r - 1) % Nx] * ((sp[r])) * prj[(r + 1) % Nx] * sz[(r + 2) % Nx] * prj[
                (r + 3) % Nx] * prj[(r + 4) % Nx]
        else:
            sm = sm + prj[(r - 2) % Nx] * prj[(r - 1) % Nx] * ((sp[r].dag())) * prj[(r + 1) % Nx] * sz[(r + 2) % Nx] * \
                 prj[(r + 3) % Nx] * prj[(r + 4) % Nx]

    OP[8] = sm #+ sm.dag()

    SP = qt.Qobj(SP)
    for r in range(len(OP)):
        SP = SP + prms[r]*OP[r]

    Sx = (SP + SP.dag()) / (2*eta)
    Sy = (SP - SP.dag()) / (2 * 1j * eta)
    Sz = 0.5 * (SP * SP.dag() - SP.dag() * SP)/(eta**2)
    S2 = Sx * Sx + Sy * Sy + Sz * Sz

    TP = (Sz - 1j * Sy) / 2

    return Sx, Sy, Sz, S2, OP


def sz_neel(basis, size):
    row = np.arange(0, len(basis))
    col = np.arange(0, len(basis))
    neel = 2 * np.array([np.ones(int(size / 2)), np.zeros(int(size / 2))]).T.flatten() - 1
    data = np.array([np.sum(neel * (2 * qims.ind2state(basis[r], size) - 1)) for r in range(len(basis))])

    return qt.Qobj(scipy.sparse.csr_matrix((data, (row, col)), shape=(len(basis), len(basis))))
