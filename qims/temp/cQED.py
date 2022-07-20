import os

import numpy as np
import qutip as qt
from pympler.asizeof import asizeof
from scipy.sparse import csr_matrix
from tqdm.notebook import tqdm


# Setup of the coupler Hamiltonian

def Op(Os, N, r):
    ops = []
    Id = qt.identity(Os.shape[0])

    if (r > 0) and (r < N - 1):
        [ops.append(Id) for r in range(r)]
        ops.append(Os)
        [ops.append(Id) for r in range(N - r - 1)]
        return qt.tensor(ops)

    elif r == 0:
        ops.append(Os)
        [ops.append(Id) for r in range(N - 1)]
        return qt.tensor(ops)

    elif r == N - 1:
        [ops.append(Id) for r in range(N - 1)]
        ops.append(Os)
        return qt.tensor(ops)


def HF_phi(phi_ext, N, Ncut, EJ, EC, EJb, ECb):
    Ncharge = qt.charge(Ncut)
    Tp = qt.Qobj(np.diag(np.ones(2 * Ncut + 1 - 1), 1))
    Tm = Tp.dag()
    CosPhi = (Tp + Tm) / 2

    EE = np.linalg.inv((1 / 8) * np.diag(1 / EC) + (1 / 8) * (1 / ECb) * np.ones((N, N)))

    # Charging energies
    ENN = 0
    for r2 in range(0, N):
        for r1 in range(0, N):
            ENN = ENN + 0.5 * EE[r1, r2] * Op(Ncharge, N, r1) * Op(Ncharge, N, r2)

    # Josephson inductive energies
    EJJ = 0
    for r in range(N):
        EJJ = EJJ - EJ[r] * Op(CosPhi, N, r)

    # Interacting cosine term
    if N > 0:
        smExpPPhi = 1
        for r in range(0, N):
            smExpPPhi = smExpPPhi * Op(Tp, N, r)

        phase = np.exp(2 * np.pi * 1j * phi_ext)
        sum_CosPhi = EJb * (phase * smExpPPhi + np.conjugate(phase) * smExpPPhi.dag()) / 2
        HF = ENN + EJJ - sum_CosPhi
    else:
        HF = ENN + EJJ
    return (HF)


# Basic operators of the system


def Phase_j(r1, N, Ncut):
    M = np.diag(np.ones(2 * Ncut + 1 - 1), k=1)

    M[0, 2 * Ncut + 1 - 1] = 1
    M[2 * Ncut + 1 - 1, 0] = 1

    return (Op(qt.Qobj(M), N, r1))


def Charge_j(r1, N, Ncut):
    M = qt.charge(Ncut, -Ncut)

    return (Op(qt.Qobj(M), N, r1))


def NN(r1, r2, N, Ncut):
    Ncharge = qt.charge(Ncut)

    return (Op(Ncharge, N, r1) * Op(Ncharge, N, r2))


def Cos_phi12(r1, r2, phi_ext, N, Ncut):
    Tp = qt.Qobj(np.diag(np.ones(2 * Ncut + 1 - 1), 1))
    Tm = Tp.dag()
    CosPhi = (Tp + Tm) / 2

    if N > 0:
        smExpPPhi = 1
        smExpPPhi = Op(Tp, N, r1) * Op(Tp, N, r2)

        phase = np.exp(2 * np.pi * 1j * phi_ext)
        smCosPhi = (phase * smExpPPhi + np.conjugate(phase) * smExpPPhi.dag()) / 2
        HF = - smCosPhi
    else:
        HF = 0

    return (HF)


def TensorArrange(O, DIMS):
    return qt.Qobj(O.full(), dims=DIMS)


# Funnctions for symmetry-resolved system


# def Index2Ket(n, b, N):
#     if n == 0:
#         return [-int((b-1)/2) for r in range(N)]
#     digits = []
#     while n:
#         digits.append(int(n % b))
#         n //= b

#     digits = (np.array(digits[::-1])-int((b-1)/2)).tolist()
#     for r in range(N-len(digits)):
#         digits.insert(0,-int((b-1)/2))

#     return digits

# def Ket2Index(st,b):
#     return int(''.join(map(str, st+int((b-1)/2))),b)

def T1(v, Ket2Index):
    return Ket2Index[str(np.roll(v, -1))]
    # return np.roll(v, -1)


def Ket(ns, Ncut):
    return qt.tensor([qt.basis(2 * Ncut + 1, ns[r] + Ncut) for r in range(len(ns))])


def ToKet(coefs, sts, bs, Ncut):
    sm = 0
    for r in range(len(coefs)):
        sm = sm + coefs[r] * Ket(bs[sts[r]], Ncut)

    return sm


def ind2occ(s, r, N, Ncut):
    return int((s // ((2 * Ncut + 1) ** (N - r - 1))) % (2 * Ncut + 1))


def ind2state(s, N, Ncut):
    return np.array([ind2occ(s, r, N, Ncut) for r in range(N)]) - Ncut


def state2ind(state, N, Ncut):
    sps = (2 * Ncut + 1) ** (N - np.arange(N) - 1)
    return np.dot(state + Ncut, sps)


def TransInd(s, N, Ncut):
    return int(
        ind2occ(s, N - 1, N, Ncut) * ((2 * Ncut + 1) ** (N - 1)) - (ind2occ(s, N - 1, N, Ncut) / (2 * Ncut + 1)) + s / (
                    2 * Ncut + 1))


def GenerateMomentumBasis(N, Ncut):
    dms = (2 * Ncut + 1) ** N
    lst = list(range(dms))
    it = 1
    vbs = []

    while len(lst) > 0:

        np.savetxt("Monitor_" + str([N, Ncut]) + ".txt", [(100 * len(lst) / dms), asizeof(vbs) / (10 ** 9)],
                   fmt='%1.4f')

        bas = [lst[0]]
        tmp = TransInd(bas[-1], N, Ncut)

        while (tmp - bas[0]) != 0:
            bas.append(tmp)
            tmp = TransInd(bas[-1], N, Ncut)

        vbs.append(bas)

        lst = list(set(lst) - set(bas))

    return vbs


def amps(cycle, k):
    N0 = len(cycle)
    seq = np.arange(N0)
    return (1 / np.sqrt(N0)) * np.exp(-1j * 2 * np.pi * (k) * seq)


def ChargeToTranslation(N, Ncut):
    flnms = os.listdir()
    flag = 0
    for fname in flnms:
        if 'vbs_' + str((N, Ncut)) + '.npz' in fname:
            flag = 1
    if flag == 0:
        vbs = np.array(GenerateMomentumBasis(N, Ncut), dtype=object)
        np.savez_compressed('vbs_' + str((N, Ncut)), vbs=vbs)
    else:
        loaded = np.load('vbs_' + str((N, Ncut)) + '.npz', allow_pickle=True)
        vbs = loaded['vbs'].tolist()

    DIMS = [[2 * Ncut + 1 for r in range(N)], [2 * Ncut + 1 for r in range(N)]]

    mms = np.array([n / N for n in range(N)])
    lns = np.unique(list(map(len, vbs)))

    m_seqs = [[] for r in range(N)]
    for N0 in lns:
        tmp = np.array([n / N0 for n in range(N0)])
        for k in range(N):
            if mms[k] in tmp:
                m_seqs[k].append(N0)

    V = []
    sm = 0
    print('Full Hilbert space dimension = ', (2 * Ncut + 1) ** N)
    for n in tqdm(range(N)):
        row = []
        col = []
        data = []
        ind = 0
        for state in tqdm(range(len(vbs))):
            if len(vbs[state]) in m_seqs[n]:
                vec = amps(vbs[state], n / N)
                row.append(vbs[state])
                col.append((ind * np.ones(len(vec), dtype=int)).tolist())
                data.append(vec.tolist())
                ind = ind + 1
        col = [item for sublist in col for item in sublist]
        row = [item for sublist in row for item in sublist]
        data = [item for sublist in data for item in sublist]
        print('Sector ' + str(n) + '=', ind)
        sm = sm + ind
        V.append(qt.Qobj(csr_matrix((data, (row, col)),
                                    shape=((2 * Ncut + 1) ** N, ind)), dims=DIMS))
    print('Sum of sector dimensions = ', sm)
    if sm != (2 * Ncut + 1) ** N:
        np.savetxt('WARNING_' + str([N, Ncut]) + '.txt', [1])
    return V


def SectorDiagonalization(H, V, Nvals):
    N = len(H.dims[0])
    Ncut = int((H.dims[0][0] - 1) / 2)

    #     V = ChargeToTranslation(N,Ncut)

    data = []
    #     for k in tqdm(range(len(V))):
    for k in range(len(V)):
        HF0 = V[k].dag() * H * V[k]
        evals, evecs = HF0.eigenstates(eigvals=Nvals, sparse=True)
        data.append([evals, [V[k] * evecs[r] for r in range(len(evecs))]])

    return data


def SortedDiagonalization(H, V, Nvals):
    tst = SectorDiagonalization(H, V, Nvals)

    vecs = []
    eigs = []
    for k in range(len(tst)):
        tmp = np.array(tst[k][1])
        tmpe = np.array(tst[k][0])
        for r in range(len(tmp)):
            vecs.append(tmp[r].T[0])
            eigs.append(tmpe[r])
    eigs = np.array(eigs)
    vecs = np.array(vecs)
    vecs = vecs[np.argsort(eigs)]
    eigs = eigs[np.argsort(eigs)]

    return eigs, vecs, tst


def SectorDiagonalization_Energies(H, V, Nvals):
    N = len(H.dims[0])
    Ncut = int((H.dims[0][0] - 1) / 2)

    #     V = ChargeToTranslation(N,Ncut)

    data = []
    #     for k in tqdm(range(len(V))):
    for k in range(len(V)):
        HF0 = V[k].dag() * H * V[k]
        evals = HF0.eigenenergies(eigvals=Nvals, sparse=True)
        data.append(evals)

    return data


def SortedDiagonalization_Energies(H, V, Nvals):
    tst = SectorDiagonalization_Energies(H, V, Nvals)

    eigs = np.sort(np.concatenate(tst))

    return eigs, tst


# Functions to obtain Wigner distributions


def Wmatrix(phi, n, m, mp, N):
    return (1 / (2 * np.pi)) * np.exp(1j * (m - mp) * phi / N) * np.sin((n - (m + mp) / 2) * np.pi / N + 0.0000001) / (
                (n - (m + mp) / 2) * np.pi / N + 0.0000001)


def WignerPoint(n, phi, Ncut, rho, N):
    xaxis = np.arange(-Ncut, Ncut + 1)
    yaxis = np.arange(-Ncut, Ncut + 1)
    return np.real(np.sum(rho * Wmatrix(phi, n, xaxis[:, None], yaxis[None, :], N)))


def WignerArray(rho, N):
    Ncut = int((len(rho) - 1) / 2)
    #     print(Ncut)
    n_list = np.arange(-Ncut, Ncut + 1)
    #     print(len(n_list))
    phi_list = 2 * np.pi * N * np.arange(0, 2 * Ncut + 1) / (2 * Ncut + 1)

    scanT = []
    for ph in tqdm(phi_list):
        scan = []
        for n in n_list:
            scan.append(WignerPoint(n, ph, Ncut, rho, N))
        scanT.append(scan)

    return np.array(scanT)


def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out

# Useful code


# Explicit calculation of the symmetry operator

# T = np.zeros(((2*Ncut+1)**N,(2*Ncut+1)**N))
# scan = []
# for r in range(len(bs)):
#     flag = 0
#     it = 0
#     while flag ==0:
#         if np.linalg.norm(bs[it]-T1(bs[r]))==0.0:
# #                 if np.linalg.norm(bs[it]-np.roll(bs[r],-1))==0.0:
#             flag = 1
#             scan.append([it,r])
#             T[it,r]=1
#         it = it+1
# T = qt.Qobj(T,dims = [[2*Ncut+1 for r in range(N)],[2*Ncut+1 for r in range(N)]])