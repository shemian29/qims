import qims as qims
import numpy as np
import qutip as qt
from tqdm.notebook import tqdm
import scipy.linalg

def PartBasis(size):
    bsl = {}
    bsl_ind = {}
    for l in tqdm(range(1,size)):
        bsl[l], bsl_ind[l] = qims.basis(int(l), parallel = False, bc = "open")
    return bsl, bsl_ind

def ent_entropy(Hilbert_k_dims, Ukevecs, basis, basis_ind, partial_basis, size, check = False):

    k_list = np.arange(0, size) / size

    print("Generating mapping between partial and full basis states for Schmidt decomposition:")
    PsiInds = {}
    for l in range(1, size):
        scan = []
        for indL in range(len(partial_basis[l])):
            for indR in range(len(partial_basis[size - l])):
                tmp = qims.state2ind(list(qims.ind2state(partial_basis[l][indL], size=l)) \
                                     + list(qims.ind2state(partial_basis[size - l][indR], size=size - l)))


                if qims.idstates(tmp, size=size) != -1:
                    scan.append([indL, indR, basis_ind[tmp]])
        PsiInds[l] = scan
        # if l==1:
            # print(PsiInds[l])

    S = {}
    flag = 0
    # print("test")
    print("Evaluate entanglement entropy for each momentum, eigenstate and sub-system size:")
    for K in tqdm(k_list):
        # Uk = qt.Qobj(U[K])

        for n in tqdm(range(Hilbert_k_dims[K])):
            # OP = ((U[K] * evecs[K][n]).full().T[0])
            OP = Ukevecs[K][n].full().T[0]

            for l in range(1, size):



                Psi = np.zeros((len(partial_basis[l]), len(partial_basis[size - l])), dtype=complex)


                for inds in PsiInds[l]:
                    Psi[inds[0], inds[1]] = OP[inds[2]]#[0, 0]

                # u, s, vh = np.linalg.svd(Psi, full_matrices=True)
                s = 10 ** (-20)+ scipy.linalg.svd(Psi,
                                                    full_matrices=False,
                                                    compute_uv=False,
                                                    check_finite=False,
                                                    overwrite_a=True
                                                    )



                S[K, n, l] = -np.dot(s ** 2, np.log(s ** 2))



                if check and size <=10:
                    v = np.zeros(2 ** size, dtype=complex)
                    v[[basis[r] for r in range(len(basis))]] = Ukevecs[K][n].full().T[0]
                    v = qt.Qobj(v, dims=[(2 * np.ones(size, dtype=int)).tolist(), [1]])

                    lmbds = 10**(-20)+v.ptrace(tuple(np.arange(0,l))).eigenenergies()
                    lmbds = [lmbds[r] for r in range(len(lmbds)) if lmbds[r]>0]
                    tmp = np.abs(-np.dot(lmbds, np.log(lmbds)) - S[K, n, l])
                    flag = 1
                    if tmp > 10**(-10):
                        print("Error: ",tmp, [K,n,l])
                        flag = 2

    if flag == 1:
        print("Successful check: calculation matches in two independent ways")
    elif flag == 2:
        print("Checks done: WARNING ERRORS FOUND")
    else:
        print("No checks performed, either by choice or system size was too large")

    return S

