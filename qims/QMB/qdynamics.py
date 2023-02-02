import qutip as qt
import numpy as np
import qims as qims
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

def SytSy_alt(SpinOp, bs, Nx, Ukevecs, evals):
    """

    :param SpinOp: spin operator odd under translation symmetry by one lattice site
    :param bs:
    :param Nx:
    :param Ukevecs:
    :param evals:
    :return:
    """
    Corr3 = {}
    Corrrand = {}
    wscar = 1
    wmax = 5 * wscar
    dw = wscar / 20
    tmax = (2 * wmax / (2 * wmax + dw)) * (2 * np.pi / dw)
    dt = 2 * np.pi / (2 * wmax + dw)
    tlist = np.linspace(0, tmax, int(tmax / dt) + 1)
    k_list = np.arange(0, Nx) / Nx
    wlist = np.linspace(-wmax, wmax, len(tlist))

    itmax = 500
    tmp = [qt.rand_ket(len(bs)) for _ in range(itmax)]
    sts_rand = qims.npqt2qtqt(tmp)

    for k in tqdm(k_list[0:int(Nx / 2)]):
        kit = ((k * Nx + int(Nx / 2)) % Nx) / Nx

        aux_k = qims.npqt2qtqt(Ukevecs[k])
        aux_kp = qims.npqt2qtqt(Ukevecs[kit])
        Skkp = aux_k.dag() * SpinOp * aux_kp
        Skkpaux_k = Skkp.dag() * aux_k.dag()
        Skkpaux_kp = Skkp * aux_kp.dag()

        rand_aux_k = (sts_rand.dag()) * aux_k
        rand_aux_kp = (sts_rand.dag()) * aux_kp
        rand_Skkpaux_k = Skkp.dag() * aux_k.dag() * sts_rand
        rand_Skkpaux_kp = Skkp * aux_kp.dag() * sts_rand

        for t in tqdm(tlist):
            phase_k = qt.Qobj(np.diag(np.exp(1j * evals[k] * t)))
            phase_kp = qt.Qobj(np.diag(np.exp(-1j * evals[kit] * t)))

            phkkp = phase_k * Skkp * phase_kp
            t1 = aux_k.full()
            t2 = (phkkp * Skkpaux_k).full()
            Corr3[k, t] = np.sum(t1.T * t2, axis=0)
            t1 = aux_kp.full()
            t2 = (phkkp.dag() * Skkpaux_kp).full()
            Corr3[kit, t] = np.sum(t1.T * t2, axis=0)

            t1 = rand_aux_k.full()
            t2 = (phkkp * rand_Skkpaux_k).full()
            Corrrand[k, t] = np.sum(t1.T * t2, axis=0)
            t1 = rand_aux_kp.full()
            t2 = (phkkp.dag() * rand_Skkpaux_kp).full()
            Corrrand[kit, t] = np.sum(t1.T * t2, axis=0)

    return Corr3, Corrrand, tlist, wlist

def SpinOp_t_SpinOp_0(Sx, SpinOp, bs, Nx, Ukevecs, evals, check = False):

    #Define temporal and frequency-domain grids with respect to scar frequency
    wscar = 1

    dw = wscar / 20
    wmax = 5 * wscar

    dt = 2 * np.pi / (2 * wmax + dw)
    tmax = (2 * wmax / (2 * wmax + dw)) * (2 * np.pi / (dw))

    tlist = np.linspace(0, tmax, int(tmax / dt) + 1)
    wlist = np.linspace(-wmax, wmax, len(tlist))

    itmax = 500
    tmp = [qt.rand_ket(len(bs)) for n in range(itmax)]
    sts_rand = qims.npqt2qtqt(tmp)

    #Calculate correlation functions for product states and random states
    Corr_bare = {}
    Corr_rand = {}
    k_list = np.arange(0, Nx) / Nx
    for k in tqdm(k_list[0:int(Nx / 2)]):

        # Momentum at k + pi
        kp = ((k * Nx + int(Nx / 2)) % Nx) / Nx

        SpinOp_data = {}

        # Eigenvalue from k and k + pi
        SpinOp_data["ek"] = evals[k]
        SpinOp_data["ekp"] = evals[kp]

        # Eigenvectors from k and k + pi in qutip matrix form
        # Eigenvectors represented in bare basis
        SpinOp_data["Uk"] = qims.npqt2qtqt(Ukevecs[k])
        SpinOp_data["Ukp"] = qims.npqt2qtqt(Ukevecs[kp])

        # Representation of SpinOp between k and k + pi eigenvectors
        # SpinOp initially provided in bare basis
        # This product is in the eigenvector basis
        SpinOp_data["Ukd_SpinOp_Ukp"] = SpinOp_data["Uk"].dag() * SpinOp * SpinOp_data["Ukp"]

        # Product of SpinOp^{\dagger}_{k,k+pi} with eigenvectors from k
        # This product is in the (eigenvector kp)-(bare k) basis
        SpinOp_data["Ukpd_SpinOp_Uk_Ukd"] = SpinOp_data["Ukd_SpinOp_Ukp"].dag() * SpinOp_data["Uk"].dag()

        # Product of SpinOp_{k,k+pi} with eigenvectors from kp
        # This product is in the (eigenvector k)-(bare kp) basis
        SpinOp_data["Ukd_SpinOp_Ukp_Ukpd"] = SpinOp_data["Ukd_SpinOp_Ukp"] * SpinOp_data["Ukp"].dag()

        # Calculate correlation function by parallel map on time list
        Corr_bare[k] = qt.parallel_map(Corr_t_map, tlist, task_kwargs=SpinOp_data, progress_bar=True)
        print("Done with bare states")

        SpinOp_data["Ukpd_SpinOp_Uk_Ukd_Vrand"] = SpinOp_data["Ukpd_SpinOp_Uk_Ukd"] * sts_rand
        SpinOp_data["Skkpaux_kp"] = SpinOp_data["Ukd_SpinOp_Ukp_Ukpd"] * sts_rand
        SpinOp_data["Uk"] = (sts_rand.dag()) * SpinOp_data["Uk"]
        SpinOp_data["Ukp"] = (sts_rand.dag()) * SpinOp_data["Ukp"]
        print("Starting with random states")
        return Corr_t_map(tlist[0],**SpinOp_data)
    #     Corr_rand[k] = qt.parallel_map(Corr_t_map, tlist, task_kwargs=SpinOp_data, progress_bar=False)
    #
    #
    # if check and Nx<=14:
    #
    #     # Array of possible momentum indices
    #     k_list = np.arange(0, int(Nx / 2)) / Nx
    #
    #     #Calculate correlation function summed over all momenta for bare states
    #     scan = []
    #     for t in range(len(tlist)):
    #         scan.append(np.sum([np.sum(Corr_bare[k][t], axis=0) for k in k_list], axis=0))
    #     scan = np.array(scan).T
    #
    #     #Calculate correlation function summed over all momenta for random states
    #     rand_scan = []
    #     for t in range(len(tlist)):
    #         rand_scan.append(np.sum([np.sum(Corr_rand[k][t], axis=0) for k in k_list], axis=0))
    #     rand_scan = np.array(rand_scan).T
    #
    #     scan_qutip = []
    #     rand_scan_qutip = []
    #     print("Initiating product state check")
    #     for r in tqdm(range(len(bs))):
    #         psi0 = qt.basis(len(bs), r)
    #         corr1 = qt.correlation_2op_1t(Sx, psi0, tlist, [], SpinOp, SpinOp)
    #         scan_qutip.append(corr1)
    #
    #     print("Initiating random states check")
    #     for r in tqdm(range(sts_rand.shape[1])):
    #         psi0 = qt.Qobj(sts_rand[:, r])
    #         corr1 = qt.correlation_2op_1t(Sx, psi0, tlist, [], SpinOp, SpinOp)
    #         rand_scan_qutip.append(corr1)
    #
    #     U = qt.propagator(Sx, tlist, c_op_list=[], args={}, options=None, unitary_mode='batch', parallel=True,
    #                       progress_bar=True, _safe_mode=True)
    #     tmp = [np.real((U[t].dag() * SpinOp * U[t] * SpinOp).diag()) for t in tqdm(range(len(tlist)))]
    #     errU = np.abs(np.abs(np.array(scan) - np.array(tmp).T))
    #
    #     err = np.abs(np.abs(np.array(scan) - np.array(scan_qutip)))
    #     errqtqt = np.abs(np.abs(np.array(scan_qutip) - np.array(tmp).T))
    #     # err_rand = np.abs(np.abs(np.array(rand_scan) - np.array(rand_scan_qutip)))
    #
    #
    #     plt.plot(errqtqt.T[1:]);
    #     plt.yscale('log')
    #     plt.show()
    #
    #     plt.plot(errU.T[1:]);
    #     plt.yscale('log')
    #     plt.show()
    #
    #     plt.plot(err.T[1:]);
    #     plt.yscale('log')
    #     plt.show()
    #     err_rand = 1
    #
    #     if np.max(err)<10**(-4) and np.max(err_rand)<10**(-4):
    #         print("Passed check, max and mean values:", (np.max(err), np.mean(err)),(np.max(err_rand), np.mean(err_rand)))
    #     else:
    #         print("WARNING: ",  (np.max(err), np.mean(err)),(np.max(err_rand), np.mean(err_rand)))
    # return Corr_bare, Corr_rand, tlist, wlist, sts_rand


def Corr_t_map(t, **SpinOp_data):

    # Phase matrices from k and k + pi
    phase_k = qt.qdiags(np.exp(1j * SpinOp_data["ek"] * t),0)
    phase_kp = qt.qdiags(np.exp(-1j * SpinOp_data["ekp"] * t),0)
    # print(1)
    # SpinOp_{k,k+pi} sandwiched between phase matrices from k and k + pi
    # This product is in (eigenvector k) - (eigenvector kp) basis
    phasek_Ukd_SpinOp_Ukp_phasekp = phase_k * SpinOp_data["Ukd_SpinOp_Ukp"] * phase_kp

    # print(2)
    # Skkpaux_k is in the (eigenvector k)-(bare kp) basi
    Uk = SpinOp_data["Uk"].full()
    phasek_Ukd_SpinOp_Ukp_phasekp_Ukpd_SpinOp_Uk_Ukd = (phasek_Ukd_SpinOp_Ukp_phasekp * SpinOp_data["Ukpd_SpinOp_Uk_Ukd"]).full()
    # Calculate t1.T * t2, which is in the bare basis
    Corr_k = np.sum(Uk.T * phasek_Ukd_SpinOp_Ukp_phasekp_Ukpd_SpinOp_Uk_Ukd, axis=0)
    Ukp = SpinOp_data["Ukp"].full()
    phasekp_Ukpd_SpinOp_Uk_phasek_Ukd_SpinOp_Ukp_Ukpd = (phasek_Ukd_SpinOp_Ukp_phasekp.dag() * SpinOp_data["Ukd_SpinOp_Ukp_Ukpd"]).full()
    Corr_kp = np.sum(Ukp.T * phasekp_Ukpd_SpinOp_Uk_phasek_Ukd_SpinOp_Ukp_Ukpd, axis=0)
    print(3)
    return [Corr_k, Corr_kp]

