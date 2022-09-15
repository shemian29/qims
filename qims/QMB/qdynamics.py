import qutip as qt
import numpy as np
import qims as qims
from tqdm.notebook import tqdm

def SytSy(Sy, bs, Nx, Ukevecs, evals):
    Corr3 = {}
    Corrrand = {}
    wscar = 1
    wmax = 5 * wscar
    dw = wscar / 20
    tmax = (2 * wmax / (2 * wmax + dw)) * (2 * np.pi / (dw))
    dt = 2 * np.pi / (2 * wmax + dw)
    tlist = np.linspace(0, tmax, int(tmax / dt) + 1)
    k_list = np.arange(0, Nx) / Nx
    wlist = np.linspace(-wmax, wmax, len(tlist))

    itmax = 500
    tmp = [qt.rand_ket(len(bs)) for n in range(itmax)]
    sts_rand = qims.npqt2qtqt(tmp)

    for k in tqdm(k_list[0:int(Nx / 2)]):
        kit = ((k * Nx + int(Nx / 2)) % Nx) / Nx

        aux_k = qims.npqt2qtqt(Ukevecs[k])
        aux_kp = qims.npqt2qtqt(Ukevecs[kit])
        Skkp = aux_k.dag() * Sy * aux_kp
        Skkpaux_k = Skkp.dag() * aux_k.dag()
        Skkpaux_kp = Skkp * aux_kp.dag()

for n in tqdm(range(len(bs[nx]))):
    ProdSt = qt.basis(len(bs[nx]), n)
    sol[nx, n] = qt.krylovsolve(HQ, ProdSt, t_list, krylov_dim=20, e_ops=[SZ])
    qt.qsave(sol, 'sol_0')
    np.savetxt('Monitor_0.txt', [100 * n / len(bs[nx])], fmt='%1.2f')