import scqubits as scq
import qutip as qt
import numpy as np





sol = {}

H = qop.pxp_hamiltonian(bs ,bs_ind ,12)
SZ = qop.sz_neel(size = 12, basis = bs)

wscar = 2* 0.636
wmax = 3 * wscar
dw = wscar / 20
tmax = (2 * wmax / (2 * wmax + dw)) * (2 * np.pi / (dw))
dt = 2 * np.pi / (2 * wmax + dw)
t_list = np.linspace(0, tmax, int(tmax / dt))
print(dt)

for n in tqdm(range(len(bs[nx]))):
    ProdSt = qt.basis(len(bs[nx]), n)
    sol[nx, n] = qt.krylovsolve(HQ, ProdSt, t_list, krylov_dim=20, e_ops=[SZ])
    qt.qsave(sol, 'sol_0')
    np.savetxt('Monitor_0.txt', [100 * n / len(bs[nx])], fmt='%1.2f')