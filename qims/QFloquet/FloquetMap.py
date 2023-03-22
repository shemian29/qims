import qutip as qt
from scipy.sparse import diags
from scipy.optimize import minimize
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

class DirectFloquetMap:

    def __init__(self, mcut = 10, wd = 1.0, Ez = 0.5):

        self.optionsODE = qt.Options(nsteps=10000000,
                                     store_states=True,
                                     rtol=1e-12,
                                     atol=1e-12,
                                     max_step=0.001)
        self.Nt_steps = 200


        self.m_list = np.arange(-mcut, mcut + 1)
        self.wd = wd
        self.T = 2 * np.pi / self.wd
        self.Ez = Ez
        self.tlist = np.linspace(0, self.T, len(self.m_list))

    def Optimal_gs(self):

        eq_cons = {'type': 'eq',
                   'fun': lambda g_full: [np.linalg.norm(
                       [np.abs(self.g_t(g_full, 'x', t) ** 2 + self.g_t(g_full, 'y', t) ** 2 + self.g_t(g_full, 'z',
                                                                                                        t) ** 2 - 1) for
                        t in
                        self.tlist])]}

        eq_cons2 = {'type': 'eq',
                    'fun': lambda g_full: [np.abs((1 / self.T) * np.diff(self.tlist)[0] * np.sum([(g_full[-1] - np.real(
                        self.dδexp2jβdt(g_full, t, 0.000001) / (2 * 1j * self.δexp2jβ(g_full, t)))) * self.g_t(g_full,
                                                                                                               'z', t)
                                                                                                  for t in
                                                                                                  self.tlist]) - self.Ez)]}

        lng = int((len(self.m_list) + 1) / 2)
        g0 = np.zeros(6 * lng - 3 + 1) + .00001
        g0[0] = 1
        g0[-1] = self.Ez
        self.res = minimize(self.rate, g0, method='SLSQP',
                       constraints=[eq_cons, eq_cons2],
                       options={'disp': True, "maxiter": 20})
        print('Constraint evaluations: ', eq_cons['fun'](self.res.x), eq_cons2['fun'](self.res.x))



    def δexp2jβ(self, gfull, t):
        f1 = ((self.g_t(gfull, 'x', t) + 1j * self.g_t(gfull, 'y', t)) ** 2) / (1 - self.g_t(gfull, 'z', t) ** 2)
        return f1

    def dδexp2jβdt(self, gfull, t, dt):

        f1 = ((self.g_t(gfull, 'x', t) + 1j * self.g_t(gfull, 'y', t)) ** 2) / (1 - self.g_t(gfull, 'z', t) ** 2)
        f2 = ((self.g_t(gfull, 'x', t + dt) + 1j * self.g_t(gfull, 'y', t + dt)) ** 2) / (
                    1 - self.g_t(gfull, 'z', t + dt) ** 2)

        return (f2 - f1) / dt

    def g_mat(self, g_coeffs):

        gs = diags(g_coeffs, self.m_list,
                   shape=(int((len(self.m_list) + 1) / 2), int((len(self.m_list) + 1) / 2))).toarray()

        return gs

    def g_t(self, gf, choice, t):

        return np.real(np.sum(self.g_s(gf, choice, 'c') * np.exp([1j * m * self.wd * t for m in self.m_list])))

    def spectral_density(self, m):
        return (np.abs(m) - 3) ** 2

    def g_s(self, g_full, channel, choice):
        lng = int((len(self.m_list) + 1) / 2)

        if channel == 'x' and choice == 'r':
            g_pick = g_full[0: lng]

        elif channel == 'x' and choice == 'i':
            g_pick = g_full[lng:2 * lng - 1]

        elif channel == 'x' and choice == 'c':
            g_pick = g_full[0:lng] + 1j * np.concatenate(([0], g_full[lng:2 * lng - 1]))

        # ------------------------------------------------------------------------------

        elif channel == 'y' and choice == 'r':
            g_pick = g_full[2 * lng - 1:3 * lng - 1]

        elif channel == 'y' and choice == 'i':
            g_pick = g_full[3 * lng - 1:4 * lng - 2]

        elif channel == 'y' and choice == 'c':
            g_pick = g_full[2 * lng - 1:3 * lng - 1] + 1j * np.concatenate(([0], g_full[3 * lng - 1:4 * lng - 2]))

        # ------------------------------------------------------------------------------

        elif channel == 'z' and choice == 'r':
            g_pick = g_full[4 * lng - 2:5 * lng - 2]

        elif channel == 'z' and choice == 'i':
            g_pick = g_full[5 * lng - 2:6 * lng - 3]

        elif channel == 'z' and choice == 'c':
            g_pick = g_full[4 * lng - 2:5 * lng - 2] + 1j * np.concatenate(([0], g_full[5 * lng - 2:6 * lng - 3]))

        return self.g_complete(g_pick)

    def g_complete(self, g_a_c):
        return np.concatenate((np.conjugate(np.flip(g_a_c[1:])), g_a_c))

    def rate(self, g_full):
        g_x_c = self.g_s(g_full, 'x', 'c')
        g_y_c = self.g_s(g_full, 'y', 'c')
        g_z_c = self.g_s(g_full, 'z', 'c')
        ϵ01 = g_full[-1]

        return np.dot((np.abs(g_x_c + 1j * g_y_c) ** 2), self.spectral_density(self.m_list * self.wd - ϵ01)) \
               + np.dot((np.abs(g_z_c) ** 2), self.spectral_density(self.m_list * self.wd))


    def φ(self,t):
        return self.wd * t


    def dφdt(self,t, dt):
        f1 = self.φ(t)
        f2 = self.φ(t + dt)
        return (f2 - f1) / dt


    def cosφ(self,t):
        return np.cos(self.φ(t))


    def sinφ(self,t):
        return np.sin(self.φ(t))


    def A1(self,g_full, t, dt):
        return (g_full[-1] - np.real(self.dδexp2jβdt(g_full, t, dt) / (2 * 1j * self.δexp2jβ(g_full, t))))


    def cosθ(self,g_full, t):
        return self.g_t(g_full, 'z', t)


    def sinθ(self,g_full, t):
        return np.sqrt(1 - self.cosθ(g_full, t) ** 2)


    def dgzzdt(self,g_full, t, dt):
        f1 = self.g_t(g_full, 'z', t)
        f2 = self.g_t(g_full, 'z', t + dt)

        return (f2 - f1) / dt


    def dθdt(self,t, dt):
        return -self.dgzzdt(self.res.x, t, dt) / self.sinθ(self.res.x, t)


    def Vx(self, t, dt):
        return 0.5 * (self.A1(self.res.x, t, dt) * self.cosφ(t) * self.sinθ(self.res.x, t) + self.dθdt(self.res.x, t, dt) * self.sinφ(t))


    def Vy(self, t, dt):
        return 0.5 * (self.A1(self.res.x, t, dt) * self.sinφ(t) * self.sinθ(self.res.x, t) + self.dθdt(self.res.x, t, dt) * self.cosφ(t))


    def Vz(self, t, dt):
        return 0.5 * (self.A1(self.res.x, t, dt) * self.cosθ(self.res.x, t) - self.dφdt(t, dt) - self.Ez)


    def Plot_gs(self):

        figure(figsize=(10, 6), dpi=80)

        plt.plot(self.m_list * self.wd, np.abs(self.g_s(self.res.x, 'x', 'c') - 1j * self.g_s(self.res.x, 'y', 'c')), '.-', color='pink',
                 label=r'$g_{-}$')
        plt.plot(self.m_list * self.wd, np.abs(self.g_s(self.res.x, 'x', 'c') + 1j * self.g_s(self.res.x, 'y', 'c')), '.-', color='orange',
                 label=r'$g_{+}$')
        plt.plot(self.m_list * self.wd, np.abs(self.g_s(self.res.x, 'z', 'c')), 'g.-', label=r'$g_{z}$')

        plt.plot(self.wd * self.m_list, 0.1 * self.spectral_density(self.wd * self.m_list), '.-', color='purple', label='S(m)')

        plt.ylim([-0.1, 1.5])
        plt.legend(fontsize=15)
        plt.xlabel('m', size=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid()