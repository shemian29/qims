import qutip as qt
from scipy.sparse import diags
from scipy.optimize import minimize, differential_evolution
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid

δf = 1.8 * (10 ** (-6))
EL = 1.3  # GHz
φge = 1.996
Af = 2 * np.pi * δf * EL * np.abs(φge)


class DirectFloquetMap:
    def __init__(self, mcut: int = 10, wd: float = 1.0, Ez: float = 0.5) -> object:
        """

        :type mcut: object
        """
        self.optionsODE = qt.Options(
            nsteps=10000000, store_states=True, rtol=1e-12, atol=1e-12, max_step=0.001
        )
        self.Nt_steps = 200
        self.dt = 0.0001

        self.m_list = np.arange(-mcut, mcut + 1)
        self.mcut = mcut
        self.wd = wd
        self.T = 2 * np.pi / self.wd
        self.Ez = Ez
        self.tlist = np.linspace(0, self.T, 2 * len(self.m_list))
        self.optimized = "Not yet"

    def _repr_latex_(self):

        # string = r'$\\ \begin{pmatrix} e^\cos \theta(t)  \\a   \end{pmatrix} $'
        string = "Floquet qubit of the form: "
        string = (
                string
                + r"$\vert v_{\pm}(t)\rangle = e^{i \beta_{\pm}(t)}\left( \begin{array}{c} \pm e^{-i \varphi(t)/2} \cos\left(\frac{\theta(t)}{2}\right)  \\ e^{i \varphi(t)/2}\sin\left(\frac{\theta(t)}{2}\right)  \end{array} \right).   $   "
        )
        string = string + "   Optimized: " + self.optimized
        return string

    def MaxAmp(self, g_coeff, t):
        return np.max(
            np.abs([self.A1(g_coeff, t) ** 2, self.dθdt_squared(g_coeff, t)])
        )

    def Optimal_gs_DE(self, g0):

        bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]

        result = differential_evolution(self.rate, bounds)

        return result

    def Optimal_gs(self, g0, maxiter=100, jac=True):

        eq_cons = {
            "type": "eq",
            "fun": lambda g_full: [
                np.linalg.norm(
                    [
                        np.abs(
                            self.g_t(g_full, "x", t) ** 2
                            + self.g_t(g_full, "y", t) ** 2
                            + self.g_t(g_full, "z", t) ** 2
                            - 1
                        )
                        for t in self.tlist
                    ]
                )
            ],
        }

        eq_cons2 = {
            "type": "eq",
            "fun": lambda g_full: [
                np.abs(
                    (1 / self.T)
                    * np.diff(self.tlist)[0]
                    * np.sum(
                        [
                            (
                                    g_full[-1]
                                    - np.real(
                                self.dδexp2jβdt(g_full, t)
                                / (2 * 1j * self.δexp2jβ(g_full, t))
                            )
                            )
                            * self.g_t(g_full, "z", t)
                            for t in self.tlist
                        ]
                    )
                    - self.Ez
                )
            ],
        }

        ineq_cons = {
            "type": "ineq",
            "fun": lambda g_full: [
                0.1 - np.abs(np.max([self.MaxAmp(g_full, t) for t in self.tlist]))
            ],
        }

        if jac == True:
            self.res = minimize(
                self.rate,
                g0,
                method="SLSQP",
                jac=self.rate_grad,
                constraints=[eq_cons, eq_cons2, ineq_cons],
                options={"disp": True, "maxiter": maxiter},
            )



        elif jac == False:
            self.res = minimize(
                self.rate,
                g0,
                method="SLSQP",
                constraints=[eq_cons, eq_cons2, ineq_cons],
                options={"disp": True, "maxiter": maxiter},
            )
        self.ϵ01 = self.res.x[-1]
        print(
            "Constraint evaluations: ",
            eq_cons["fun"](self.res.x),
            eq_cons2["fun"](self.res.x),
            ineq_cons["fun"](self.res.x)
        )
        self.optimized = "Yes"

    def minimize_gradual(self, step=10):
        lng = int((len(self.m_list) + 1) / 2)
        g0 = np.zeros(6 * lng - 3 + 1) + 0.00001
        g0[0] = 1
        g0[-1] = self.Ez

        maxiter = 10  # np.floor(1.5*self.omega_0/self.wd)#10
        tlist = np.linspace(0., self.T - 0., 101)
        message = 'Iteration limit reached'
        scan = []
        while message == 'Iteration limit reached':
            maxiter = maxiter + step
            self.Optimal_gs(g0=g0, maxiter=maxiter, jac=False)
            message = self.res.message
            g0 = self.res.x
            scan.append([maxiter, self.res.fun])
            print(scan)
            self.PlotData()
            plt.show()
            # self.Plot_gt(tlist)
            # plt.show()
            # self.PlotDrives(tlist)
            # plt.show()

    def δexp2jβ(self, gfull, t):
        f1 = ((self.g_t(gfull, "x", t) + 1j * self.g_t(gfull, "y", t)) ** 2) / (
                1 - self.g_t(gfull, "z", t) ** 2
        )
        return f1

    def dδexp2jβdt(self, gfull, t):

        f1 = ((self.g_t(gfull, "x", t) + 1j * self.g_t(gfull, "y", t)) ** 2) / (
                1 - self.g_t(gfull, "z", t) ** 2
        )
        f2 = (
                     (self.g_t(gfull, "x", t + self.dt) + 1j * self.g_t(gfull, "y", t + self.dt))
                     ** 2
             ) / (1 - self.g_t(gfull, "z", t + self.dt) ** 2)

        return (f2 - f1) / self.dt

    def g_mat(self, g_coeffs):

        gs = diags(
            g_coeffs,
            self.m_list,
            shape=(int((len(self.m_list) + 1) / 2), int((len(self.m_list) + 1) / 2)),
        ).toarray()

        return gs

    def g_t(self, gf, choice, t):

        return np.real(
            np.sum(
                self.g_s(gf, choice, "c")
                * np.exp([1j * m * self.wd * t for m in self.m_list])
            )
        )

    def Sf(self, ω):
        """Spectral function. Frequency in GHz."""
        δf = 1.8 * (10 ** (-6))
        EL = 1.3  # GHz
        φge = 1.996
        Af = 2 * np.pi * δf * EL * np.abs(φge)
        return (Af ** 2) * np.abs((2 * np.pi) / ω) * (10 ** 6) * (2 * np.pi)  # kHz

    def Sd(self, ω):

        EC = 0.5  # GHz
        Temp = 15 * (10 ** (-3))  # Kelvin
        ħ = 1.05457182 * (10 ** (-34))  # Js
        kB = 1.380649 * (10 ** (-23))  # J/K
        λ = (2 * np.pi * ħ / kB) * (10 ** 9)  # K/GHz
        α = np.abs(np.cosh(λ * ω / (2 * Temp)) / np.sinh(λ * ω / (2 * Temp)) + 1) / 2
        φge = 1.996
        tanδc = 1.1 * (10 ** (-6))
        Ad = (np.pi ** 2) * tanδc * (φge ** 2) / EC
        return α * Ad * ((ω / (2 * np.pi)) ** 2) * (10 ** 6) * (2 * np.pi)  # kHz

    def grad_Sf(self, ω):
        """Spectral function. Frequency in GHz."""
        δf = 1.8 * (10 ** (-6))
        EL = 1.3  # GHz
        φge = 1.996
        Af = 2 * np.pi * δf * EL * np.abs(φge)
        return (-1 / ω) * self.Sf(ω)  # kHz/GHz

    def grad_Sd(self, ω):

        EC = 0.5  # GHz
        Temp = 15 * (10 ** (-3))  # Kelvin
        ħ = 1.05457182 * (10 ** (-34))  # Js
        kB = 1.380649 * (10 ** (-23))  # J/K
        λ = (2 * np.pi * ħ / kB) * (10 ** 9)  # K/GHz
        α = np.abs(np.cosh(λ * ω / (2 * Temp)) / np.sinh(λ * ω / (2 * Temp)) + 1) / 2
        φge = 1.996
        tanδc = 1.1 * (10 ** (-6))
        Ad = (np.pi ** 2) * tanδc * (φge ** 2) / EC
        return self.Sd(ω) * (
                2 * kB * Temp + ħ * ω * (1 - np.cosh(λ * ω / (2 * Temp)) / np.sinh(λ * ω / (2 * Temp)))) / (
                kB * Temp * ω)  # kHz/GHz

    def spectral_density(self, w):
        # return 1.0 * (w**2 - (1) ** 2) ** 2 + w
        return (self.Sf(w) + self.Sd(w))

    def grad_spectral_density(self, w):
        # return (4.0) * w * (w**2 - (1) ** 2) + 1
        return self.grad_Sd(w) + self.grad_Sf(w)

    def g_s(self, g_full, channel, choice):
        lng = int((len(self.m_list) + 1) / 2)

        if channel == "x" and choice == "r":
            g_pick = g_full[0:lng]

        elif channel == "x" and choice == "i":
            g_pick = g_full[lng: 2 * lng - 1]

        elif channel == "x" and choice == "c":
            g_pick = g_full[0:lng] + 1j * np.concatenate(
                ([0], g_full[lng: 2 * lng - 1])
            )

        # ------------------------------------------------------------------------------

        elif channel == "y" and choice == "r":
            g_pick = g_full[2 * lng - 1: 3 * lng - 1]

        elif channel == "y" and choice == "i":
            g_pick = g_full[3 * lng - 1: 4 * lng - 2]

        elif channel == "y" and choice == "c":
            g_pick = g_full[2 * lng - 1: 3 * lng - 1] + 1j * np.concatenate(
                ([0], g_full[3 * lng - 1: 4 * lng - 2])
            )

        # ------------------------------------------------------------------------------

        elif channel == "z" and choice == "r":
            g_pick = g_full[4 * lng - 2: 5 * lng - 2]

        elif channel == "z" and choice == "i":
            g_pick = g_full[5 * lng - 2: 6 * lng - 3]

        elif channel == "z" and choice == "c":
            g_pick = g_full[4 * lng - 2: 5 * lng - 2] + 1j * np.concatenate(
                ([0], g_full[5 * lng - 2: 6 * lng - 3])
            )

        return self.g_complete(g_pick)

    def rate_grad(self, g_full):
        mcut = int((len(self.m_list) - 1) / 2)
        lng = int((len(self.m_list) + 1) / 2)
        ϵ01 = g_full[-1]
        g_pick = [2 * g_full[0] * self.spectral_density(-ϵ01)]

        g_pick = (
                g_pick
                + (
                        2
                        * (g_full[1: mcut + 1] + g_full[3 * mcut + 2: 4 * mcut + 2])
                        * self.spectral_density(
                    -self.m_list[mcut + 1: 2 * mcut + 1] * self.wd - ϵ01
                )
                        + 2
                        * (g_full[1: mcut + 1] - g_full[3 * mcut + 2: 4 * mcut + 2])
                        * self.spectral_density(
                    self.m_list[mcut + 1: 2 * mcut + 1] * self.wd - ϵ01
                )
                ).tolist()
        )

        g_pick = (
                g_pick
                + (
                        2
                        * (
                                g_full[mcut + 1: 2 * mcut + 1]
                                + g_full[2 * mcut + 2: 3 * mcut + 2]
                        )
                        * self.spectral_density(
                    self.m_list[mcut + 1: 2 * mcut + 1] * self.wd - ϵ01
                )
                        + 2
                        * (
                                g_full[mcut + 1: 2 * mcut + 1]
                                - g_full[2 * mcut + 2: 3 * mcut + 2]
                        )
                        * self.spectral_density(
                    -self.m_list[mcut + 1: 2 * mcut + 1] * self.wd - ϵ01
                )
                ).tolist()
        )

        # ------------------------------------------------------------------------------

        g_pick = g_pick + [2 * g_full[2 * mcut + 1] * self.spectral_density(-ϵ01)]

        g_pick = (
                g_pick
                + (
                        2
                        * (
                                g_full[mcut + 1: 2 * mcut + 1]
                                + g_full[2 * mcut + 2: 3 * mcut + 2]
                        )
                        * self.spectral_density(
                    self.m_list[mcut + 1: 2 * mcut + 1] * self.wd - ϵ01
                )
                        + 2
                        * (
                                -g_full[mcut + 1: 2 * mcut + 1]
                                + g_full[2 * mcut + 2: 3 * mcut + 2]
                        )
                        * self.spectral_density(
                    -self.m_list[mcut + 1: 2 * mcut + 1] * self.wd - ϵ01
                )
                ).tolist()
        )

        g_pick = (
                g_pick
                + (
                        2
                        * (-g_full[1: mcut + 1] + g_full[3 * mcut + 2: 4 * mcut + 2])
                        * self.spectral_density(
                    self.m_list[mcut + 1: 2 * mcut + 1] * self.wd - ϵ01
                )
                        + 2
                        * (g_full[1: mcut + 1] + g_full[3 * mcut + 2: 4 * mcut + 2])
                        * self.spectral_density(
                    -self.m_list[mcut + 1: 2 * mcut + 1] * self.wd - ϵ01
                )
                ).tolist()
        )

        # ------------------------------------------------------------------------------

        g_pick = g_pick + [
            2
            * g_full[4 * lng - 2]
            * self.spectral_density(self.m_list[lng - 1] * self.wd)
        ]

        g_pick = (
                g_pick
                + (
                        4
                        * g_full[4 * lng - 1: 5 * lng - 2]
                        * self.spectral_density(self.m_list[lng:] * self.wd)
                ).tolist()
        )

        g_pick = (
                g_pick
                + (
                        4
                        * g_full[5 * lng - 2: 6 * lng - 3]
                        * self.spectral_density(self.m_list[lng:] * self.wd)
                ).tolist()
        )

        g_x_c = self.g_s(g_full, "x", "c")
        g_y_c = self.g_s(g_full, "y", "c")

        g_pick = g_pick + [
            np.dot(
                (np.abs(g_x_c + 1j * g_y_c) ** 2),
                -self.grad_spectral_density(self.m_list * self.wd - ϵ01),
            )
        ]

        return np.array(g_pick)

    def g_complete(self, g_a_c):
        return np.concatenate((np.conjugate(np.flip(g_a_c[1:])), g_a_c))

    def rate_DF(self, g_full):
        g_x_c = self.g_s(g_full, "x", "c")
        g_y_c = self.g_s(g_full, "y", "c")
        g_z_c = self.g_s(g_full, "z", "c")
        ϵ01 = g_full[-1]

        gamma_0 = Af * 2 * (np.abs(g_z_c[self.mcut]) ) * 4 * (10 ** 6)
        g_z_c = np.delete(g_z_c, self.mcut)
        mlist = np.delete(self.m_list, self.mcut)


        norm_constr = np.linalg.norm(
                    [
                        np.abs(
                            self.g_t(g_full, "x", t) ** 2
                            + self.g_t(g_full, "y", t) ** 2
                            + self.g_t(g_full, "z", t) ** 2
                            - 1
                        )
                        for t in self.tlist
                    ]
                )

        # rate = 0.5 * np.dot(
        #     (np.abs(g_x_c + 1j * g_y_c) ** 2),
        #     self.spectral_density(self.m_list * self.wd - ϵ01),
        # ) + gamma_0 + np.dot((np.abs(g_z_c) ** 2), self.spectral_density(mlist * self.wd))

        dephasing = gamma_0 + 2*np.dot((np.abs(g_z_c) ** 2), self.spectral_density(mlist * self.wd))

        depolarization = np.dot(
            (np.abs(g_x_c - 1j * g_y_c) ** 2),
            self.spectral_density(self.m_list * self.wd + ϵ01),
        )

        excitation = np.dot(
            (np.abs(g_x_c + 1j * g_y_c) ** 2),
            self.spectral_density(self.m_list * self.wd - ϵ01),
        )

        rate = 0.5*(depolarization+excitation)+dephasing

        return rate + norm_constr/(rate)**2

    def rate_DF_vals(self, g_full):
        g_x_c = self.g_s(g_full, "x", "c")
        g_y_c = self.g_s(g_full, "y", "c")
        g_z_c = self.g_s(g_full, "z", "c")
        ϵ01 = g_full[-1]

        gamma_0 = Af * 2 * (np.abs(g_z_c[self.mcut]) ) * 4 * (10 ** 6)
        g_z_c = np.delete(g_z_c, self.mcut)
        mlist = np.delete(self.m_list, self.mcut)


        norm_constr = np.linalg.norm(
                    [
                        np.abs(
                            self.g_t(g_full, "x", t) ** 2
                            + self.g_t(g_full, "y", t) ** 2
                            + self.g_t(g_full, "z", t) ** 2
                            - 1
                        )
                        for t in self.tlist
                    ]
                )

        dephasing = gamma_0 + 2*np.dot((np.abs(g_z_c) ** 2), self.spectral_density(mlist * self.wd))

        depolarization = np.dot(
            (np.abs(g_x_c - 1j * g_y_c) ** 2),
            self.spectral_density(self.m_list * self.wd + ϵ01),
        )

        excitation = np.dot(
            (np.abs(g_x_c + 1j * g_y_c) ** 2),
            self.spectral_density(self.m_list * self.wd - ϵ01),
        )

        rate = 0.5*(depolarization+excitation)+dephasing

        return [dephasing, depolarization,excitation,rate,norm_constr]

    def rate(self, g_full):
        g_x_c = self.g_s(g_full, "x", "c")
        g_y_c = self.g_s(g_full, "y", "c")
        g_z_c = self.g_s(g_full, "z", "c")
        ϵ01 = g_full[-1]

        gamma_0 = Af * 2 * (np.abs(g_z_c[self.mcut]) ** 2) * 4 * (10 ** 6)
        g_z_c = np.delete(g_z_c, self.mcut)
        mlist = np.delete(self.m_list, self.mcut)
        return 0.5 * np.dot(
            (np.abs(g_x_c + 1j * g_y_c) ** 2),
            self.spectral_density(self.m_list * self.wd - ϵ01),
        ) + gamma_0 + np.dot((np.abs(g_z_c) ** 2), self.spectral_density(mlist * self.wd))

    def φ(self, t):
        return self.wd * t

    def dφdt(self, t):
        f1 = self.φ(t)
        f2 = self.φ(t + self.dt)
        return (f2 - f1) / self.dt

    def cosφ(self, t):
        return np.cos(self.φ(t))

    def sinφ(self, t):
        return np.sin(self.φ(t))

    def A1(self, g_full, t):
        return g_full[-1] - np.real(
            self.dδexp2jβdt(g_full, t) / (2 * 1j * self.δexp2jβ(g_full, t))
        )

    def cosθ(self, g_full, t):
        return self.g_t(g_full, "z", t)

    def sinθ(self, g_full, t):
        return np.sqrt(1 - self.cosθ(g_full, t) ** 2)

    def dgzzdt(self, g_full, t):
        f1 = self.g_t(g_full, "z", t)
        f2 = self.g_t(g_full, "z", t + self.dt)

        return (f2 - f1) / self.dt

    def dθdt(self, g_full, t):
        return -self.dgzzdt(g_full, t) / self.sinθ(g_full, t)

    def dθdt_squared(self, g_full, t):
        return (self.dgzzdt(g_full, t) ** 2) / (1 - self.cosθ(g_full, t) ** 2)

    def Vx(self, t):
        return 0.5 * (
                self.A1(self.res.x, t) * self.cosφ(t) * self.sinθ(self.res.x, t)
                + self.dθdt(self.res.x, t) * self.sinφ(t)
        )

    def Vy(self, t):
        return 0.5 * (
                self.A1(self.res.x, t) * self.sinφ(t) * self.sinθ(self.res.x, t)
                + self.dθdt(self.res.x, t) * self.cosφ(t)
        )

    def Vz(self, t):
        return 0.5 * (
                self.A1(self.res.x, t) * self.cosθ(self.res.x, t) - self.dφdt(t) - self.Ez
        )

    def Plot_gs(self):

        figure(figsize=(10, 6), dpi=80)

        plt.plot(
            self.m_list * self.wd,
            np.abs(
                self.g_s(self.res.x, "x", "c") - 1j * self.g_s(self.res.x, "y", "c")
            ),
            ".-",
            color="pink",
            label=r"$g_{-}$",
        )
        plt.plot(
            self.m_list * self.wd,
            np.abs(
                self.g_s(self.res.x, "x", "c") + 1j * self.g_s(self.res.x, "y", "c")
            ),
            ".-",
            color="orange",
            label=r"$g_{+}$",
        )
        plt.plot(
            self.m_list * self.wd,
            np.abs(self.g_s(self.res.x, "z", "c")),
            "g.-",
            label=r"$g_{z}$",
        )

        plt.plot(
            self.wd * self.m_list,
            0.01 * self.spectral_density(self.wd * self.m_list),
            ".-",
            color="purple",
            label="S(m)",
        )

        plt.ylim([-0.1, 1.5])
        plt.legend(fontsize=15)
        plt.xlabel("m", size=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid()

    def Hamiltonian(self):

        sx = qt.sigmax()
        sy = qt.sigmay()
        sz = qt.sigmaz()

        H_static = 0.5 * self.Ez * sz

        H_dynamic = list([H_static])

        H_dynamic.append([sx, lambda t, args: self.Vx(t)])

        H_dynamic.append([sy, lambda t, args: self.Vy(t)])

        H_dynamic.append([sz / 2, lambda t, args: self.Vz(t)])

        return H_dynamic

    def BlochSpherePath(self, file_name="anim"):

        tlist = np.linspace(0.001, 4 * self.T, 4 * 301)
        sx = [self.cosφ(t) * self.sinθ(self.res.x, t) for t in tlist]
        sy = [self.sinφ(t) * self.sinθ(self.res.x, t) for t in tlist]
        sz = [self.cosθ(self.res.x, t) for t in tlist]

        fig = plt.figure(figsize=[10.4, 8.8], dpi=100.0)
        ax = Axes3D(fig, azim=-40, elev=30, auto_add_to_figure=False)
        fig.add_axes(ax)
        sphere = qt.Bloch(axes=ax)

        def animate(i):
            sphere.clear()
            sphere.add_points([sx[: i + 1], sy[: i + 1], sz[: i + 1]])
            sphere.make_sphere()
            return ax

        def init():
            sphere.vector_color = ["r"]
            return ax

        ani = animation.FuncAnimation(
            fig, animate, np.arange(len(sx)), init_func=init, blit=False, repeat=False
        )
        ani.save(file_name + ".mp4", fps=20)

    def PlotDrives(self, tlist_int):

        figure(figsize=(10, 6), dpi=80)

        plt.plot(
            tlist_int / self.T,
            [self.Vx(t) for t in tlist_int],
            ".-",
            color="forestgreen",
            label=r"$V_x(t)$",
        )
        plt.plot(
            tlist_int / self.T,
            [self.Vy(t) for t in tlist_int],
            ".-",
            color="gold",
            label=r"$V_y(t)$",
        )
        plt.plot(
            tlist_int / self.T,
            [self.Vz(t) for t in tlist_int],
            ".-",
            color="royalblue",
            label=r"$V_z(t)$",
        )

        plt.grid()
        plt.xlabel("time/T", size=20)
        plt.ylabel("drive", size=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=15)

    def Plot_gt(self, tlist):

        # tlist = np.linspace(0, self.T, 200)

        figure(figsize=(10, 6), dpi=80)

        plt.plot(
            tlist / self.T,
            [self.g_t(self.res.x, "x", t) for t in tlist],
            color="red",
            label=r"$g_x(t)$",
        )
        plt.plot(
            tlist / self.T,
            [self.g_t(self.res.x, "y", t) for t in tlist],
            color="pink",
            label=r"$g_y(t)$",
        )
        plt.plot(
            tlist / self.T,
            [self.g_t(self.res.x, "z", t) for t in tlist],
            color="lightblue",
            label=r"$g_z(t)$",
        )
        plt.plot(
            tlist / self.T,
            [
                self.g_t(self.res.x, "x", t) ** 2
                + self.g_t(self.res.x, "y", t) ** 2
                + self.g_t(self.res.x, "z", t) ** 2
                for t in tlist
            ],
            color="blue",
            label=r"sum",
        )

        plt.grid()
        plt.xlabel("time/T", size=20)
        plt.ylabel("g coefficients", size=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=15)

    def PlotSpectralDensity(self):

        figure(figsize=(10, 10), dpi=80)
        wlist = np.linspace(-1, 1, 800)

        plt.plot(wlist, [(self.spectral_density(ω)) for ω in wlist], color='red')
        plt.grid()
        plt.yscale('log')
        plt.ylim([0.5, 30])
        plt.xlim([-0.83, 0.83])

    # def GeneralStateDynamics(t, Psi0):
    #     f_modes_t = qt.floquet_modes_t_lookup(fmodes_table, t, T)
    #     f_coeff = qt.floquet_state_decomposition(f_modes, f_energies, Psi0)
    #     psi_t = qt.floquet_wavefunction(f_modes_t, f_energies, f_coeff, t)
    #
    #     return psi_t

    # f_modes, f_energies = qt.floquet_modes(
    #     H=H_dynamic, T=T, sort=True, options=opts
    # )

    def PlotData(self):
        tlist2 = np.linspace(0., self.T - 0., 101)
        wlist = np.linspace(-1, 1, 800)

        fig = plt.figure(figsize=(15., 15.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(1, 3),  # creates 2x2 grid of axes
                         axes_pad=0.25,  # pad between axes in inch.
                         )
        grid[0].plot(
            self.m_list * self.wd,
            np.abs(
                self.g_s(self.res.x, "x", "c") - 1j * self.g_s(self.res.x, "y", "c")
            ),
            ".-",
            color="pink",
            label=r"$g_{-}$",
        )

        grid[0].plot(
            self.m_list * self.wd,
            np.abs(
                self.g_s(self.res.x, "x", "c") + 1j * self.g_s(self.res.x, "y", "c")
            ),
            ".-",
            color="orange",
            label=r"$g_{+}$",
        )
        grid[0].plot(
            self.m_list * self.wd,
            np.abs(self.g_s(self.res.x, "z", "c")),
            "g.-",
            label=r"$g_{z}$",
        )

        grid[0].grid(color="lightgray", linestyle="-", linewidth=0.35)

        grid[1].plot(
            tlist2 / self.T,
            [self.g_t(self.res.x, "x", t) for t in tlist2],
            color="red",
            label=r"$g_x(t)$",
        )
        grid[1].plot(
            tlist2 / self.T,
            [self.g_t(self.res.x, "y", t) for t in tlist2],
            color="pink",
            label=r"$g_y(t)$",
        )
        grid[1].plot(
            tlist2 / self.T,
            [self.g_t(self.res.x, "z", t) for t in tlist2],
            color="lightblue",
            label=r"$g_z(t)$",
        )

        grid[1].plot(
            tlist2 / self.T,
            [
                self.g_t(self.res.x, "x", t) ** 2
                + self.g_t(self.res.x, "y", t) ** 2
                + self.g_t(self.res.x, "z", t) ** 2
                for t in tlist2
            ],
            color="blue",
            label=r"sum",
        )
        # grid[1].set_aspect(0.4)
        grid[1].grid(color="lightgray", linestyle="-", linewidth=0.35)

        grid[0].plot(
            self.wd * self.m_list,
            self.spectral_density(self.wd * self.m_list + 0.0001),
            ".",
            color="purple",
            label="S(m)",
        )

        grid[0].plot(wlist, [(self.spectral_density(ω)) for ω in wlist], color='red')
        grid[0].grid(color="lightgray", linestyle="-", linewidth=0.35)

        grid[0].set_ylim([-1, 4])

        grid[2].plot(
            tlist2 / self.T,
            [self.Vx(t) for t in tlist2],
            ".-",
            color="forestgreen",
            label=r"$V_x(t)$",
        )
        grid[2].plot(
            tlist2 / self.T,
            [self.Vy(t) for t in tlist2],
            ".-",
            color="gold",
            label=r"$V_y(t)$",
        )
        grid[2].plot(
            tlist2 / self.T,
            [self.Vz(t) for t in tlist2],
            ".-",
            color="royalblue",
            label=r"$V_z(t)$",
        )
        grid[2].grid(color="lightgray", linestyle="-", linewidth=0.35)

        grid[0].legend()
        grid[1].legend()
        grid[2].legend()
        grid[0].set_aspect(0.2)
        grid[1].set_aspect(0.2)
        grid[2].set_aspect(0.2)

        grid[0].set_xlim([-0.75, 0.75])
