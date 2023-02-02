import scqubits as scq
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm


class FloquetQubit:

    def __init__(self, system="qubit", drive="pulse"):

        self.system = system
        self.drive = drive
        self.setup_prompt(system, drive)

        self.system_params = None
        self.drive_params = None

        self.optionsODE = qt.Options(nsteps=100000,
                                     store_states=True,
                                     rtol=1e-12,
                                     atol=1e-12,
                                     max_step=0.001)
        self.Nt_steps = 200

    def setup_prompt(self, system, drive):

        if system == "qubit":
            print("- Example system dictionary:\n")
            print("system_params = {'b1':1, 'b2':1, 'b3':1} # the b_i defines coefficients of Pauli matrices \n")

        if drive == "pulse":
            print("- Possible drive operators: sx, sy, sz. Example of drive dictionary:\n")
            print(
                "drive_params = {'pattern' : [1.0,'sx',2.0,'sy',1.0],'angles' : np.array([np.pi/2, np.pi/2]),'widths': np.array([0.1,0.1])} ")
            print("")

    def SetDrivenSystem(self):

        if self.drive == "pulse":
            all_spacings = [r for r in self.drive_params["pattern"] if isinstance(r, float)]
            self.T_drive = np.sum(self.drive_params["widths"]) + np.sum(all_spacings)

            self.tlist = np.linspace(0, self.T_drive, self.Nt_steps)

            self.freq = np.fft.fftfreq(self.tlist.shape[-1])

        if self.system == "qubit":
            self.operators = [["sx", qt.sigmax()], ["sy", qt.sigmay()], ["sz", qt.sigmaz()]]

            H_static = self.system_params["b1"] * qt.sigmax() \
                       + self.system_params["b2"] * qt.sigmay() \
                       + self.system_params["b3"] * qt.sigmaz()

            H_dynamic = list([H_static])
            H_dynamic.append([self.operators[0][1], lambda t, args: self.drive_profile(t, self.operators[0][0])])
            H_dynamic.append([self.operators[1][1], lambda t, args: self.drive_profile(t, self.operators[1][0])])
            H_dynamic.append([self.operators[2][1], lambda t, args: self.drive_profile(t, self.operators[2][0])])

            self.H_static = H_static
            self.H_dynamic = H_dynamic

    def drive_profile(self, t, operator):

        if self.drive == "pulse":

            Npulses = len(self.drive_params["widths"])

            aux_operators = [r for r in self.drive_params["pattern"] if isinstance(r, str)]  # extract operators
            aux_sep = [r for r in self.drive_params["pattern"][0:2 * Npulses] if
                       isinstance(r, float)]  # extract spacings between operators

            aux_timeline = np.array([aux_sep, self.drive_params["widths"]]).T.flatten()
            bins = np.cumsum(aux_timeline).reshape((Npulses, 2))

            aux_driveassign = {}
            for op in self.operators:
                aux_driveassign[op[0]] = []

            for it, b in enumerate(bins):
                aux_driveassign[aux_operators[it]].append(b)

            all_spacings = [r for r in self.drive_params["pattern"] if isinstance(r, float)]

            aux_locs = ((np.where(np.array(self.drive_params["pattern"]) == operator)[-1] - 1) / 2).astype(int)

            for it, interval in enumerate(aux_driveassign[operator]):
                if interval[0] < t % self.T_drive < interval[1]:
                    return (self.drive_params["angles"][aux_locs[it]] / self.drive_params["widths"][aux_locs[it]])
        return 0

    def PlotDrive(self):

        clrs = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for it, operator in enumerate(self.operators):
            plt.plot(self.tlist, [self.drive_profile(t, operator[0]) for t in self.tlist], clrs[it] + '-',
                     label=operator[0])

        plt.legend()
        plt.grid()
        plt.xlim([0, self.T_drive])

    def FloquetSpectrum(self):

        self.f_modes, self.f_energies = qt.floquet_modes(H=self.H_dynamic,
                                                         T=self.T_drive,
                                                         sort=True,
                                                         options=self.optionsODE)

        self.fmodes_table = qt.floquet_modes_table(self.f_modes,
                                                   self.f_energies,
                                                   self.tlist,
                                                   self.H_dynamic,
                                                   self.T_drive,
                                                   options=self.optionsODE)

        return self.f_modes, self.f_energies

    def GeneralStateDynamics(self, t, Psi0):
        f_modes_t = qt.floquet_modes_t_lookup(self.fmodes_table, t, self.T_drive)
        f_coeff = qt.floquet_state_decomposition(self.f_modes, self.f_energies, Psi0)
        psi_t = qt.floquet_wavefunction(f_modes_t, self.f_energies, f_coeff, t)

        return psi_t

    def BlochVector_StaticBasis(self, Psi0, monitor=True):

        sx_ave = []
        sy_ave = []
        sz_ave = []

        if monitor:
            for t in tqdm(self.tlist):
                psi_t = self.GeneralStateDynamics(t, Psi0)
                sxa = qt.expect(qt.sigmax(), psi_t)
                sya = qt.expect(qt.sigmay(), psi_t)
                sza = qt.expect(qt.sigmaz(), psi_t)
                sx_ave.append(sxa)
                sy_ave.append(sya)
                sz_ave.append(sza)
        else:
            for t in self.tlist:
                psi_t = self.GeneralStateDynamics(t, Psi0)
                sxa = qt.expect(qt.sigmax(), psi_t)
                sya = qt.expect(qt.sigmay(), psi_t)
                sza = qt.expect(qt.sigmaz(), psi_t)
                sx_ave.append(sxa)
                sy_ave.append(sya)
                sz_ave.append(sza)

        self.sx_ave = sx_ave
        self.sy_ave = sy_ave
        self.sz_ave = sz_ave

    def phase(self, n, t):
        return np.exp(1j * t * self.f_energies[n])

    def Operator_FloquetBasis(self, operator, monitor=True):

        taux_ave = []
        tauy_ave = []
        tauz_ave = []

        if monitor:
            for t in tqdm(self.tlist):
                psi_t_0 = self.GeneralStateDynamics(t, self.f_modes[0])
                psi_t_1 = self.GeneralStateDynamics(t, self.f_modes[1])

                tau_x = self.phase(0, t) * psi_t_0 * psi_t_1.dag() * self.phase(1, -t) \
                        + self.phase(1, t) * psi_t_1 * psi_t_0.dag() * self.phase(0, -t)

                tau_y = -1j * self.phase(1, t) * psi_t_1 * psi_t_0.dag() * self.phase(0, -t) \
                        + 1j * self.phase(0, t) * psi_t_0 * psi_t_1.dag() * self.phase(1, -t)

                tau_z = self.phase(1, t) * psi_t_1 * psi_t_1.dag() * self.phase(1, -t) \
                        - self.phase(0, t) * psi_t_0 * psi_t_0.dag() * self.phase(0, -t)

                tau_xa = (tau_x * operator).tr()
                tau_ya = (tau_y * operator).tr()
                tau_za = (tau_z * operator).tr()
                tt = np.linalg.norm([tau_xa, tau_ya, tau_za])
                taux_ave.append(tau_xa / tt)
                tauy_ave.append(tau_ya / tt)
                tauz_ave.append(tau_za / tt)
        else:
            for t in self.tlist:
                psi_t_0 = self.GeneralStateDynamics(t, self.f_modes[0])
                psi_t_1 = self.GeneralStateDynamics(t, self.f_modes[1])

                tau_x = self.phase(0, t) * psi_t_0 * psi_t_1.dag() * self.phase(1, -t) \
                        + self.phase(1, t) * psi_t_1 * psi_t_0.dag() * self.phase(0, -t)

                tau_y = -1j * self.phase(1, t) * psi_t_1 * psi_t_0.dag() * self.phase(0, -t) \
                        + 1j * self.phase(0, t) * psi_t_0 * psi_t_1.dag() * self.phase(1, -t)

                tau_z = self.phase(1, t) * psi_t_1 * psi_t_1.dag() * self.phase(1, -t) \
                        - self.phase(0, t) * psi_t_0 * psi_t_0.dag() * self.phase(0, -t)

                tau_xa = (tau_x * operator).tr()
                tau_ya = (tau_y * operator).tr()
                tau_za = (tau_z * operator).tr()
                tt = np.linalg.norm([tau_xa, tau_ya, tau_za])
                taux_ave.append(tau_xa / tt)
                tauy_ave.append(tau_ya / tt)
                tauz_ave.append(tau_za / tt)

        self.taux_ave = np.array(taux_ave)
        self.tauy_ave = np.array(tauy_ave)
        self.tauz_ave = np.array(tauz_ave)

    def FloquetBlochSphere(self, vec):

        b = qt.Bloch()

        pnts = vec
        b.add_points(pnts)
        b.point_size = [1]
        # b.view = [-90,-90]
        b.render()
        b.show()

    def gcoeff(self, m, tau_ave):
        return (1 / self.T_drive) * np.sum(np.exp(1j * m * 2 * np.pi * self.tlist / self.T_drive) * tau_ave) * np.mean(
            np.diff(self.tlist))


#     def DecoherenceRates(self, operator):
#
#         self.Delta, self.X, self.Gamma, self.Amat = qt.floquet_master_equation_rates(
#             self.f_modes, self.f_energies, operator, self.H_dynamic, self.T_drive, args = None, noise_spectrum,
#             temp, kmax, self.fmodes_table
#         )
#
#
# kmax = 1
#
#         Delta, X, Gamma, Amat = floquet_master_equation_rates(
#             fq.f_modes, fq.f_energies, c_ops[0], fq.H_dynamic, fq.T_drive, args, noise_spectrum,
#             temp, kmax, fq.fmodes_table
#         )
#
#         eps = 1
#
#         while eps > 10 ** (-5):
#             kmax += 1
#
#             Delta1, X1, Gamma1, Amat1 = Delta, X, Gamma, Amat
#
#             Delta, X, Gamma, Amat = floquet_master_equation_rates(
#                 fq.f_modes, fq.f_energies, c_ops[0], fq.H_dynamic, fq.T_drive, args, noise_spectrum,
#                 temp, kmax, fq.fmodes_table
#             )
#
#             eps = np.sum(np.abs(Amat - Amat1))
