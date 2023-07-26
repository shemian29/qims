from typing import List

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.optimize import differential_evolution
import scipy as scp

static_pauli = {"x": qt.sigmax(),
                "y": qt.sigmay(),
                "z": qt.sigmaz()}

δf = 1.8 * (10 ** (-6))
EL = 1.3  # GHz
φge = 1.996
Af = 2 * np.pi * δf * EL * np.abs(φge)

nu0 = 0.1 # GHz
class FloquetQubit:

    def __init__(self, E01: float):
        self.E01 = E01

        # self.νfloquet = 0.3
        # self.ωfloquet = 2 * np.pi * self.νfloquet
        # self.T = 1 / self.νfloquet

        self.time_points = 200  # increases the number of points sampled on the frequency lattice since T is fixed

        #
        # self.νlist = np.fft.rfftfreq(len(self.tlist), np.mean(np.diff(self.tlist)))

        # self.νcut = self.νlist[-1]
        # self.ωcut = 2 * np.pi * self.νcut

        # self.dt = np.mean(np.diff(self.tlist))

    def su2_rotation(self, φ: float, θ: float, β: float) -> qt.Qobj:
        """Calculate the unitary matrix that maps static qubit states to Floquet qubit states

        :param φ: azimuthal angle
        :param θ: polar angle
        :param β: angle of rotation about the z-axis

        :return: 2x2 unitary matrix

        Example:
        >>> su2_rotation(0,0,0)

        """
        # Checked signs of angles in exponents

        # Rotation about z-axis by β
        exp_phi = np.exp(-1j * φ / 2)
        sm = qt.Qobj([[exp_phi, 0], [0, 1 / exp_phi]])

        # Rotation about y-axis by θ
        cos_th = np.cos(θ / 2)
        sin_th = np.sin(θ / 2)
        sm = sm * qt.Qobj([[cos_th, -sin_th], [sin_th, cos_th]])

        # Rotation about z-axis by β
        exp_beta = np.exp(-1j * β / 2)
        sm = sm * qt.Qobj([[exp_beta, 0], [0, 1 / exp_beta]])

        return sm

    def su2_rotation_dot(self, φ: float, θ: float, β: float, φ_dot: float, θ_dot: float, β_dot: float) -> qt.Qobj:
        """Calculate the unitary matrix that maps static qubit states to Floquet qubit states

        :param φ: azimuthal angle
        :param θ: polar angle
        :param β: angle of rotation about the z-axis
        :param φ_dot: derivative of φ with respect to time
        :param θ_dot: derivative of θ with respect to time
        :param β_dot: derivative of β with respect to time

        :return: time derivative of dmat

        Example:
        >>> su2_rotation_dot(0,0,0,0,0,0)

        """
        # Checked sign changes in su2_rotation matrices from taking derivative of su2_rotation

        sm = 0
        sm = sm + (-1j * φ_dot * static_pauli["z"] / 2) * self.su2_rotation(φ, θ, β)
        sm = sm + (-1j * θ_dot * static_pauli["y"] / 2) * self.su2_rotation(-φ, θ, β)
        sm = sm + (-1j * β_dot * static_pauli["z"] / 2) * self.su2_rotation(φ, -θ, β)

        return sm

    def so3_rotation(self, φ: float, θ: float, β: float) -> qt.Qobj:
        """Calculate the SO(3) orthogonal matrix associated with SU(2) matrix that maps static qubit states to Floquet qubit states

        :param φ: azimuthal angle
        :param θ: polar angle
        :param β: angle of rotation about the z-axis

        :return: 3x3 orthogonal matrix

        Example:
        >>> so3_rotation(0,0,0)

        """
        su2_rot = self.su2_rotation(φ, θ, β)

        axis_map = {"x": 0, "y": 1, "z": 2}
        tmp = np.zeros((3, 3))
        for a in ["x", "y", "z"]:
            floquet_pauli_a = su2_rot.dag() * static_pauli[a] * su2_rot
            for b in ["x", "y", "z"]:
                tmp[axis_map[a], axis_map[b]] = 0.5 * (floquet_pauli_a * static_pauli[b]).tr()

        return qt.Qobj(tmp)

    def angle_time(self, angle_freq: List[complex]) -> np.ndarray:
        """Calculate the temporal components of an angle variable from its frequency components.

        :param angle_freq: frequency components of the three angles φ, θ, β
        :return: list of angles in the time domain with number of time points equal to time_points

        Example:
        >>> angle_time([1,2*1j,3])
        """
        return np.fft.irfft(angle_freq, n=self.time_points, norm="forward")

    def angle_time_dot(self, angle_freq: complex, νfloquet: float) -> np.ndarray:
        """
        Calculate the temporal components of the time derivative of an angle variable from its frequency components.

        :param angle_freq: frequency components of the three angles φ, θ, β

        :return: list of time derivatives of angle in the time domain with number of time points equal to time_points

        Example:
        >>> angle_time_dot([1,2,3])
        """
        return np.fft.irfft((2*np.pi*νfloquet * 1j) * np.arange(0, len(angle_freq)) * angle_freq, n=self.time_points,
                            norm="forward")

    def su2_rotation_freq_to_time(self, angle_freq: complex) -> List[qt.Qobj]:
        """
        Create temporal list of SU(2) matrices that map static qubit states to Floquet qubit states in the time domain from its frequency components.
        :param angle_freq: frequency components of the three angles φ, θ, β
        :return: list of 2x2 matrices with number of time points equal to time_points

        Example:
        >>> su2_rotation_freq_to_time([1,2,3])
        """
        angle_freq_tmp = np.array(angle_freq).reshape((3, int(len(angle_freq) / 3)))

        φ_t = self.angle_time(angle_freq_tmp[0])
        θ_t = self.angle_time(angle_freq_tmp[1])
        β_t = self.angle_time(angle_freq_tmp[2])

        return [self.su2_rotation(φ_t[it], θ_t[it], β_t[it]) for it in range(self.time_points)]

    def su2_rotation_freq_to_time_dot(self, angle_freq: complex, νfloquet: float) -> List[qt.Qobj]:
        """
        Create temporal list of time-derivative of SU(2) matrices that map static qubit states to Floquet qubit states in the time domain from its frequency components.

        :param angle_freq: frequency components of the three angles φ, θ, β

        :return: list of 2x2 matrices with number of time points equal to time_points

        Example:
        >>> su2_rotation_freq_to_time_dot([1,2,3])
        """
        angle_freq_tmp = np.array(angle_freq).reshape((3, int(len(angle_freq) / 3)))

        φ_t = self.angle_time(angle_freq_tmp[0])
        θ_t = self.angle_time(angle_freq_tmp[1])
        β_t = self.angle_time(angle_freq_tmp[2])

        φ_tdot = self.angle_time_dot(angle_freq_tmp[0], νfloquet)
        θ_tdot = self.angle_time_dot(angle_freq_tmp[1], νfloquet)
        β_tdot = self.angle_time_dot(angle_freq_tmp[2], νfloquet)

        return [self.su2_rotation_dot(φ_t[it], θ_t[it], β_t[it], φ_tdot[it], θ_tdot[it], β_tdot[it]) for it in
                range(self.time_points)]

    def hamiltonian(self, ε01: float, su2_rotation: List[qt.Qobj], su2_rotation_dot: List[qt.Qobj]) -> List[qt.Qobj]:
        """
        Calculate the time-dependent Hamiltonian of the Floquet qubit.
        :param ε01: Floquet quasi-energy
        :param su2_rotation: list of SU(2) rotations that map static qubit states to Floquet qubit states
        :param su2_rotation_dot: list of time derivatives of SU(2) rotations that map static qubit states to Floquet qubit states

        :return: list of 2x2 Hamiltonian matrices with number of time points equal to time_points

        Example:
        >>> hamiltonian(1, FQ.su2_rotation_freq_to_time([0.1,0.2,0.3]), FQ.su2_rotation_freq_to_time_dot([0.1,0.2,0.3]))
        """
        return [0.5 * ε01 * su2_rotation[it] * static_pauli["z"] * su2_rotation[it].dag() + 1j * su2_rotation_dot[it] *
                su2_rotation[it].dag()
                for it in range(self.time_points)]

    # def ε01_constraint(self,  su2_rotation: List[qt.Qobj], su2_rotation_dot: List[qt.Qobj]) -> List[qt.Qobj]:
    #     """
    #     Calculate the time-dependent Hamiltonian of the Floquet qubit.
    #     :param ε01: Floquet quasi-energy
    #     :param su2_rotation: list of SU(2) rotations that map static qubit states to Floquet qubit states
    #     :param su2_rotation_dot: list of time derivatives of SU(2) rotations that map static qubit states to Floquet qubit states
    #
    #     :return: list of 2x2 Hamiltonian matrices with number of time points equal to time_points
    #
    #     Example:
    #     >>> hamiltonian(1, FQ.su2_rotation_freq_to_time([0.1,0.2,0.3]), FQ.su2_rotation_freq_to_time_dot([0.1,0.2,0.3]))
    #     """
    #
    #
    #     [0.5 * ε01 * su2_rotation[it] * static_pauli["z"] * su2_rotation[it].dag() + 1j * su2_rotation_dot[it] *
    #      su2_rotation[it].dag()
    #      for it in range(self.time_points)]
    #
    #     term_1 = [0.5 * self.E01 * static_pauli['z'] - 1j * su2_rotation_dot[it] *
    #      su2_rotation[it].dag()
    #      for it in range(self.time_points)]
    #
    #     term_2 = [0.5 * self.E01 * static_pauli['z'] - 1j * su2_rotation_dot[it] *
    #      su2_rotation[it].dag()
    #      for it in range(self.time_points)]
    #
    #     return
    def hstatic_matching(self, parameters: List[complex]) -> float:
        """
        Calculate the cost function for matching the static qubit Hamiltonian to the Floquet qubit Hamiltonian.

        :param parameters: list of parameters to be optimized. The first parameter is the Floquet quasi-energy ε01. The remaining parameters are the frequency components of the three angles φ, θ, β.
        :return: cost function value

        Example:
        >>> hstatic_matching([1,2,3,4])

        """
        number_freq = int(len(parameters[2:]) / 6)
        ε01 = parameters[0]
        νfloquet = parameters[1]
        angle_frequencies = parameters[2:3*number_freq+2]+1j*parameters[3*number_freq+2:6*number_freq + 2]
        su2_rot = self.su2_rotation_freq_to_time(angle_frequencies)
        su2_rot_dot = self.su2_rotation_freq_to_time_dot(angle_frequencies, νfloquet)

        hstatic = 0.5 * self.E01 * static_pauli['z']

        h_time = self.hamiltonian(ε01, su2_rot, su2_rot_dot)
        h_time_average = (1/self.time_points) * np.sum(np.array([h_time[it] for it in range(self.time_points)]), axis=0)

        return (hstatic - h_time_average).norm()

    def delta(self, m, n):
        if m == n:
            return 1
        else:
            return 0

    def Sf(self, ω):
        """
        Calculate the spectral density associated with flux noise.

        :param ω: frequency in GHz
        :return: spectral density value in kHz

        """
        δf = 1.8 * (10 ** (-6))
        EL = 1.3  # GHz
        φge = 1.996
        Af = 2 * np.pi * δf * EL * np.abs(φge)
        return (Af ** 2) * np.abs((2 * np.pi) / ω) * (10 ** 6) * (2 * np.pi)  # kHz

    def Sd(self, ω: float) -> float:
        """
        Calculate the spectral density associated with dielectric noise. Frequency in GHz.

        :param ω: frequency in GHz
        :return: spectral density value in kHz

        """
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

    def spectral_density(self, ω):
        return (self.Sf(ω) + self.Sd(ω))

    def decoherence_rate(self, parameters: List[complex]) -> float:
        """
        Calculate the decoherence rate of a fluxonium Floquet qubit.
        :param parameters: list of parameters to be optimized. The first parameter is the Floquet quasi-energy ε01. The remaining parameters are the frequency components of the three angles φ, θ, β.
        :return: decoherence rate

        Example:
        >>> decoherence_rate([1,2,3,4])
        """
        #
        # parameters = np.array([parameters[0],  # quasi-energy
        #                        parameters[1],  # νfloquet
        #                        parameters[2],  # zero-frequency component, which is real
        #                        tmp[0] + 1j * tmp[
        #                            1]])  # non-zero complex frequency components of the three angles φ, θ, β

        ε01 = parameters[0]
        νfloquet = parameters[1]

        angle_frequencies = parameters.reshape((2, int(parameters[3:].shape[0] / 2)))
        angle_frequencies = np.concatenate([parameters[2],angle_frequencies[0]+1j*angle_frequencies[1]])
        su2_rot = self.su2_rotation_freq_to_time(angle_frequencies)

        gzx = np.fft.rfft(
            [(su2_rot[it].dag() * static_pauli["x"] * su2_rot[it] * static_pauli["z"]).tr() for it in
             range(self.time_points)],
            norm='forward')
        gzy = np.fft.rfft(
            [(su2_rot[it].dag() * static_pauli["y"] * su2_rot[it] * static_pauli["z"]).tr() for it in
             range(self.time_points)],
            norm='forward')
        gzz = np.fft.rfft(
            [(su2_rot[it].dag() * static_pauli["z"] * su2_rot[it] * static_pauli["z"]).tr() for it in
             range(self.time_points)],
            norm='forward')

        gamma_0 = Af * 2 * (np.abs(gzz[0])) * 4 * (10 ** 6)
        mlist_aux = np.arange(1, len(gzz))

        dephasing = gamma_0 + 2 * np.dot((np.abs(gzz[1:]) ** 2),
                                         self.spectral_density(mlist_aux * νfloquet) + self.spectral_density(
                                             -mlist_aux * νfloquet))

        # mlist = np.arange(0,len(gzz))
        depolarization = np.dot(
            (np.abs(gzx - 1j * gzy) ** 2),
            np.concatenate(([self.spectral_density(0 * νfloquet + ε01)],
                            self.spectral_density(mlist_aux * νfloquet + ε01) + self.spectral_density(
                                -mlist_aux * νfloquet + ε01))),
        )

        excitation = np.dot(
            (np.abs(gzx + 1j * gzy) ** 2),
            np.concatenate(([self.spectral_density(0 * νfloquet - ε01)],
                            self.spectral_density(mlist_aux * νfloquet - ε01) + self.spectral_density(
                                -mlist_aux * νfloquet - ε01))),
        )

        return 0.5 * (depolarization + excitation) + dephasing

    def cost_function(self, parameters: List[complex], hyper_parameter: float = 100, normalization: float = 1) -> float:
        """
        Calculate the cost function for the optimization algorithm.

        :param parameters: list of parameters to be optimized. The first parameter is the Floquet quasi-energy ε01. The remaining parameters are the frequency components of the three angles φ, θ, β.
        :param hyper_parameter: controls the relative weight of the decoherence rate and the static qubit Hamiltonian matching.
        :param normalization: normalization factor for the cost function with respect to the static system.

        :return: cost function value

        Example:
        >>> cost_function([1,2,3,4], hyper_parameter=100, normalization=1)
        """

        tmp = parameters.reshape((2, int(parameters[5:].shape[0] / 2)))
        parameters = np.array([parameters[0],# quasi-energy
                               parameters[1],# νfloquet
                               parameters[2:5],# zero-frequency component, which is real, for all angles φ, θ, β
                               tmp[0]+1j*tmp[1]]) # non-zero complex frequency components of the three angles φ, θ, β
        decoh_rate = hyper_parameter * self.decoherence_rate(parameters)

        return (self.hstatic_matching(parameters) / decoh_rate + decoh_rate) / normalization

    def record_differential_optimization_path(self, parameters, convergence=1):
        """
        Record the parameters, cost function, and h0 matching at each iteration of the differential evolution optimization algorithm.
        :param parameters: list of parameters to be optimized. The first parameter is the Floquet quasi-energy ε01. The remaining parameters are the frequency components of the three angles φ, θ, β.
        :param convergence: convergence parameter of the differential evolution optimization algorithm.

        Example:
        >>> record_differential_optimization_path([1,2,3,4], convergence=1)
        """
        global iteration_step, parameters_differential_evolution, rate_record, h0match_rec

        parameters_differential_evolution.append(parameters)
        rate_record.append(self.decoherence_rate(parameters))
        h0match_rec.append(self.hstatic_matching(parameters))

        np.savetxt('parameters_differential_evolution.txt', parameters_differential_evolution, delimiter=',')
        np.savetxt('rate_record.txt', rate_record)
        np.savetxt('h0match_rec_FQ.txt', h0match_rec)

        iteration_step += 1

    def search_floquet_qubit(self, number_frequencies: int = 2, maxiter: int = 10000000) -> scp.optimize.OptimizeResult:
        """
        Search for the optimal Floquet qubit parameters.

        :param number_frequencies: Number of frequency components for each angle to optimize. The rest is assumed to be zero.
        :param maxiter: maximum number of iterations for the optimization algorithm.

        :return: data from differential evolution optimization algorithm

        Example:
        >>> search_floquet_qubit(number_frequencies=2, maxiter=10000000)
        """

        global parameters_differential_evolution, iteration_step, rate_record, h0match_rec

        iteration_step = 1
        parameters_differential_evolution = []
        rate_record = []
        h0match_rec = []
        return differential_evolution(
            self.cost_function,
            [(0, 10*np.max([self.E01,nu0])), (0.1*np.min([self.E01,nu0]),10*np.max([self.E01,nu0]))]
            + [(-1.0, 1.0) for _ in range(3 * 2 * number_frequencies-3)],
            callback=self.record_differential_optimization_path,
            disp=True,
            workers=-1,  # Use all cores available
            popsize=100,
            init="halton",  # Best initialization of parameters according to Scipy documentation
            strategy="randtobest1bin",
            # This option appears to work well based on a few test cases # "currenttobest1bin", #"randtobest1bin",
            maxiter=maxiter,
        )

    def plot_data(self, solx, h_time, iteration, rate_data, freq_data, ε01s, dmat_time, dmat_timedot):

        self.tlist = np.linspace(0, 1 / self.νfloquet, self.time_points)

        ncols, nrows = 3, 5
        fig, axs = plt.subplots(nrows, ncols, figsize=(24, 10))
        freq_num = int(len(solx[1:]) / 3)

        rgb_colors = [(0.2, 0.4, 0.6), (0.8, 0.2, 0.4), (0.6, 0.4, 0.2)]

        axs[0][0].set_title('angle dynamics', fontsize=15)
        axs[0][0].plot(self.tlist / self.T, (self.angle_time(solx[1:freq_num + 1]) % (2 * np.pi)) / (2 * np.pi), '.',
                       label='φ(t)')
        axs[0][0].plot(self.tlist / self.T,
                       (self.angle_time(solx[1 + freq_num:freq_num + 1 + freq_num]) % (2 * np.pi)) / (2 * np.pi), '.',
                       label='θ(t)')
        axs[0][0].plot(self.tlist / self.T,
                       (self.angle_time(solx[1 + 2 * freq_num:freq_num + 1 + 2 * freq_num]) % (2 * np.pi)) / (
                                   2 * np.pi), '.',
                       label='β(t)')
        axs[0][0].set_xlim([0, 1])
        axs[0][0].set_ylim([0, 1])
        axs[0][0].set_ylabel(r'(Euler angle)$/(2\pi)$', fontsize=15)
        axs[0][0].legend()

        axs[1][0].set_title('periodic drives', fontsize=15)
        axs[1][0].plot(self.tlist / self.T, 0.5 * np.real([(h_time[it] * static_pauli["x"]).tr()
                                                           for it in range(self.time_points)]) / self.E01, '.-',
                       label=r'$h_x(t)$',
                       color=rgb_colors[0])
        axs[1][0].plot(self.tlist / self.T, 0.5 * np.real([(h_time[it] * static_pauli["y"]).tr()
                                                           for it in range(self.time_points)]) / self.E01, '.-',
                       label=r'$h_y(t)$',
                       color=rgb_colors[1])
        axs[1][0].plot(self.tlist / self.T, 0.5 * np.real([(h_time[it] * static_pauli["z"]).tr()
                                                           for it in range(self.time_points)]) / self.E01, '.-',
                       label=r'$h_z(t)$',
                       color=rgb_colors[2])
        axs[1][0].set_xlim([0, 1])
        axs[1][0].set_xlabel('time/T', fontsize=15)
        axs[1][0].set_ylabel(r'$h_a(t)/E_{01}$', fontsize=15)
        axs[1][0].legend()

        axs[0][1].plot(rate_data / self.decoherence_rate([self.E01, 0, 0, 0]), '.-');
        axs[0][1].set_title('iteration = ' + str(iteration))
        axs[0][1].set_ylabel(r'$\Gamma_2/\Gamma^{(0)}_2$', fontsize=15)

        axs[1][1].plot(np.loadtxt('h0match_rec_FQ.txt', delimiter=',') / self.E01, '.-');
        axs[1][1].set_title('iteration = ' + str(iteration))
        axs[1][1].set_xlabel('iteration', fontsize=15)
        axs[1][1].set_ylabel(r'(h0 matching)$/E01$', fontsize=15)

        axs[2][1].imshow(np.abs(freq_data[:, 1:freq_num + 1].T), aspect='auto', cmap='plasma', origin='lower')
        axs[2][1].set_title(r'$\vert\widetilde{φ}(m)\vert$ for iteration = ' + str(iteration))
        axs[2][1].set_xlabel('iteration', fontsize=15)
        axs[2][1].set_ylabel(r'$\omega/ω_{floquet}$', fontsize=15)

        axs[3][1].imshow(np.abs(freq_data[:, 1 + freq_num:freq_num + 1 + freq_num].T), aspect='auto', cmap='plasma',
                         origin='lower')
        axs[3][1].set_title(r'$\vert\widetilde{θ}(m)\vert$ for iteration = ' + str(iteration))
        axs[3][1].set_xlabel('iteration', fontsize=15)
        axs[3][1].set_ylabel(r'$\omega/ω_{floquet}$', fontsize=15)

        axs[4][1].imshow(np.abs(freq_data[:, 1 + 2 * freq_num:freq_num + 1 + 2 * freq_num].T), aspect='auto',
                         cmap='plasma',
                         origin='lower')
        axs[4][1].set_title(r'$\vert\widetilde{β}(m)\vert$ for iteration = ' + str(iteration))
        axs[4][1].set_xlabel('iteration', fontsize=15)
        axs[4][1].set_ylabel(r'$\omega/ω_{floquet}$', fontsize=15)

        # cbar = fig.colorbar(im, ax = axs)

        # axs[3][1].imshow(freq_data[:,:].T, aspect='auto')

        axs[2][0].set_title('angle dynamics', fontsize=15)
        axs[2][0].plot(self.tlist / self.T, (self.angle_time(solx[1:freq_num + 1])) / (2 * np.pi), '.', label='φ(t)')
        axs[2][0].plot(self.tlist / self.T, (self.angle_time(solx[1 + freq_num:freq_num + 1 + freq_num])) / (2 * np.pi),
                       '.', label='θ(t)')
        axs[2][0].plot(self.tlist / self.T,
                       (self.angle_time(solx[1 + 2 * freq_num:freq_num + 1 + 2 * freq_num])) / (2 * np.pi), '.',
                       label='β(t)')
        axs[2][0].set_xlim([0, 1])
        # axs[2][0].set_ylim([0,1])
        axs[2][0].set_ylabel(r'(Euler angle)$/(2\pi)$', fontsize=15)
        axs[2][0].legend()

        # plt.sca(axs[0, 2])
        # b.axes

        [axs[r][s].grid(True, color='gray', linestyle='--', linewidth=0.5) for r in range(nrows) for s in range(ncols)]
        [axs[r][s].tick_params(axis='both', which='both', labelsize=15) for r in range(nrows) for s in range(ncols)]
        plt.subplots_adjust(top=1.5, hspace=0.5, wspace=0.25)
