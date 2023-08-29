from typing import List

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from numpy import ndarray
from scipy.optimize import differential_evolution
import scipy as scp

static_pauli = {"x": qt.sigmax(),
                "y": qt.sigmay(),
                "z": qt.sigmaz()}

δf = 1.8 * (10 ** (-6))
EL = 1.3  # GHz
φge = 1.996  # fluxonium matrix element with respect to 0.5*phi0 flux value
Af = 2 * np.pi * δf * EL * np.abs(φge)

nu0 = 0.1  # GHz


class FloquetQubit_alt:

    # def __init__(self, ν01_0: float, φ_0: float = 0, θ_0: float = 0, β_0: float = 0):
    def __init__(self, system):

        self.optimal_qubit = None
        self.cost0 = None
        self.time_points = 200  # increases the number of points sampled on the frequency lattice since T is fixed

        self.floquet_qubit_parameters = system

        self.φ_0 = system["static"]["φ_0"]
        self.θ_0 = system["static"]["θ_0"]
        self.β_0 = system["static"]["β_0"]
        self.E01 = system["static"]["ν01"]
        self.h_qubit = 0.5 * self.E01 * (np.cos(self.φ_0) * np.sin(self.θ_0) * static_pauli["x"] +
                                         np.sin(self.φ_0) * np.sin(self.θ_0) * static_pauli["y"] +
                                         np.cos(self.θ_0) * static_pauli["z"])
        # self.tlist = np.arange(0, 1 / system["dynamic"]["νfloquet"],
        #                        1 / (self.time_points * system["dynamic"]["νfloquet"]))
        self.tlist = np.linspace(0, 1 / system["dynamic"]["νfloquet"],
                               self.time_points+1)[:-1]
        self.νfloquet = system["dynamic"]["νfloquet"]
    def _repr_latex_(self):

        string = "Floquet qubit of the form: "
        string = (
                string
                + r"$\vert v_{\sigma}(t)\rangle = e^{-i(-1)^{\sigma} \frac{\beta_(t)}{2}}\left( \begin{array}{c}  e^{"
                  r"-i \varphi(t)/2} \cos\left(\frac{\theta(t)+\sigma \pi}{2}\right)  \\ e^{i \varphi(t)/2}\sin\left("
                  r"\frac{\theta(t)+\sigma \pi}{2}\right)  \end{array} \right).   $"
        )

        return string
    def update_time_points(self, time_points: int):
        """
        Update the number of points.

        :param time_points: number of points.

        Example:
        >>> update_time_points(200)
        """
        self.time_points = time_points
        # self.tlist = np.arange(0, 1 / self.floquet_qubit_parameters["dynamic"]["νfloquet"],
        #                        1 / (self.time_points * self.floquet_qubit_parameters["dynamic"]["νfloquet"]))
        self.tlist = np.linspace(0, 1 / self.floquet_qubit_parameters["dynamic"]["νfloquet"],
                                 self.time_points + 1)[:-1]
    def search_floquet_qubit(self, number_frequencies: int = 2, maxiter: int = 10000000) -> scp.optimize.OptimizeResult:
        """
        Search for the optimal Floquet qubit parameters.

        :param number_frequencies: Number of frequency components for each angle to optimize. The rest is assumed to
        be zero. :param maxiter: maximum number of iterations for the optimization algorithm.

        :return: data from differential evolution optimization algorithm

        Example:
        >>> search_floquet_qubit(number_frequencies=2, maxiter=10000000)
        """

        global parameters_differential_evolution, iteration_step, rate_record, h0match_rec

        try:
            self.cost0 = self.cost_function([self.E01, 12.3 * self.E01, self.φ_0, self.θ_0, self.β_0],
                                            normalize=False)  # The choice of Floquet frequency is arbitrary since it
            # does not affect the static system
        except:
            print('Issue encountered calculating the cost function for the static system')

        iteration_step = 1
        parameters_differential_evolution = []
        rate_record = []
        h0match_rec = []
        quasi_energy_range = (0.0123456, 10 * np.max([self.E01, nu0]))
        frequency_range = (0.1 * np.min([self.E01, nu0]), 10 * np.max([self.E01, nu0]))
        frequency_components_range = (-1.0, 1.0)
        self.optimal_qubit = differential_evolution(
            self.cost_function,
            [quasi_energy_range, frequency_range]
            + [frequency_components_range for _ in range(3 * 2 * number_frequencies - 3)],
            # x3 angles, x2 numbers per complex component, -3 for the zero frequency components which have no
            # imaginary component
            callback=self.record_differential_optimization_path,
            disp=True,
            workers=-1,  # Use all cores available
            popsize=100,
            init="halton",  # Best initialization of parameters according to Scipy documentation
            strategy="randtobest1bin",
            # This option appears to work well based on a few test cases # "currenttobest1bin", #"randtobest1bin",
            maxiter=maxiter,
        )

    def cost_function(self, parameters_extended: np.ndarray, hyper_parameter: float = 100, normalize=True) -> float:
        """
        Calculate the cost function for the optimization algorithm.

        :param parameters_extended: list of parameters to be optimized. The first parameter is the Floquet
        quasi-energy ε01. The remaining parameters are the frequency components of the three angles φ, θ,
        β. :param hyper_parameter: controls the relative weight of the decoherence rate and the static qubit
        Hamiltonian matching. :param normalization: normalization factor for the cost function with respect to the
        static system.

        :return: cost function value

        Example:
        >>> cost_function([1,2,3,4], hyper_parameter=100, normalize=True)
        """
        parameters_extended = np.array(parameters_extended)

        frequency_components = self.complexify(parameters_extended)
        internal_floquet_qubit_parameters = {
            "dynamic": {
                "ε01": parameters_extended[0].real,
                "φ_m": frequency_components[0].tolist(),
                "θ_m": frequency_components[1].tolist(),
                "β_m": frequency_components[2].tolist(),
                "νfloquet": parameters_extended[1].real
            }
        }

        decoh_rate = hyper_parameter * self.decoherence_rate(parameters=internal_floquet_qubit_parameters)

        if normalize:
            return (self.hstatic_matching(
                parameters=internal_floquet_qubit_parameters) / decoh_rate + decoh_rate) / self.cost0
        else:
            return self.hstatic_matching(parameters=internal_floquet_qubit_parameters) / decoh_rate + decoh_rate

    def record_differential_optimization_path(self, parameters, convergence=1):
        """
        Record the parameters, cost function, and h0 matching at each iteration of the differential evolution
        optimization algorithm. :param parameters: list of parameters to be optimized. The first parameter is the
        Floquet quasi-energy ε01. The remaining parameters are the frequency components of the three angles φ, θ,
        β. :param convergence: convergence parameter of the differential evolution optimization algorithm.

        Example:
        >>> record_differential_optimization_path([0.1, 0.2, 1, 0.5, 0.4, 0.4, 0.4, 0.9, 0.12, 0.4, 0.43], convergence=1)
        """
        global iteration_step, parameters_differential_evolution, rate_record, h0match_rec

        parameters_differential_evolution.append(parameters)
        rate_record.append(self.decoherence_rate(parameters))
        h0match_rec.append(self.hstatic_matching(parameters))

        np.savetxt('parameters_differential_evolution.txt', parameters_differential_evolution, delimiter=',')
        np.savetxt('rate_record.txt', rate_record)
        np.savetxt('h0match_rec_FQ.txt', h0match_rec)

        iteration_step += 1

    def decoherence_rate(self, parameters=None) -> float:
        """
        Calculate the decoherence rate of a fluxonium Floquet qubit. :param parameters: list of parameters to be
        optimized. The first parameter is the Floquet quasi-energy ε01. The remaining parameters are the frequency
        components of the three angles φ, θ, β. :return: decoherence rate

        Example:
        >>> decoherence_rate()
        """
        if parameters is None:
            parameters = self.floquet_qubit_parameters

        ε01 = parameters['dynamic']['ε01']
        νfloquet = parameters['dynamic']['νfloquet']

        # print(parameters)
        su2_rot = self.su2_rotation_freq_to_time(parameters=parameters)

        Rxx_freq = np.fft.rfft(
            [(su2_rot[it].dag() * static_pauli["x"] * su2_rot[it] * static_pauli["x"]).tr() for it in
             range(self.time_points)],
            norm='forward')
        Ryx_freq = np.fft.rfft(
            [(su2_rot[it].dag() * static_pauli["y"] * su2_rot[it] * static_pauli["x"]).tr() for it in
             range(self.time_points)],
            norm='forward')
        Rzx_freq = np.fft.rfft(
            [(su2_rot[it].dag() * static_pauli["z"] * su2_rot[it] * static_pauli["x"]).tr() for it in
             range(self.time_points)],
            norm='forward')

        gamma_0 = Af * 2 * (np.abs(Rzx_freq[0])) * 4 * (10 ** 6)
        mlist_aux = np.arange(1, len(Rzx_freq))

        dephasing = gamma_0 + 2 * np.dot((np.abs(Rzx_freq[1:]) ** 2),
                                         self.spectral_density(mlist_aux * νfloquet) + self.spectral_density(
                                             -mlist_aux * νfloquet))

        depolarization = np.dot(
            (np.abs(Rxx_freq - 1j * Ryx_freq) ** 2),
            np.concatenate(([self.spectral_density(0 * νfloquet + ε01)],
                            self.spectral_density(mlist_aux * νfloquet + ε01) + self.spectral_density(
                                -mlist_aux * νfloquet + ε01))),
        )

        excitation = np.dot(
            (np.abs(Rxx_freq + 1j * Ryx_freq) ** 2),
            np.concatenate(([self.spectral_density(0 * νfloquet - ε01)],
                            self.spectral_density(mlist_aux * νfloquet - ε01) + self.spectral_density(
                                -mlist_aux * νfloquet - ε01))),
        )

        return 0.5 * (depolarization + excitation) + dephasing  # , dephasing, depolarization, excitation,gzx,gzy,gzz

    def hstatic_matching(self, parameters=None) -> float:
        """
        Calculate the cost function for matching the static qubit Hamiltonian to the Floquet qubit Hamiltonian.

        :param parameters: list of parameters to be optimized. The first parameter is the Floquet quasi-energy ε01.
        The remaining parameters are the frequency components of the three angles φ, θ, β. :return: cost function value

        Example:
        >>> hstatic_matching()
        """
        if parameters is None:
            parameters = self.floquet_qubit_parameters

        h_time = self.hamiltonian()
        h_time_average = (1 / self.time_points) * np.sum(np.array([h_time[it] for it in range(self.time_points)]),
                                                         axis=0)

        return (self.h_qubit - h_time_average).norm() + np.sum(
            np.abs([(static_pauli['z'] * h_time[it]).tr() for it in range(self.time_points)])) * (
                1 / self.time_points) + np.sum(
            np.abs([(static_pauli['y'] * h_time[it]).tr() for it in range(self.time_points)])) * (1 / self.time_points)

    def hamiltonian(self) -> List[qt.Qobj]:
        """
        Calculate the time-dependent Hamiltonian of the Floquet qubit. :param ε01: Floquet quasi-energy :param
        su2_rotation: list of SU(2) rotations that map static qubit states to Floquet qubit states :param
        su2_rotation_dot: list of time derivatives of SU(2) rotations that map static qubit states to Floquet qubit
        states

        :return: list of 2x2 Hamiltonian matrices with number of time points equal to time_points

        Example:
        >>> hamiltonian()
        """
        ε01 = self.floquet_qubit_parameters['dynamic']['ε01']

        su2_rot = self.su2_rotation_freq_to_time()
        su2_rot_dot = self.su2_rotation_freq_to_time_dot()

        return [0.5 * ε01 * su2_rot[it] * static_pauli["z"] * su2_rot[it].dag()
                + 1j * su2_rot_dot[it] * su2_rot[it].dag()
                for it in range(self.time_points)]

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

    def check_su2_rotation_freq_to_time_dot(self, parameters=None) -> ndarray:
        print("Check time derivative of SU(2) matrix by comparing analytic with numerical expressions:")
        su2fq = self.su2_rotation_freq_to_time()
        su2fq_dot = self.su2_rotation_freq_to_time_dot()
        return np.array([(((su2fq[it + 1] - su2fq[it]) / np.diff(self.tlist)[0]) - su2fq_dot[it]).full()
                         for it in range(self.time_points - 1)])

    def su2_rotation_freq_to_time(self, parameters=None) -> List[qt.Qobj]:
        """
        Create temporal list of SU(2) matrices that map static qubit states to Floquet qubit states in the time
        domain from its frequency components. :param angle_freq: frequency components of the three angles φ, θ,
        β :return: list of 2x2 matrices with number of time points equal to time_points

        Example:
        >>> su2_rotation_freq_to_time()
        """
        if parameters is None:
            parameters = self.floquet_qubit_parameters

        angle_freq = parameters['dynamic']['φ_m']  # As complex numbers
        angle_freq = angle_freq + parameters['dynamic']['θ_m']  # As complex numbers
        angle_freq = angle_freq + parameters['dynamic']['β_m']  # As complex numbers

        angle_freq_tmp = np.array(angle_freq).reshape((3, int(len(angle_freq) / 3)))

        φ_t = self.angle_time(angle_freq_tmp[0])
        θ_t = self.angle_time(angle_freq_tmp[1])
        β_t = self.angle_time(angle_freq_tmp[2])

        return [self.su2_rotation(φ_t[it], θ_t[it], β_t[it]) for it in range(self.time_points)]

    def su2_rotation_freq_to_time_dot(self, parameters=None) -> List[qt.Qobj]:
        """
        Create temporal list of time-derivative of SU(2) matrices that map static qubit states to Floquet qubit
        states in the time domain from its frequency components.

        :param angle_freq: frequency components of the three angles φ, θ, β

        :return: list of 2x2 matrices with number of time points equal to time_points

        Example:
        >>> su2_rotation_freq_to_time_dot()
        """
        if parameters is None:
            parameters = self.floquet_qubit_parameters

        νfloquet = parameters['dynamic']['νfloquet']
        angle_freq = parameters['dynamic']['φ_m']  # As complex numbers
        angle_freq = angle_freq + parameters['dynamic']['θ_m']  # As complex numbers
        angle_freq = angle_freq + parameters['dynamic']['β_m']  # As complex numbers

        angle_freq_tmp = np.array(angle_freq).reshape((3, int(len(angle_freq) / 3)))

        φ_t = self.angle_time(angle_freq_tmp[0])
        θ_t = self.angle_time(angle_freq_tmp[1])
        β_t = self.angle_time(angle_freq_tmp[2])

        φ_tdot = self.angle_time_dot(angle_freq_tmp[0])
        θ_tdot = self.angle_time_dot(angle_freq_tmp[1])
        β_tdot = self.angle_time_dot(angle_freq_tmp[2])

        return [self.su2_rotation_dot(φ_t[it], θ_t[it], β_t[it], φ_tdot[it], θ_tdot[it], β_tdot[it]) for it in
                range(self.time_points)]

    def so3_rotation(self, φ: float, θ: float, β: float) -> qt.Qobj:
        """Calculate the SO(3) orthogonal matrix associated with SU(2) matrix that maps static qubit states to
        Floquet qubit states

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

        :param angle_freq: frequency components of the three angles φ, θ, β :return: list of angles in the time
        domain with number of time points equal to time_points. Assumes padding with zeros in the frequency domain.
        The first component is the zero-frequency component and is real, and the rest are complex.

        Example:
        >>> angle_time([1,2*1j,3])
        """
        return np.fft.irfft(angle_freq, n=self.time_points, norm="forward")

    def angle_time_dot(self, angle_freq: complex) -> np.ndarray:
        """
        Calculate the temporal components of the time derivative of an angle variable from its frequency components.

        :param angle_freq: frequency components of the three angles φ, θ, β

        :return: list of time derivatives of angle in the time domain with number of time points equal to time_points

        Example:
        >>> angle_time_dot([1,2,3])
        """
        return np.fft.irfft((2 * np.pi * self.νfloquet * 1j) * np.arange(0, len(angle_freq)) * angle_freq,
                            n=self.time_points,
                            norm="forward")

    def drives_xyz(self) -> float:
        """
        Calculate the periodic drives derived from the inferred Hamiltonian

        :param parameters: list of parameters to be optimized. The first parameter is the Floquet quasi-energy ε01.
        The remaining parameters are the frequency components of the three angles φ, θ, β. :return: drives in the x,
        y, z directions

        Example:
        >>> drives_xyz()
        """

        h_time = self.hamiltonian()

        return np.real(np.array([[(static_pauli['x'] * h_time[it]).tr() / 2,
                                  (static_pauli['y'] * h_time[it]).tr() / 2,
                                  (static_pauli['z'] * h_time[it]).tr() / 2]
                                 for it in range(self.time_points)]).T)

    def complexify(self, parameters_extended):
        # Count the number of frequency components by discarding the quasi-energy and Floquet frequency components
        num_freqs = int((int(len(parameters_extended[2:]) / 3) + 1) / 2)
        frequency_components = np.array(parameters_extended[2:]).reshape((3, 2 * num_freqs - 1))
        frequency_components = np.insert(
            frequency_components[:, 1:2 * num_freqs - 1:2] + 1j * frequency_components[:, 2:2 * num_freqs - 1:2], 0,
            frequency_components[:, 0], axis=1)
        return frequency_components

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
        # α = np.abs(np.cosh(λ * ω / (2 * Temp)) / np.sinh(λ * ω / (2 * Temp)) + 1) / 2
        α = np.abs(1 / np.tanh(λ * ω / (2 * Temp)) + 1) / 2
        φge = 1.996
        tanδc = 1.1 * (10 ** (-6))
        Ad = (np.pi ** 2) * tanδc * (φge ** 2) / EC
        return α * Ad * ((ω / (2 * np.pi)) ** 2) * (10 ** 6) * (2 * np.pi)  # kHz

    def spectral_density(self, ω):
        return (self.Sf(ω) + self.Sd(ω))

    def plot_data(self, rate_data, h0match_data, full_data):

        final_data = full_data[-1]
        ε01s = final_data[0]
        νfloquet = final_data[1]

        parameters_extended = np.array(final_data)
        frequency_components = self.complexify(parameters_extended)

        su2_time = self.su2_rotation_freq_to_time(frequency_components.flatten())
        su2_timedot = self.su2_rotation_freq_to_time_dot(frequency_components.flatten(), νfloquet)

        # self.tlist = np.linspace(0, 1 / νfloquet, self.time_points)
        self.tlist = np.arange(0, 1 / νfloquet, 1 / (self.time_points * νfloquet))
        h_time = self.hamiltonian(ε01s, su2_time, su2_timedot)

        ncols, nrows = 3, 5
        fig, axs = plt.subplots(nrows, ncols, figsize=(24, 10))

        rgb_colors = [(0.2, 0.4, 0.6), (0.8, 0.2, 0.4), (0.6, 0.4, 0.2)]

        axs[0][0].set_title('angle dynamics', fontsize=15)
        axs[0][0].plot(self.tlist * νfloquet, (self.angle_time(frequency_components[0])) / (2 * np.pi), '.',
                       label='φ(t)')
        axs[0][0].plot(self.tlist * νfloquet,
                       (self.angle_time(frequency_components[1])) / (2 * np.pi), '.',
                       label='θ(t)')
        axs[0][0].plot(self.tlist * νfloquet,
                       (self.angle_time(frequency_components[2])) / (
                               2 * np.pi), '.',
                       label='β(t)')
        axs[0][0].set_xlim([0, 1])
        axs[0][0].set_ylabel(r'(Euler angle)$/(2\pi)$', fontsize=15)
        axs[0][0].legend()

        axs[1][0].set_title('periodic drives', fontsize=15)
        axs[1][0].plot(self.tlist * νfloquet, 0.5 * np.real([(h_time[it] * static_pauli["x"]).tr()
                                                             for it in range(self.time_points)]) / self.E01, '.-',
                       label=r'$h_x(t)$',
                       color=rgb_colors[0])
        axs[1][0].plot(self.tlist * νfloquet, 0.5 * np.real([(h_time[it] * static_pauli["y"]).tr()
                                                             for it in range(self.time_points)]) / self.E01, '.-',
                       label=r'$h_y(t)$',
                       color=rgb_colors[1])
        axs[1][0].plot(self.tlist * νfloquet, 0.5 * np.real([(h_time[it] * static_pauli["z"]).tr()
                                                             for it in range(self.time_points)]) / self.E01, '.-',
                       label=r'$h_z(t)$',
                       color=rgb_colors[2])
        axs[1][0].set_xlim([0, 1])
        axs[1][0].set_xlabel('time/T', fontsize=15)
        axs[1][0].set_ylabel(r'$h_a(t)/E_{01}$', fontsize=15)
        axs[1][0].legend()

        axs[0][1].plot(rate_data / self.decoherence_rate([ε01s, νfloquet, 0, 0, 0]), '.-');
        axs[0][1].set_xlabel('iteration', fontsize=15)
        axs[0][1].set_ylabel(r'$\Gamma_2/\Gamma_{2,static}$', fontsize=15)

        axs[1][1].plot(h0match_data / self.hstatic_matching([ε01s, νfloquet, 0, 0, 0]), '.-');
        axs[1][1].set_xlabel('iteration', fontsize=15)
        axs[1][1].set_ylabel(r'$(h0 matching)/(h0 matching)_{static}$', fontsize=15)

        axs[2][1].imshow(np.abs(list(map(self.complexify, full_data)))[:, 0].T, aspect='auto', cmap='plasma',
                         origin='lower')
        axs[2][1].set_title(r'$\vert\widetilde{φ}(m)\vert$')
        axs[2][1].set_xlabel('iteration', fontsize=15)
        axs[2][1].set_ylabel(r'$\omega/ω_{floquet}$', fontsize=15)

        axs[3][1].imshow(np.abs(list(map(self.complexify, full_data)))[:, 1].T, aspect='auto', cmap='plasma',
                         origin='lower')
        axs[3][1].set_title(r'$\vert\widetilde{θ}(m)\vert$')
        axs[3][1].set_xlabel('iteration', fontsize=15)
        axs[3][1].set_ylabel(r'$\omega/ω_{floquet}$', fontsize=15)

        axs[4][1].imshow(np.abs(list(map(self.complexify, full_data)))[:, 2].T, aspect='auto',
                         cmap='plasma',
                         origin='lower')
        axs[4][1].set_title(r'$\vert\widetilde{β}(m)\vert$ ')
        axs[4][1].set_xlabel('iteration', fontsize=15)
        axs[4][1].set_ylabel(r'$\omega/ω_{floquet}$', fontsize=15)

        [axs[r][s].grid(True, color='gray', linestyle='--', linewidth=0.5) for r in range(nrows) for s in
         range(ncols)]
        [axs[r][s].tick_params(axis='both', which='both', labelsize=15) for r in range(nrows) for s in range(ncols)]
        plt.subplots_adjust(top=1.5, hspace=0.5, wspace=0.25)