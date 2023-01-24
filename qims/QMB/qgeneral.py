import qutip as qt
import matplotlib.pyplot as plt



class GeneralizedSimulation:
    def __init__(self, N, couplings):
        """
        Initializes the generalized simulation

        Parameters:
        N (int) : number of qubits in the system
        couplings (list) : list of tuple of the form (i, j, J_zz, J_xx, J_yy) representing the coupling
                           between qubit i and j with strengths J_zz, J_xx, J_yy for szsz, sxsx and sysy respectively
        """
        self.N = N
        self.couplings = couplings
        self.H = self.generalized_model()
        self.energies = None
        self.states = None

    def pauli_matrix(self, position, pauli_type):
        """
        Returns the Pauli matrix of the given type for the qubit of given position in N qubits hilbert space

        Parameters:
        position (int) : position of qubit that the Pauli matrix is required for, 0-indexed
        pauli_type : the Pauli operator, for example qt.sigmax()

        Returns:
        qutip.Qobj : the Pauli matrix of the given type for the qubit of given position in N qubits hilbert space
        """
        identity_list = [qt.qeye(2)] * (position) + [pauli_type] + [qt.qeye(2)] * (self.N - 1 - position)
        return qt.tensor(identity_list)

    def generalized_model(self):
        """
        Returns the Hamiltonian of the generalized model

        Returns:
        qutip.Qobj : the Hamiltonian of the generalized model
        """
        H = 0
        for i, j, J_zz, J_xx, J_yy in self.couplings:
            H += J_zz / 2 * (self.pauli_matrix(i, qt.sigmaz()) * self.pauli_matrix(j, qt.sigmaz()))
            H += J_xx / 2 * (self.pauli_matrix(i, qt.sigmax()) * self.pauli_matrix(j, qt.sigmax()))
            H += J_yy / 2 * (self.pauli_matrix(i, qt.sigmay()) * self.pauli_matrix(j, qt.sigmay()))
        return H

    def diagonalize(self):
        """
        Diagonalize the Hamiltonian to produce eigenenergies and eigenvectors

        Returns:
        tuple : a tuple of the form (energies, states) where energies is a 1D array of the eigenenergies,
                and states is a list of the eigenstates
        """
        self.energies, self.states = self.H.eigenstates()
        return self.energies, self.states

    def plot_energies(self, title="Eigenenergies", xlabel="State", ylabel="Energy"):
        """
        Plot the eigenenergies of the Hamiltonian

        Parameters:
        title (str): title of the plot
        xlabel (str): label for the x-axis
        ylabel (str): label for the y-axis
        """
        if self.energies is None:
            self.diagonalize()
        plt.plot(self.energies, 'bo')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def time_evolve(self, initial_state, times, evo_type='unitary', args=None):
        """
        Time evolve the given initial state using the Hamiltonian

        Parameters:
        initial_state (qutip.Qobj): the initial state
        times (list or array): times at which the state is to be evaluated
        evo_type (str): type of evolution. 'unitary' for unitary evolution, 'measurement' for measurement-induced non-unitary evolution.
        args (list): additional arguments for sesolve, required for evo_type='measurement'

        Returns:
        qutip.Qobj: the evolved state
        """
        if evo_type == 'unitary':
            return qt.sesolve(self.H, initial_state, times)
        elif evo_type == 'measurement':
            return qt.sesolve(self.H, initial_state, times, args=args)
        else:
            raise ValueError("Invalid evo_type. Choose 'unitary' or 'measurement'")

    def plot_basis_probabilities(self, psi0, times):
        """
        Plots the probabilities of each basis state over time

        Parameters:
        psi0 (qutip.Qobj) : initial state of the system
        times (list) : list of times to compute the probabilities at

        Returns:
        None
        """
        result = qt.mesolve(self.H, psi0, times)

        for i in range(2 ** self.N):
            prob = [abs(result.states[j].overlap(qt.basis(2 ** self.N, i))) ** 2 for j in range(len(times))]
            plt.plot(times, prob, label=f"Basis state {i}")
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.legend()
        plt.show()