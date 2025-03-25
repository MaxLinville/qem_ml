import numpy as np
from sklearn.neural_network import MLPClassifier
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, MCXGate
from qiskit.quantum_info import Kraus, SuperOp
from qiskit.visualization import plot_histogram
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)

"""Provide the primary functions."""

def create_neural_network():
    """Create a simple neural network."""
    X = [[0., 0.], [1., 1.]]
    y = [0, 1]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1, activation='relu')

    clf.fit(X, y)
    print(clf.predict([[2., 2.], [-1., -2.]]))
    print([coef.shape for coef in clf.coefs_])
    print(clf.predict_proba([[2., 2.], [1., 2.]]))

def create_quantum_circuit():
    """Create a simple quantum circuit."""
    # System Specification
    n_qubits = 4
    circ = QuantumCircuit(n_qubits)
    
    # Test Circuit
    circ.h(0)
    for qubit in range(n_qubits - 1):
        circ.cx(qubit, qubit + 1)
    circ.measure_all()
    print(f"Circuit: \n{circ}")

    #ideal simulation
    # Ideal simulator and execution
    sim_ideal = AerSimulator()
    result_ideal = sim_ideal.run(circ).result()
    plot_histogram(result_ideal.get_counts(0)).savefig("./qem_ml/data/ideal_histogram.png")

    # basic model
    # Example error probabilities
    p_reset = 0.3
    p_meas = 0.1
    p_gate1 = 0.5
    
    # QuantumError objects
    error_reset = pauli_error([("X", p_reset), ("I", 1 - p_reset)])
    error_meas = pauli_error([("X", p_meas), ("I", 1 - p_meas)])
    error_gate1 = pauli_error([("X", p_gate1), ("I", 1 - p_gate1)])
    error_gate2 = error_gate1.tensor(error_gate1)
    
    # Add errors to noise model
    noise_bit_flip = NoiseModel()
    noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
    noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
    noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
    noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])
    
    print(f"noise bit flip: {noise_bit_flip}")

    #execute noisy sim:

    # Create noisy simulator backend
    sim_noise = AerSimulator(noise_model=noise_bit_flip)
    
    # Transpile circuit for noisy basis gates
    passmanager = generate_preset_pass_manager(
        optimization_level=3, backend=sim_noise
    )
    circ_tnoise = passmanager.run(circ)
    
    # Run and get counts
    result_bit_flip = sim_noise.run(circ_tnoise).result()
    counts_bit_flip = result_bit_flip.get_counts(0)
    
    # Plot noisy output
    plot_histogram(counts_bit_flip).savefig("./qem_ml/data/noisy_histogram.png")


if __name__ == "__main__":
    # create_neural_network()
    create_quantum_circuit()
