import numpy as np
import random
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



'''
TODO:
- fully implement create_circuit_from_file
- test full noise model implementation
- generate a dataset of circuits with noise, inputs, outputs, and expected outputs
- train a neural network on the dataset given the circuit, inputs, and outputs (find the noise parameters)
- test the neural network on a new set of circuits with unknown noise parameters
- implement a function to generate a dataset of circuits with noise, inputs, outputs, and expected outputs
- add a "noise reversal" function (or maybe just the neural network can learn directly) to attempt to mitigate errors
- determine cutoff for noise parameters in which the neural network can no longer learn or be effective



https://postquantum.com/quantum-computing/quantum-error-correction/

Probabilistic error cancellation (PEC)
'''

if __name__ == "__main__":
    # Create a quantum circuit
    circuit = create_quantum_circuit()
    print("Created quantum circuit")
    
    # Run the circuit without noise
    clean_counts = run_circuit(circuit, filename_prefix="clean_")
    print("Clean circuit execution complete")
    
    # Add noise to the circuit
    noisy_circuit, noise_model, noise_params = add_circuit_noise(circuit)
    print("Added noise to circuit")
    
    # Run the noisy circuit
    noisy_counts = run_circuit(noisy_circuit, noise_model, filename_prefix="noisy_")
    print("Noisy circuit execution complete")
