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

def create_circuit_from_file(filename: str) -> QuantumCircuit:
    '''
    Reads a file that contains a 2D array, where each index represents an operation on an initial state.
    e.g. [[H,0,0],[CNOT(0),0,0],[0,CNOT(0,1),0],[0,0,CNOT(2)]] would create a circuit with 4 qubits.
    The hadamard acts on qbit zero in the first operation, while in the 2nd positon there is a CNOT gate from qbut 1 acting on qbit 0,
    then a cnot from qbit 2 acting on qbit 1, and then a cnot from qbit 3 acting on qbit 2 in the last operation.
    '''
    

def create_quantum_circuit() -> QuantumCircuit:
    """Create a simple quantum circuit."""
    # System Specification
    n_qubits = 4
    circ = QuantumCircuit(n_qubits)
    
    # Test Circuit
    circ.h(0)
    for qubit in range(n_qubits - 1):
        circ.cx(qubit, qubit + 1)
    circ.measure_all()
    return circ

def create_random_noise_model(circuit: QuantumCircuit = None, p_reset: float=0.0, p_meas: float = 0.0, 
                              gate_error_prob: float = 0.0) -> tuple[NoiseModel, dict]:
    """Create a random noise model based on a quantum circuit.
    
    Args:
        circuit: Quantum circuit to analyze for gate types
        p_reset: Reset error probability (random if 0.0)
        p_meas: Measurement error probability (random if 0.0)
        gate_error_prob: Error probability for all gates (random per gate if 0.0)
        
    Returns:
        tuple: (noise_model, parameters_dict)
    """
    # Generate random noise parameters if none provided
    if p_reset == 0.0:
        p_reset = random.uniform(0.01, 0.3)
    if p_meas == 0.0:
        p_meas = random.uniform(0.01, 0.2)
    
    # Create QuantumError objects for reset and measurement
    error_reset = pauli_error([("X", p_reset), ("I", 1 - p_reset)])
    error_meas = pauli_error([("X", p_meas), ("I", 1 - p_meas)])
    
    # Initialize noise model and parameters dictionary
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_reset, ["reset"])
    noise_model.add_all_qubit_quantum_error(error_meas, ["measure"])
    
    params = {
        "p_reset": p_reset,
        "p_meas": p_meas,
    }
    
    # If a circuit is provided, create errors for each gate type
    if circuit:
        # Get unique operation names from the circuit
        gate_types = set(inst.operation.name for inst in circuit.data)
        
        for gate_type in gate_types:
            # Skip measurement operations as they're handled separately
            if gate_type in ['measure', 'reset', 'barrier']:
                continue
                
            # Generate random error probability for this gate type if not specified
            p_gate = gate_error_prob if gate_error_prob > 0 else random.uniform(0.01, 0.2)
            params[f"p_{gate_type}"] = p_gate
            
            # Create appropriate error based on gate type
            # Determine if this is a multi-qubit gate based on the first occurrence
            for inst in circuit.data:
                if inst.operation.name == gate_type:
                    num_qubits = len(inst.qubits)
                    break
            
            # Create the quantum error
            gate_error = depolarizing_error(p_gate, num_qubits)
            noise_model.add_all_qubit_quantum_error(gate_error, [gate_type])
    
    # Print the noise model
    print("Noise model:")
    print(noise_model)
    print("Noise parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")
    
    return noise_model, params

def add_circuit_noise(circuit: QuantumCircuit) -> tuple[QuantumCircuit, NoiseModel, dict]:
    """
    Add random noise to a quantum circuit based on its gate composition.
    
    Returns:
        tuple: (noisy_circuit, noise_model, noise_parameters)
    """
    # Create a random noise model using the circuit
    noise_model, noise_params = create_random_noise_model(circuit=circuit)
    
    print(f"Noise parameters: {noise_params}")

    # Create noisy simulator backend
    sim_noise = AerSimulator(noise_model=noise_model)
    
    # Transpile circuit for noisy basis gates
    passmanager = generate_preset_pass_manager(
        optimization_level=3, backend=sim_noise
    )
    noisy_circuit = passmanager.run(circuit)
    
    return noisy_circuit, noise_model, noise_params

def run_circuit(circuit: QuantumCircuit, noise_model: NoiseModel = None, filename_prefix: str = "") -> dict:
    """
    Run the quantum circuit with or without noise.
    
    Args:
        circuit: The quantum circuit to run
        noise_model: Optional noise model to apply
        filename_prefix: Prefix for the output histogram filename
    
    Returns:
        dict: The measurement counts
    """
    # Create a simulator backend with optional noise model
    backend = AerSimulator(noise_model=noise_model)
    
    # Transpile circuit
    passmanager = generate_preset_pass_manager(
        optimization_level=3, backend=backend
    )
    circ_t = passmanager.run(circuit)
    
    # Run and get counts
    result = backend.run(circ_t).result()
    counts = result.get_counts(0)
    
    # Plot output with appropriate filename
    output_filename = f"{filename_prefix}histogram.png"
    plot_histogram(counts).savefig(output_filename)
    print(f"Histogram saved as {output_filename}")

    return counts

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
