import json
import random

from qiskit import QuantumCircuit
from qiskit.circuit.library import *
from qiskit.quantum_info import Kraus, SuperOp
from qiskit.visualization import plot_histogram
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit.circuit.random import random_circuit
from qiskit import qasm3
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)

def load_circuit_from_file(file_path: str) -> QuantumCircuit:
    '''
    Load a quantum circuit from a QASM file.
    '''
    return qasm3.load(file_path)

def save_circuit_to_file(circuit: QuantumCircuit, file_path: str) -> None:
    with open(file_path, 'w') as f:
        qasm_code = qasm3.dumps(circuit)
        f.write(qasm_code)

def create_random_circuit(num_qubits: int = 1, depth: int = 1) -> QuantumCircuit:
    """
    Create a random quantum circuit
    Args:
        num_qubits: Number of qubits in the circuit
        depth: Depth of the circuit (number of layers)
    Returns:
        QuantumCircuit: A random quantum circuit
    """
    circuit = random_circuit(num_qubits=num_qubits, depth=depth)
    return circuit

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

def main():
    # create_random_circuit()
    #load the random_circuit.qasm file and plot
    circuit = load_circuit_from_file("random_circuit.qasm")
    print("Loaded random circuit from file")
    print(circuit)
    # add noise model
    noisy_circuit, noise_model, noise_params = add_circuit_noise(circuit)
    print("Added noise to circuit")

if __name__ == '__main__':
    main()