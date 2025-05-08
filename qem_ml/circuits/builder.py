import os
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit.random import random_circuit

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
