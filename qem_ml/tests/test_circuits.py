import os
import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel

from qem_ml.circuits.builder import (
    load_circuit_from_file,
    save_circuit_to_file,
    create_random_circuit
)
from qem_ml.circuits.simulator import Simulator


# Define test directory and ensure it exists
TEST_DIR = os.path.join(os.path.dirname(__file__), 'data', 'circuit_data')
os.makedirs(TEST_DIR, exist_ok=True)

# Define fixture for a simple quantum circuit
@pytest.fixture
def simple_circuit():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

# Define fixture for a test file path
@pytest.fixture
def test_qasm_path():
    return os.path.join(TEST_DIR, 'test_circuit.qasm')

# Define fixture for a Simulator instance
@pytest.fixture
def simulator(simple_circuit):
    return Simulator(simple_circuit)

# Clean up test files after tests
@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Clean up any test files
    for file in os.listdir(TEST_DIR):
        file_path = os.path.join(TEST_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


class TestCircuitBuilder:
    """Test the circuit builder functionality."""
    
    def test_create_random_circuit(self):
        """Test creating random circuits with different parameters."""
        # Test with default parameters
        circ1 = create_random_circuit()
        assert circ1.num_qubits == 1
        
        # Test with custom parameters
        num_qubits, depth = 3, 5
        circ2 = create_random_circuit(num_qubits=num_qubits, depth=depth)
        assert circ2.num_qubits == num_qubits
        # Depth may not exactly match requested depth due to how random_circuit works
        
        # Test that random circuits are different
        circ3 = create_random_circuit(num_qubits=num_qubits, depth=depth)
        assert circ2.data != circ3.data  # Circuit data should be different
    
    def test_save_and_load_circuit(self, simple_circuit, test_qasm_path):
        """Test saving and loading circuits to/from QASM files."""
        # Save circuit to file
        save_circuit_to_file(simple_circuit, test_qasm_path)
        assert os.path.exists(test_qasm_path)
        
        # Load circuit from file
        loaded_circuit = load_circuit_from_file(test_qasm_path)
        
        # Check number of qubits and operations
        assert loaded_circuit.num_qubits == simple_circuit.num_qubits
        assert len(loaded_circuit.data) == len(simple_circuit.data)
        
        # Check that gate types match
        original_ops = [inst.operation.name for inst in simple_circuit.data]
        loaded_ops = [inst.operation.name for inst in loaded_circuit.data]
        assert original_ops == loaded_ops
    
    def test_load_circuit_from_nonexistent_file(self):
        """Test that loading from a nonexistent file raises an error."""
        nonexistent_path = os.path.join(TEST_DIR, 'nonexistent.qasm')
        with pytest.raises(FileNotFoundError):
            load_circuit_from_file(nonexistent_path)


class TestSimulator:
    """Test the quantum circuit simulator functionality."""
    
    def test_simulator_initialization(self, simple_circuit):
        """Test simulator initialization with different parameters."""
        # Test with circuit
        sim1 = Simulator(simple_circuit)
        assert sim1.circuit == simple_circuit
        assert sim1.noise_model is None
        
        # Test with empty initialization
        sim2 = Simulator()
        assert sim2.circuit is None
        assert sim2.noise_model is None
        
        # Test with noise model
        noise_model = NoiseModel()
        sim3 = Simulator(simple_circuit, noise_model)
        assert sim3.circuit == simple_circuit
        assert sim3.noise_model == noise_model
    
    def test_run_clean(self, simulator):
        """Test running a clean simulation."""
        counts = simulator.run_clean(filename_prefix="test_clean_", generate_histogram=False)
        
        # Check that we got some counts
        assert len(counts) > 0

    
    def test_run_without_circuit(self):
        """Test that running without a circuit raises an error."""
        empty_sim = Simulator()
        with pytest.raises(ValueError, match="No circuit provided for simulation"):
            empty_sim.run_clean(generate_histogram=False)
    
    def test_add_noise(self, simulator):
        """Test adding noise to a circuit."""
        noisy_circuit, noise_model, params = simulator.add_noise()
        
        # Check that noise components were created
        assert noisy_circuit is not None
        assert noise_model is not None
        assert len(params) > 0
        
        # Check that parameters contain reset and measurement errors
        assert 'p_reset' in params
        assert 'p_meas' in params
        
        # Check that per-gate errors were added for gates in the circuit
        assert 'p_h' in params  # For Hadamard gate
        assert 'p_cx' in params  # For CNOT gate
    
    def test_add_noise_with_specific_probabilities(self, simulator):
        """Test adding noise with specific error probabilities."""
        p_reset = 0.05
        p_meas = 0.1
        gate_error = 0.15
        
        # Fix: Changed from float to dictionary
        _, _, params = simulator.add_noise(
            p_reset=p_reset, 
            p_meas=p_meas, 
            gate_error_probs={'h': gate_error, 'cx': gate_error}  # Dictionary mapping gate names to probabilities
        )
        
        # Check that specified probabilities were used
        assert params['p_reset'] == p_reset
        assert params['p_meas'] == p_meas
        assert params['p_h'] == gate_error  # All gates should use gate_error_probs
        assert params['p_cx'] == gate_error
    
    def test_run_noisy(self, simulator):
        """Test running a noisy simulation."""
        # Add noise first - Fix: Changed from float to dictionary
        simulator.add_noise(p_reset=0.1, p_meas=0.1, gate_error_probs={'h': 0.1, 'cx': 0.1})
        
        # Run noisy simulation
        counts = simulator.run_noisy(filename_prefix="test_noisy_", generate_histogram=False)
        
        # Check that we got some counts
        assert len(counts) > 0
    
    
    def test_run_noisy_without_explicit_noise(self, simulator):
        """Test that run_noisy adds noise if none is specified."""
        # Run noisy simulation without explicitly adding noise first
        counts = simulator.run_noisy(filename_prefix="test_auto_noisy_", generate_histogram=False)
        
        # Check that noise was automatically added
        assert simulator.noise_model is not None
        assert simulator.noisy_circuit is not None
        
        # Check that we got counts
        assert len(counts) > 0


# Test for circuits from data directory
class TestExternalCircuits:
    """Test using external circuit files from the data directory."""
    
    def test_load_data_circuits(self):
        """Test loading circuits from the data directory."""
        # Define path to data directory
        data_dir = TEST_DIR
        
        # Create a test file if none exists
        if not any(f.endswith('.qasm') for f in os.listdir(data_dir)):
            test_circuit = QuantumCircuit(2, 2)
            test_circuit.h(0)
            test_circuit.cx(0, 1)
            test_circuit.measure_all()
            save_circuit_to_file(test_circuit, os.path.join(data_dir, 'test_circuit.qasm'))
        
        # Get list of QASM files
        qasm_files = [f for f in os.listdir(data_dir) if f.endswith('.qasm')]
        assert len(qasm_files) > 0, f"No QASM files found in {data_dir}"
        
        # Test first QASM file
        test_file = os.path.join(data_dir, qasm_files[0])
        circuit = load_circuit_from_file(test_file)
        
        # Create simulator with loaded circuit
        sim = Simulator(circuit)
        
        # Run clean and noisy simulations
        clean_counts = sim.run_clean(filename_prefix=f"data_clean_{qasm_files[0]}_", generate_histogram=False)
        noisy_counts = sim.run_noisy(filename_prefix=f"data_noisy_{qasm_files[0]}_", generate_histogram=False)
        
        # Verify results
        assert len(clean_counts) > 0
        assert len(noisy_counts) > 0