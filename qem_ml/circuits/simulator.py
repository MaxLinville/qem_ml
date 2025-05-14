import random
from typing import Dict, Optional, Tuple, List, Union
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    amplitude_damping_error,
    phase_damping_error,
    thermal_relaxation_error,
    kraus_error,
)

class Simulator:
    """
    A quantum circuit simulator that can run circuits with or without noise models.
    
    This class encapsulates functionality for running quantum circuits with clean
    or noisy simulation, and for adding noise to circuits.
    
    Attributes:
        circuit (QuantumCircuit): The original quantum circuit
        noise_model (NoiseModel): The noise model to apply during simulation
        noisy_circuit (QuantumCircuit): The circuit with noise applied
        noise_params (Dict): Parameters of the applied noise model
    """
    
    # Define error types as constants
    PAULI_X = 'pauli_x'
    PAULI_Z = 'pauli_z' 
    DEPOLARIZING = 'depolarizing'
    AMPLITUDE_DAMPING = 'amplitude_damping'
    PHASE_DAMPING = 'phase_damping'
    THERMAL_RELAXATION = 'thermal_relaxation'
    
    def __init__(self, circuit: QuantumCircuit = None, 
                 noise_model: NoiseModel = None):
        """
        Initialize the simulator with an optional circuit and noise model.
        
        Args:
            circuit: Quantum circuit to simulate
            noise_model: Optional noise model to apply during simulation
        """
        self.circuit: QuantumCircuit = circuit
        self.noise_model: NoiseModel = noise_model
        self.noisy_circuit: QuantumCircuit = None
        self.noise_params: dict = {}
    
    def run_clean(self, filename_prefix: str = "clean_",  generate_histogram: bool = True) -> Dict:
        """
        Run the quantum circuit without noise.
        
        Args:
            filename_prefix: Prefix for the output histogram filename
        
        Returns:
            dict: The measurement counts
        """
        if self.circuit is None:
            raise ValueError("No circuit provided for simulation")
        
        # Create a simulator backend without noise
        backend = AerSimulator()
        
        # Transpile circuit
        passmanager = generate_preset_pass_manager(
            optimization_level=3, backend=backend
        )
        circ_t = passmanager.run(self.circuit)
        
        # Run and get counts
        result = backend.run(circ_t).result()
        counts = result.get_counts(0)
        
        # Plot output with appropriate filename
        if generate_histogram:
            output_filename = f"{filename_prefix}histogram.png"
            plot_histogram(counts).savefig(output_filename)
            print(f"Histogram saved as {output_filename}")

        return counts
    
    def run_noisy(self, filename_prefix: str = "noisy_", generate_histogram: bool = True, draw_circuit: bool = False) -> Dict:
        """
        Run the quantum circuit with noise.
        
        If no noise model has been set and no noisy circuit exists, 
        this will generate them using add_noise().
        
        Args:
            filename_prefix: Prefix for the output histogram filename
        
        Returns:
            dict: The measurement counts
        """
        if self.circuit is None:
            raise ValueError("No circuit provided for simulation")
        
        # If no noise model or noisy circuit, create them
        if self.noise_model is None or self.noisy_circuit is None:
            self.add_noise()
        
        if draw_circuit:
            # Draw the noisy circuit
            self.noisy_circuit.draw(output='mpl', filename=f"{filename_prefix}noisy_circuit.png")
            print(f"Noisy circuit saved as {filename_prefix}noisy_circuit.png")
        # Create a simulator backend with noise model
        backend = AerSimulator(noise_model=self.noise_model)
        
        # Transpile circuit
        passmanager = generate_preset_pass_manager(
            optimization_level=3, backend=backend
        )
        circ_t = passmanager.run(self.noisy_circuit)
        
        # Run and get counts
        result = backend.run(circ_t).result()
        counts = result.get_counts(0)
        
        # Plot output with appropriate filename
        if generate_histogram:
            output_filename = f"{filename_prefix}histogram.png"
            plot_histogram(counts).savefig(output_filename)
            print(f"Histogram saved as {output_filename}")

        return counts
    
    def reset_noise_model(self):
        """Reset the noise model to None"""
        self.noise_model = None
        self.noise_params = {}
        self.noisy_circuit = None

    def add_noise(
        self, 
        error_types: List[str] = None,
        p_reset: float = 0.0,
        p_meas: float = 0.0,
        gate_error_probs: Dict[str, float] = None,
        t1: float = 50.0,  # T1 relaxation time (μs)
        t2: float = 70.0,  # T2 relaxation time (μs)
        gate_time: float = 0.1,  # Gate time in μs
    ) -> Tuple[QuantumCircuit, NoiseModel, Dict]:
        """
        Add various types of noise to the quantum circuit.
        
        Args:
            error_types: List of error types to include. Options are:
                'pauli_x', 'pauli_z', 'depolarizing', 'amplitude_damping',
                'phase_damping', 'thermal_relaxation'.
                If None, defaults to ['pauli_x', 'depolarizing']
            p_reset: Reset error probability (random if 0.0)
            p_meas: Measurement error probability (random if 0.0)
            gate_error_probs: Dict mapping gate names to error probabilities
                (random per gate if None or empty)
            t1: T1 relaxation time in microseconds (for thermal relaxation)
            t2: T2 relaxation time in microseconds (for thermal relaxation)
            gate_time: Gate execution time in microseconds (for thermal relaxation)
        
        Returns:
            tuple: (noisy_circuit, noise_model, noise_parameters)
        """
        if self.circuit is None:
            raise ValueError("No circuit provided to add noise to")
        
        # Default to basic error types if none specified
        if error_types is None:
            error_types = [self.PAULI_X, self.DEPOLARIZING]
        
        # Create noise model and parameters
        noise_model, params = self._create_noise_model(
            error_types=error_types,
            p_reset=p_reset,
            p_meas=p_meas,
            gate_error_probs=gate_error_probs,
            t1=t1,
            t2=t2,
            gate_time=gate_time,
        )
        self.noise_model = noise_model
        self.noise_params = params
        
        # Create noisy simulator backend
        sim_noise = AerSimulator(noise_model=self.noise_model)
        
        # Transpile circuit for noisy basis gates
        passmanager = generate_preset_pass_manager(
            optimization_level=3, backend=sim_noise
        )
        self.noisy_circuit = passmanager.run(self.circuit)
        
        return self.noisy_circuit, self.noise_model, self.noise_params
    
    def _create_noise_model(
        self,
        error_types: List[str],
        p_reset: float = 0.0,
        p_meas: float = 0.0,
        gate_error_probs: Dict[str, float] = None,
        t1: float = 50.0,
        t2: float = 70.0,
        gate_time: float = 0.1,
    ) -> Tuple[NoiseModel, Dict]:
        """
        Create a complex noise model based on the quantum circuit.
        
        Args:
            error_types: Types of errors to include
            p_reset: Reset error probability (random if 0.0)
            p_meas: Measurement error probability (random if 0.0)
            gate_error_probs: Dict mapping gate names to error probabilities
            t1: T1 relaxation time in microseconds
            t2: T2 relaxation time in microseconds
            gate_time: Gate execution time in microseconds
        
        Returns:
            tuple: (noise_model, parameters_dict)
        """
        # Generate random noise parameters if none provided
        if p_reset == 0.0:
            p_reset = random.uniform(0.01, 0.2)
        if p_meas == 0.0:
            p_meas = random.uniform(0.01, 0.15)
        
        # Initialize gate error probabilities dict if not provided
        if gate_error_probs is None:
            gate_error_probs = {}
        
        # Initialize noise model and parameters dictionary
        noise_model = NoiseModel()
        params = {
            "p_reset": p_reset,
            "p_meas": p_meas,
            "error_types": error_types,
        }
        
        if self.THERMAL_RELAXATION in error_types:
            params["t1"] = t1
            params["t2"] = t2
            params["gate_time"] = gate_time
        
        # Add reset and measurement errors
        if self.PAULI_X in error_types:
            error_reset_x = pauli_error([("X", p_reset), ("I", 1 - p_reset)])
            error_meas_x = pauli_error([("X", p_meas), ("I", 1 - p_meas)])
            noise_model.add_all_qubit_quantum_error(error_reset_x, ["reset"])
            noise_model.add_all_qubit_quantum_error(error_meas_x, ["measure"])
        
        if self.PAULI_Z in error_types:
            p_reset_z = p_reset
            p_meas_z = p_meas
            params["p_reset_z"] = p_reset_z
            params["p_meas_z"] = p_meas_z
            
            error_reset_z = pauli_error([("Z", p_reset_z), ("I", 1 - p_reset_z)])
            error_meas_z = pauli_error([("Z", p_meas_z), ("I", 1 - p_meas_z)])
            noise_model.add_all_qubit_quantum_error(error_reset_z, ["reset"])
            noise_model.add_all_qubit_quantum_error(error_meas_z, ["measure"])
        
        # If no circuit is provided, we can't add gate errors
        if not self.circuit:
            return noise_model, params
        
        # Get unique operation names from the circuit
        gate_types = set(inst.operation.name for inst in self.circuit.data 
                         if inst.operation.name not in ['measure', 'reset', 'barrier'])
        
        # For each gate type in the circuit, add appropriate errors
        for gate_type in gate_types:
            # Determine qubit count for this gate
            num_qubits = 0
            for inst in self.circuit.data:
                if inst.operation.name == gate_type:
                    num_qubits = len(inst.qubits)
                    break
            
            # Get or generate error probability for this gate
            p_gate = gate_error_probs.get(gate_type, 0.0)
            if p_gate == 0.0:
                p_gate = random.uniform(0.005, 0.03) # Default gate error range
            
            params[f"p_{gate_type}"] = p_gate
            
            # Add appropriate error models for this gate
            gate_errors = []
            
            if self.DEPOLARIZING in error_types:
                gate_errors.append(depolarizing_error(p_gate, num_qubits))
            
            if self.PAULI_X in error_types:
                # For multi-qubit gates, apply X error to each qubit with probability p/n
                x_prob = p_gate / num_qubits if num_qubits > 1 else p_gate * 0.5
                if num_qubits == 1:
                    gate_errors.append(pauli_error([("X", x_prob), ("I", 1 - x_prob)]))
                else:
                    # For multi-qubit gates we'd need to create appropriate Pauli strings
                    # This is simplified for demonstration
                    x_error = pauli_error([("X" + "I" * (num_qubits-1), x_prob), 
                                          ("I" * num_qubits, 1 - x_prob)])
                    gate_errors.append(x_error)
            
            if self.PAULI_Z in error_types:
                # Similar approach for Z errors
                z_prob = p_gate / num_qubits if num_qubits > 1 else p_gate * 0.7
                if num_qubits == 1:
                    gate_errors.append(pauli_error([("Z", z_prob), ("I", 1 - z_prob)]))
                else:
                    z_error = pauli_error([("Z" + "I" * (num_qubits-1), z_prob), 
                                          ("I" * num_qubits, 1 - z_prob)])
                    gate_errors.append(z_error)
            
            if self.AMPLITUDE_DAMPING in error_types:
                # Amplitude damping (T1-like relaxation)
                damp_prob = p_gate * 0.8  # Scale factor for amplitude damping
                params[f"p_{gate_type}_damping"] = damp_prob
                gate_errors.append(amplitude_damping_error(damp_prob))
            
            if self.PHASE_DAMPING in error_types:
                # Phase damping (T2-like dephasing)
                phase_prob = p_gate * 0.6  # Scale factor for phase damping
                params[f"p_{gate_type}_phase"] = phase_prob
                gate_errors.append(phase_damping_error(phase_prob))
            
            if self.THERMAL_RELAXATION in error_types:
                # Thermal relaxation based on T1, T2, and gate time
                # We create this per gate type as the gate time might differ
                gate_errors.append(thermal_relaxation_error(t1, t2, gate_time))
            
            # Apply each error type separately instead of combining them
            if len(gate_errors) == 1:
                # Single error type is straightforward
                noise_model.add_all_qubit_quantum_error(gate_errors[0], [gate_type], warnings=False)
            elif len(gate_errors) > 1:
                # For multiple error types, apply each one with its own probability
                for error in gate_errors:
                    # Add each error individually (will be applied sequentially)
                    noise_model.add_all_qubit_quantum_error(error, [gate_type], warnings=False)
            else:
                continue  # No errors to add
                    
        return noise_model, params