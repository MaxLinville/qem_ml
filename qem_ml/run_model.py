import os
import numpy as np
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from qem_ml.circuits.simulator import Simulator
from qem_ml.models.qe_neural_net import QuantumErrorNeuralNet
from qem_ml.functions import apply_error_mitigation
from qem_ml.circuits import load_circuit_from_file


def test_model_with_inputs(
    model_path: str,
    input_values: List[int],
    circuit: QuantumCircuit | None = None,
    circuit_file: str | None = None,
    error_rate: float = 0.05,
    shots: int = 8192,
    output_dir: str = "./test_results"
):
    """
    Test a trained error mitigation model with specific circuit inputs.
    
    Args:
        circuit_file: Path to the circuit file
        circuit: QuantumCircuit object (if not using a file)
        model_path: Path to the trained model file
        input_values: List of values (0 or 1) to initialize qubits
        error_rate: Error rate to apply during testing
        shots: Number of shots for the simulation
        output_dir: Directory to save outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load circuit
    if circuit_file:
        circuit = load_circuit_from_file(circuit_file)
    elif circuit:
        circuit = circuit
    else:
        raise ValueError("Either circuit_file or circuit must be provided.")
    
    # Apply input values
    circuit_with_inputs = circuit.copy()
    
    # Reset all qubits first
    circuit_with_inputs.reset(range(circuit.num_qubits))
    
    # Apply X gates to specified qubits
    for i, val in enumerate(input_values):
        if i >= circuit.num_qubits:
            break
        if val == 1:
            circuit_with_inputs.x(i)
    
    print(f"Testing with input state: {input_values}")
    
    # Create simulator
    simulator = Simulator(circuit_with_inputs)
    
    # Run clean simulation
    clean_counts = simulator.run_clean(
        filename_prefix=f"{output_dir}/clean_", 
        generate_histogram=True
    )
    
    # Add noise
    gate_error_probs = {
        'h': error_rate,
        'x': error_rate,
        'cx': error_rate,
        'ccx': error_rate * 2  # Multi-qubit gates typically have higher error
    }
    
    simulator.add_noise(
        p_reset=error_rate/2,
        p_meas=error_rate,
        gate_error_probs=gate_error_probs
    )
    
    # Run noisy simulation
    noisy_counts = simulator.run_noisy(
        filename_prefix=f"{output_dir}/noisy_", 
        generate_histogram=True
    )
    
    # Apply error mitigation
    mitigated_counts = apply_error_mitigation(
        circuit=circuit_with_inputs,
        counts=noisy_counts,
        model_path=model_path
    )

    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get all possible bitstrings
    all_bitstrings = sorted(set(list(clean_counts.keys()) + 
                             list(noisy_counts.keys()) + 
                             list(mitigated_counts.keys())))
    
    
    # Convert all bitstring keys to integers for consistent comparison
    def normalize_counts(counts_dict):
        normalized = {}
        for bitstr, count in counts_dict.items():
            # Remove any spaces first
            clean_bitstr = bitstr.replace(" ", "")
            # Convert bitstring to integer
            int_val = int(clean_bitstr, 2)
            normalized[int_val] = normalized.get(int_val, 0) + count
        return normalized
    
     # Normalize all count dictionaries
    clean_counts_norm = normalize_counts(clean_counts)
    noisy_counts_norm = normalize_counts(noisy_counts)
    mitigated_counts_norm = normalize_counts(mitigated_counts)
    
    # Get all possible integer values
    all_int_values = sorted(set(
        list(clean_counts_norm.keys()) + 
        list(noisy_counts_norm.keys()) + 
        list(mitigated_counts_norm.keys())
    ))

     # Use the full number of qubits for display (since neural net uses all qubits)
    num_qubits = circuit_with_inputs.num_qubits

    # Convert integers back to consistently formatted bitstrings for display
    display_bitstrings = [format(val, f'0{num_qubits}b') for val in all_int_values]
    
    x = np.arange(len(all_int_values))
    width = 0.25

    # Use the normalized counts for plotting
    clean_sum = sum(clean_counts_norm.values())
    noisy_sum = sum(noisy_counts_norm.values())
    mitigated_sum = sum(mitigated_counts_norm.values())
    
    # Prepare data for plotting using normalized counts
    clean_values = [clean_counts_norm.get(val, 0)/clean_sum for val in all_int_values]
    noisy_values = [noisy_counts_norm.get(val, 0)/noisy_sum for val in all_int_values]
    mitigated_values = [mitigated_counts_norm.get(val, 0)/mitigated_sum for val in all_int_values]
    
    # Create the bars
    ax.bar(x - width, clean_values, width, label='Ideal', alpha=0.7)
    ax.bar(x, noisy_values, width, label='Noisy', alpha=0.7)
    ax.bar(x + width, mitigated_values, width, label='Mitigated', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Bitstring')
    ax.set_ylabel('Probability')
    ax.set_title(f'Error Mitigation Results (Error Rate: {error_rate})')
    ax.set_xticks(x)
    ax.set_xticklabels(display_bitstrings, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_plot.png")
    
    plot_histogram(mitigated_counts).savefig(f"{output_dir}/mitigated_histogram.png")
    # Calculate metrics using the normalized distributions
    clean_dist = np.array(clean_values)
    noisy_dist = np.array(noisy_values)
    mitigated_dist = np.array(mitigated_values)
    
    # Mean squared error
    noisy_mse = np.mean((noisy_dist - clean_dist) ** 2)
    mitigated_mse = np.mean((mitigated_dist - clean_dist) ** 2)
    
    # Print results
    print("\nTest Results:")
    print(f"Noisy MSE:      {noisy_mse:.6f}")
    print(f"Mitigated MSE:  {mitigated_mse:.6f}")
    print(f"Improvement:    {(1 - mitigated_mse/noisy_mse)*100:.2f}%")
    
    # Save results
    with open(f"{output_dir}/test_results.txt", "w") as f:
        f.write(f"Test circuit: {circuit_file}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Input values: {input_values}\n")
        f.write(f"Error rate: {error_rate}\n\n")
        f.write(f"Noisy MSE: {noisy_mse:.6f}\n")
        f.write(f"Mitigated MSE: {mitigated_mse:.6f}\n")
        f.write(f"Improvement: {(1 - mitigated_mse/noisy_mse)*100:.2f}%\n")
    
    return {
        "clean_counts": clean_counts,
        "noisy_counts": noisy_counts,
        "mitigated_counts": mitigated_counts,
        "noisy_mse": noisy_mse,
        "mitigated_mse": mitigated_mse
    }


if __name__ == "__main__":
    ...
    