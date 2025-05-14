"""
Integration functions for quantum error mitigation using neural networks.

This module provides high-level functions that integrate the various components
of the QEM-ML package for end-to-end quantum error mitigation workflows.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qem_ml.circuits.simulator import Simulator
from qem_ml.models.qe_neural_net import QuantumErrorNeuralNet


def train_error_mitigation_model(
    circuit: QuantumCircuit,
    noisy_distributions: np.ndarray,
    ideal_distributions: np.ndarray,
    output_dir: str,
    test_noisy_distributions: Optional[np.ndarray] = None,
    test_ideal_distributions: Optional[np.ndarray] = None,
    hidden_layers: Tuple[int, ...] = None,
    learning_rate: float = 0.001,
    max_iter: int = 1000,
    model_name: str = "qem_model"
) -> Tuple[QuantumErrorNeuralNet, Dict]:
    """
    Train a quantum error mitigation model and evaluate its performance.
    
    Args:
        circuit: The quantum circuit for which to train the error mitigation model
        noisy_distributions: Array of noisy output distributions (training set)
        ideal_distributions: Array of corresponding ideal distributions (training set)
        output_dir: Directory to save outputs
        test_noisy_distributions: Optional array of test noisy distributions
        test_ideal_distributions: Optional array of corresponding test ideal distributions
        hidden_layers: Structure of hidden layers for neural network (default: scaled based on circuit size)
        learning_rate: Learning rate for the neural network
        max_iter: Maximum iterations for training
        model_name: Name for the saved model and results
    
    Returns:
        Tuple containing (trained_model, results_dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get number of qubits from the circuit
    num_qubits = circuit.num_qubits
    print(f"Creating error mitigation model for {num_qubits}-qubit circuit")
    
    # Create neural network model
    model = QuantumErrorNeuralNet.from_num_qubits(
        num_qubits=num_qubits,
        hidden_layer_sizes=hidden_layers,
        learning_rate_init=learning_rate,
        max_iter=max_iter
    )
    
    # Train model
    print("Training error mitigation model...")
    score = model.train(noisy_distributions, ideal_distributions)
    print(f"Training completed with score: {float(score):.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Collect results
    results = {
        "num_qubits": num_qubits,
        "training_score": score,
        "model_path": model_path,
    }
    
    # Evaluate on test data if provided
    if test_noisy_distributions is not None and test_ideal_distributions is not None:
        print("Evaluating model on test data...")
        test_predictions = model.predict(test_noisy_distributions)
        
        # Calculate MSE
        test_mse = np.mean((test_predictions - test_ideal_distributions) ** 2)
        results["test_mse"] = float(test_mse)
        print(f"Test MSE: {test_mse:.6f}")
        
        # Save predictions
        save_distribution_comparison(
            test_noisy_distributions,
            test_ideal_distributions,
            test_predictions,
            output_dir,
            f"{model_name}_test_results"
        )
    
    # Save results summary
    with open(os.path.join(output_dir, f"{model_name}_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results


def apply_error_mitigation(
    circuit: QuantumCircuit,
    counts: Dict[str, int],
    model_path: str
) -> Dict[str, int]:
    """
    Apply error mitigation to a circuit's measurement counts.
    
    Args:
        circuit: Quantum circuit that was executed
        counts: Measurement counts from the noisy circuit execution
        model_path: Path to the saved error mitigation model
    
    Returns:
        Dict containing the mitigated counts
    """
    # Load the model
    model = QuantumErrorNeuralNet.load_model(model_path)
    
    # Format counts to distribution
    num_qubits = circuit.num_qubits
    noisy_dist = QuantumErrorNeuralNet.format_counts_to_distribution(counts, num_qubits)
    
    # Apply error mitigation
    mitigated_dist = model.predict(noisy_dist)
    
    # Convert back to counts
    total_shots = sum(counts.values())
    mitigated_counts = QuantumErrorNeuralNet.distribution_to_counts(
        mitigated_dist[0], total_shots=total_shots
    )
    
    return mitigated_counts


def generate_training_data(
    circuit: QuantumCircuit,
    num_samples: int = 50,
    error_rates_range: Tuple[float, float] = (0.01, 0.25),
    verbose: bool= False,
    randomize_inputs: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data for error mitigation by simulating the circuit with different noise levels.
    
    Args:
        circuit: Quantum circuit to simulate
        num_samples: Number of noisy samples to generate
        error_rates_range: Range of error rates for noise simulation (min, max)
        verbose: If True, print additional information
        randomize_inputs: If True, randomize the input state of the circuit

    Returns:
        Tuple of (noisy_distributions, ideal_distributions)
    """
    # Get number of qubits from the circuit
    num_qubits = circuit.num_qubits
    
    # Generate noisy and ideal samples
    noisy_dists = []
    ideal_dists = []
    min_error, max_error = error_rates_range
    
    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}...")
        # If randomizing inputs, create a new circuit with random initialization
        if randomize_inputs:
            # Create a new empty circuit with same registers
            new_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
            
            # Add reset operations to the beginning
            new_circuit.reset(range(num_qubits))
            
            # Randomly apply X gates to initialize qubits
            input_state = []
            for qubit in range(num_qubits):
                if np.random.random() > 0.5:
                    new_circuit.x(qubit)
                    input_state.append(1)
                else:
                    input_state.append(0)
            
            # Now append the original circuit operations (excluding any initial resets)
            original_instructions = circuit.data
            for instruction in original_instructions:
                inst = instruction.operation
                qargs = instruction.qubits
                cargs = instruction.clbits
                # Skip any reset operations from the original circuit
                if inst.name != 'reset':
                    new_circuit.append(inst, qargs, cargs)
            
            circuit_copy = new_circuit
        else:
            circuit_copy = circuit

        # Create simulator
        simulator = Simulator(circuit_copy)
        
        # Get clean distribution (ground truth)
        clean_counts = simulator.run_clean(generate_histogram=False)
        
        # Fix bitstrings and get ideal distribution
        clean_counts_fixed = {}
        for bitstring, count in clean_counts.items():
            fixed_bitstring = bitstring.replace(" ", "")
            if len(fixed_bitstring) > num_qubits:
                fixed_bitstring = fixed_bitstring[-num_qubits:]
            elif len(fixed_bitstring) < num_qubits:
                fixed_bitstring = fixed_bitstring.zfill(num_qubits)
            clean_counts_fixed[fixed_bitstring] = count
        
        ideal_dist = QuantumErrorNeuralNet.format_counts_to_distribution(
            clean_counts_fixed, num_qubits=num_qubits
        )

        # Add to ideal distributions
        ideal_dists.append(ideal_dist)

        # Generate noisy samples
        gate_error_probs = {
            'h': np.random.uniform(min_error, max_error),
            'x': np.random.uniform(min_error, max_error),
            'cx': np.random.uniform(min_error, max_error),
            'ccx': np.random.uniform(min_error, max_error)
        }
        
        # Add noise to simulator
        simulator.add_noise(
            p_reset=np.random.uniform(min_error, max_error),
            p_meas=np.random.uniform(min_error, max_error),
            gate_error_probs=gate_error_probs
        )
        
        # Run noisy simulation
        noisy_counts = simulator.run_noisy(generate_histogram=False)

        # Fix bitstrings
        noisy_counts_fixed = {}
        for bitstring, count in noisy_counts.items():
            if verbose:
                print(bitstring, num_qubits)
            fixed_bitstring = bitstring.replace(" ", "")
            if len(fixed_bitstring) > num_qubits:
                fixed_bitstring = fixed_bitstring[-num_qubits:]
            elif len(fixed_bitstring) < num_qubits:
                fixed_bitstring = fixed_bitstring.zfill(num_qubits)
            noisy_counts_fixed[fixed_bitstring] = count
        
        noisy_dist = QuantumErrorNeuralNet.format_counts_to_distribution(
            noisy_counts_fixed, num_qubits=num_qubits
        )
        
        noisy_dists.append(noisy_dist)
    
    print("\n=== Ideal Distributions Summary ===")
    for i, dist in enumerate(ideal_dists[:5]):
        print(f"Distribution {i}:")
        print(f"  Shape: {dist.shape}")
        print(f"  Sum: {np.sum(dist)}")
        print(f"  Non-zero indices: {np.nonzero(dist)[1]}")
        print(f"  Non-zero values: {dist[0, np.nonzero(dist)[1]]}")
        print("  ---")
    # Stack all distributions
    noisy_distributions = np.vstack(noisy_dists)
    # Repeat ideal distribution for each noisy one
    ideal_distributions = np.vstack(ideal_dists)
    
    return noisy_distributions, ideal_distributions


def save_distribution_comparison(
    noisy_dists: np.ndarray,
    ideal_dists: np.ndarray,
    mitigated_dists: np.ndarray,
    output_dir: str,
    filename_prefix: str
):
    """
    Save comparison of distributions to file and generate visualization.
    
    Args:
        noisy_dists: Array of noisy distributions
        ideal_dists: Array of ideal distributions
        mitigated_dists: Array of mitigated distributions
        output_dir: Directory to save outputs
        filename_prefix: Prefix for output files
    """
    # Save distributions to CSV
    np.savetxt(
        os.path.join(output_dir, f"{filename_prefix}_noisy.csv"),
        noisy_dists,
        delimiter=","
    )
    np.savetxt(
        os.path.join(output_dir, f"{filename_prefix}_ideal.csv"),
        ideal_dists,
        delimiter=","
    )
    np.savetxt(
        os.path.join(output_dir, f"{filename_prefix}_mitigated.csv"),
        mitigated_dists,
        delimiter=","
    )
    
    # Generate visualization for first sample
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(noisy_dists.shape[1])
    width = 0.25
    
    ax.bar(x - width, noisy_dists[0], width, label='Noisy', alpha=0.7)
    ax.bar(x, ideal_dists[0], width, label='Ideal', alpha=0.7)
    ax.bar(x + width, mitigated_dists[0], width, label='Mitigated', alpha=0.7)
    
    ax.set_xlabel('Bitstring (as index)')
    ax.set_ylabel('Probability')
    ax.set_title('Comparison of Quantum Circuit Output Distributions')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_comparison.png"))
    plt.close()


def end_to_end_error_mitigation(
    circuit: QuantumCircuit,
    output_dir: str,
    num_training_samples: int = 50,
    num_test_samples: int = 10,
    model_name: str = "qem_model",
    error_range: Tuple[float, float] = (0.01, 0.25),
    hidden_layers: Tuple[int, ...] = None,
    verbose: bool = True,
    randomize_inputs: bool = False,
) -> Tuple[QuantumErrorNeuralNet, Dict]:
    """
    Perform end-to-end error mitigation workflow: generate training data,
    train model, and evaluate on test data.
    
    Args:
        circuit: Quantum circuit to mitigate
        output_dir: Directory to save outputs
        num_training_samples: Number of training samples to generate
        num_test_samples: Number of test samples to generate
        model_name: Name for the saved model and results
    
    Returns:
        Tuple containing (trained_model, results_dict)
    """
    print(f"Starting end-to-end error mitigation workflow for {circuit.name if circuit.name else 'unnamed'} circuit")
    
    # Generate training data
    print(f"Generating {num_training_samples} training samples...")
    train_noisy, train_ideal = generate_training_data(
        circuit, num_samples=num_training_samples, error_rates_range=error_range,
        randomize_inputs=randomize_inputs, verbose=verbose
    )
    
    # Generate test data
    print(f"Generating {num_test_samples} test samples...")
    test_noisy, test_ideal = generate_training_data(
        circuit, num_samples=num_test_samples, error_rates_range=error_range,
        randomize_inputs=randomize_inputs, verbose=verbose
    )
    if verbose:
        # save the test and training data to a file
        np.savetxt(
            os.path.join(output_dir, f"{model_name}_train_noisy.csv"),
            train_noisy,
            delimiter=","
        )
        np.savetxt(
            os.path.join(output_dir, f"{model_name}_train_ideal.csv"),
            train_ideal,
            delimiter=","
        )
        np.savetxt(
            os.path.join(output_dir, f"{model_name}_test_noisy.csv"),
            test_noisy,
            delimiter=","
        )
        np.savetxt(
            os.path.join(output_dir, f"{model_name}_test_ideal.csv"),
            test_ideal,
            delimiter=","
        )
    
    # Train and evaluate model
    model, results = train_error_mitigation_model(
        circuit=circuit,
        noisy_distributions=train_noisy,
        ideal_distributions=train_ideal,
        test_noisy_distributions=test_noisy,
        test_ideal_distributions=test_ideal,
        output_dir=output_dir,
        model_name=model_name,
        hidden_layers=hidden_layers,
    )
    
    print(f"Error mitigation workflow completed. Results saved to {output_dir}")
    return model, results