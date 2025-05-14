import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from qiskit import QuantumCircuit
from qem_ml.functions import end_to_end_error_mitigation
from qem_ml.run_model import run_model_with_inputs
from qem_ml.circuits.simulator import Simulator
from mitiq import benchmarks, pec, zne
from qiskit_aer import AerSimulator

# Define constants
TEST_NAME = "mitiq_benchmark"
BASE_OUTPUT_DIR = f"./test_results/{TEST_NAME}_results"
MODEL_PATH = f"{BASE_OUTPUT_DIR}/{TEST_NAME}.joblib"
ERROR_RATE = 0.005  # Base error rate for noisy simulations

# Create directories
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ZNE configuration
zne_factory = zne.scaling.fold_global
zne_scales = [1.0, 2.0, 3.0]  # Scale factors for noise extrapolation

from datetime import datetime
from qiskit import QuantumCircuit
from qem_ml.functions import end_to_end_error_mitigation
from qem_ml.run_model import run_model_with_inputs
from qem_ml.circuits.simulator import Simulator
from mitiq import benchmarks, pec, zne
# Remove NQubitRepresentation that doesn't exist
from qiskit_aer import AerSimulator

# Define constants
TEST_NAME = "mitiq_benchmark"
BASE_OUTPUT_DIR = f"./test_results/{TEST_NAME}_results"
MODEL_PATH = f"{BASE_OUTPUT_DIR}/{TEST_NAME}.joblib"
ERROR_RATE = 0.005  # Base error rate for noisy simulations

# Create directories
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ZNE configuration
zne_factory = zne.scaling.fold_global
zne_scales = [1.0, 2.0, 3.0]  # Scale factors for noise extrapolation

# Create a set of benchmark circuits using Mitiq's built-in generators
def create_benchmark_circuits():
    circuits = []
    
    # Random benchmarking circuits with different qubit counts and depths
    # Using Mitiq's built-in RB circuit generator
    for n_qubits in [1, 2, 3]:
        for depth in [5, 10]:
            try:
                # Generate a randomized benchmarking circuit
                rb_circuits = benchmarks.generate_rb_circuits(
                    n_qubits=n_qubits, 
                    num_cliffords=depth, 
                    return_type="qiskit"
                )
                circuit = rb_circuits[0]  # Take the first circuit
                circuit.measure_all()  # Add measurements
                circuit.name = f"RB_{n_qubits}q_{depth}d"
                circuits.append(circuit)
            except Exception as e:
                print(f"Error generating RB circuit with {n_qubits} qubits and depth {depth}: {e}")
    
    # Add mirror circuits (another common benchmark)
    for n_qubits in [2, 3]:
        try:
            circuit = benchmarks.generate_mirror_circuit(n_qubits=n_qubits, return_type="qiskit")
            circuit.measure_all()
            circuit.name = f"Mirror_{n_qubits}q"
            circuits.append(circuit)
        except Exception as e:
            print(f"Error generating mirror circuit with {n_qubits} qubits: {e}")
            
    # Add QAOA-like circuits
    for n_qubits in [4, 6]:
        try:
            circuit = benchmarks.generate_quantum_volume_circuit(
                n_qubits=n_qubits, 
                depth=2,
                return_type="qiskit"
            )
            circuit.measure_all()
            circuit.name = f"QV_{n_qubits}q"
            circuits.append(circuit)
        except Exception as e:
            print(f"Error generating quantum volume circuit with {n_qubits} qubits: {e}")
    
    return circuits

# Function to run PEC mitigation
def run_with_pec(circuit, noise_model, shots=1000):
    """Run a circuit with Probabilistic Error Cancellation using Mitiq's current API"""
    # Create executor that runs noisy circuits with the same noise model
    def noisy_executor(circ):
        sim = AerSimulator(noise_model=noise_model)
        result = sim.run(circ, shots=shots).result()
        return result.get_counts(0)
    
    # Run with PEC mitigation - updated to use current Mitiq API without NQubitRepresentation
    try:
        # The simplest PEC approach with current Mitiq API
        mitigated_result = pec.execute_with_pec(
            circuit,
            noisy_executor,
            # No representation parameter needed anymore
        )
        return mitigated_result
    except Exception as e:
        print(f"Error running PEC: {e}")
        # Return empty counts if PEC fails
        return {}

# Function to run ZNE mitigation
def run_with_zne(circuit, noise_model, shots=1000):
    """Run a circuit with Zero Noise Extrapolation"""
    # Create executor that runs noisy circuits with the same noise model
    def noisy_executor(circ, shots=shots):
        sim = AerSimulator(noise_model=noise_model)
        result = sim.run(circ, shots=shots).result()
        return dict(result.get_counts(0))
    
    # Run with ZNE mitigation
    try:
        mitigated_result = zne.execute_with_zne(
            circuit,
            noisy_executor,
            scale_factors=zne_scales,
            factory=zne_factory,
        )
        return mitigated_result
    except Exception as e:
        print(f"Error running ZNE: {e}")
        # Return empty counts if ZNE fails
        return {}

# Function to plot comparison of results
def plot_comparison(clean_counts, noisy_counts, mitigated_counts, pec_counts, zne_counts, output_dir, title):
    """Create histogram comparison of all methods"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Normalize all distributions - check if any counts exist first
    clean_total = sum(clean_counts.values()) if clean_counts else 1
    noisy_total = sum(noisy_counts.values()) if noisy_counts else 1
    mitigated_total = sum(mitigated_counts.values()) if mitigated_counts else 1
    pec_total = sum(pec_counts.values()) if pec_counts else 1
    zne_total = sum(zne_counts.values()) if zne_counts else 1
    
    # Get all unique bitstrings
    all_bitstrings = set()
    for counts in [clean_counts, noisy_counts, mitigated_counts, pec_counts, zne_counts]:
        if counts:  # Check if counts exist
            all_bitstrings.update(counts.keys())
    all_bitstrings = sorted(list(all_bitstrings))
    
    # If no bitstrings, return early
    if not all_bitstrings:
        print("No data to plot")
        return
    
    # Prepare data for plotting
    x = np.arange(len(all_bitstrings))
    width = 0.15
    
    # Create normalized distributions with zeros for missing bitstrings
    clean_probs = [clean_counts.get(b, 0)/clean_total for b in all_bitstrings]
    noisy_probs = [noisy_counts.get(b, 0)/noisy_total for b in all_bitstrings]
    mitigated_probs = [mitigated_counts.get(b, 0)/mitigated_total for b in all_bitstrings]
    pec_probs = [pec_counts.get(b, 0)/pec_total for b in all_bitstrings]
    zne_probs = [zne_counts.get(b, 0)/zne_total for b in all_bitstrings]
    
    # Plot the bars
    ax.bar(x - 2*width, clean_probs, width, label='Clean', color='green', alpha=0.7)
    ax.bar(x - width, noisy_probs, width, label='Noisy', color='red', alpha=0.7)
    ax.bar(x, mitigated_probs, width, label='NN Mitigated', color='blue', alpha=0.7)
    ax.bar(x + width, pec_probs, width, label='PEC', color='purple', alpha=0.7)
    ax.bar(x + 2*width, zne_probs, width, label='ZNE', color='orange', alpha=0.7)
    
    # Add labels and legend
    ax.set_xticks(x)
    ax.set_xticklabels(all_bitstrings, rotation=45)
    ax.set_xlabel('Bitstring')
    ax.set_ylabel('Probability')
    ax.set_title(f'Comparison of Error Mitigation Techniques: {title}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_{title}.png")
    plt.close()

# Function to calculate MSE between two distributions
def calculate_mse(counts1, counts2):
    """Calculate Mean Squared Error between two count distributions"""
    # Get all unique bitstrings
    all_bitstrings = set(counts1.keys()) | set(counts2.keys())
    
    # Normalize counts to probabilities
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())
    
    # Calculate MSE
    mse = 0.0
    for b in all_bitstrings:
        p1 = counts1.get(b, 0) / total1
        p2 = counts2.get(b, 0) / total2
        mse += (p1 - p2) ** 2
    
    return mse / len(all_bitstrings)

# Main function
def main():
    # Step 1: Create benchmark circuits
    print("Creating benchmark circuits...")
    circuits = create_benchmark_circuits()
    
    # Step 2: Train error mitigation model on these circuits
    print(f"Training error mitigation model on benchmark circuits...")
    
    # Choose the first circuit as training circuit
    if len(circuits) > 0:
        training_circuit = circuits[0]
        
        # Train the model
        model, results = end_to_end_error_mitigation(
            circuit=training_circuit,
            output_dir=BASE_OUTPUT_DIR,
            model_name=TEST_NAME,
            error_range=(0.001, 0.01),  # Range of error rates for training
            num_training_samples=500,
            verbose=True
        )
        
        print(f"Model training completed with score: {results['training_score']}")
        
        # Step 3: Test on all benchmark circuits
        results_summary = []
        
        for circuit in circuits:
            print(f"\n{'='*50}")
            print(f"Testing circuit: {circuit.name}")
            
            # Create directory for this circuit's results
            circuit_output_dir = f"{BASE_OUTPUT_DIR}/{circuit.name}_results"
            os.makedirs(circuit_output_dir, exist_ok=True)
            
            try:
                # Run with neural network mitigation
                nn_results = run_model_with_inputs(
                    circuit_file=None,
                    circuit=circuit,
                    model_path=MODEL_PATH,
                    input_values=None,  # No specific input values
                    error_rate=ERROR_RATE,
                    output_dir=circuit_output_dir
                )
                
                # Extract counts from NN results
                clean_counts = nn_results['clean_counts']
                noisy_counts = nn_results['noisy_counts']
                mitigated_counts = nn_results['mitigated_counts']
                
                # Get noise model from the simulator for consistency
                simulator = Simulator(circuit)
                simulator.add_noise(
                    p_reset=ERROR_RATE,
                    p_meas=ERROR_RATE,
                    gate_error_probs={'h': ERROR_RATE, 'x': ERROR_RATE, 'cx': ERROR_RATE}
                )
                noise_model = simulator.noise_model
                
                # Run with PEC
                print("Running with PEC mitigation...")
                pec_counts = run_with_pec(circuit, noise_model)
                
                # Run with ZNE
                print("Running with ZNE mitigation...")
                zne_counts = run_with_zne(circuit, noise_model)
                
                # Calculate MSEs
                noisy_mse = nn_results['noisy_mse']
                mitigated_mse = nn_results['mitigated_mse']
                pec_mse = calculate_mse(pec_counts, clean_counts)
                zne_mse = calculate_mse(zne_counts, clean_counts)
                
                # Plot comparison
                plot_comparison(
                    clean_counts,
                    noisy_counts,
                    mitigated_counts,
                    pec_counts,
                    zne_counts,
                    circuit_output_dir,
                    circuit.name
                )
                
                # Add to results summary
                results_summary.append({
                    'circuit': circuit.name,
                    'num_qubits': circuit.num_qubits,
                    'noisy_mse': noisy_mse,
                    'mitigated_mse': mitigated_mse,
                    'pec_mse': pec_mse,
                    'zne_mse': zne_mse,
                    'nn_improvement': (1 - mitigated_mse/noisy_mse) * 100,
                    'pec_improvement': (1 - pec_mse/noisy_mse) * 100,
                    'zne_improvement': (1 - zne_mse/noisy_mse) * 100
                })
                
                print(f"\nResults for {circuit.name}:")
                print(f"Noisy MSE: {noisy_mse:.6f}")
                print(f"NN Mitigated MSE: {mitigated_mse:.6f} (Improvement: {(1 - mitigated_mse/noisy_mse)*100:.2f}%)")
                print(f"PEC MSE: {pec_mse:.6f} (Improvement: {(1 - pec_mse/noisy_mse)*100:.2f}%)")
                print(f"ZNE MSE: {zne_mse:.6f} (Improvement: {(1 - zne_mse/noisy_mse)*100:.2f}%)")
                
            except Exception as e:
                print(f"Error testing {circuit.name}: {str(e)}")
                results_summary.append({
                    'circuit': circuit.name,
                    'num_qubits': circuit.num_qubits,
                    'error': str(e)
                })
        
        # Create a DataFrame from the results summary
        df = pd.DataFrame(results_summary)
        
        # Calculate summary statistics (only if there are successful runs)
        if len(df) > 0 and 'nn_improvement' in df.columns and len(df[df['nn_improvement'].notna()]) > 0:
            summary = {
                'total_circuits': len(df),
                'successful_circuits': len(df[df['nn_improvement'].notna()]),
                'average_nn_improvement': df['nn_improvement'].mean(),
                'average_pec_improvement': df['pec_improvement'].mean(),
                'average_zne_improvement': df['zne_improvement'].mean(),
            }
            
            # Print overall summary
            print("\n" + "="*50)
            print("OVERALL BENCHMARK SUMMARY")
            print("="*50)
            print(f"Total circuits tested: {summary['total_circuits']}")
            print(f"Successful circuits: {summary['successful_circuits']}")
            print(f"Average NN improvement: {summary['average_nn_improvement']:.2f}%")
            print(f"Average PEC improvement: {summary['average_pec_improvement']:.2f}%")
            print(f"Average ZNE improvement: {summary['average_zne_improvement']:.2f}%")
            
            # Create an overall comparison plot
            successful_df = df.dropna(subset=['nn_improvement'])
            if len(successful_df) > 0:
                plt.figure(figsize=(12, 6))
                circuit_names = successful_df['circuit'].values
                x = np.arange(len(circuit_names))
                width = 0.25
                
                nn_improvements = successful_df['nn_improvement'].values
                pec_improvements = successful_df['pec_improvement'].values
                zne_improvements = successful_df['zne_improvement'].values
                
                plt.bar(x - width, nn_improvements, width, label='Neural Network', color='blue', alpha=0.7)
                plt.bar(x, pec_improvements, width, label='PEC', color='purple', alpha=0.7)
                plt.bar(x + width, zne_improvements, width, label='ZNE', color='orange', alpha=0.7)
                
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xticks(x, circuit_names, rotation=45)
                plt.xlabel('Circuit')
                plt.ylabel('Error Reduction (%)')
                plt.title('Comparison of Error Mitigation Techniques')
                plt.legend()
                plt.tight_layout()
                
                plt.savefig(f"{BASE_OUTPUT_DIR}/overall_comparison.png")
                plt.close()
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"{BASE_OUTPUT_DIR}/benchmark_results_{timestamp}.csv", index=False)
        print(f"\nDetailed results saved to: {BASE_OUTPUT_DIR}/benchmark_results_{timestamp}.csv")
    else:
        print("Error: No benchmark circuits were created.")

if __name__ == "__main__":
    main()