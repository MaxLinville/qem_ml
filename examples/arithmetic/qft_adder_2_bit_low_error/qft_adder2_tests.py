from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qem_ml.run_model import run_model_with_inputs
import numpy as np
import os
import pandas as pd
from datetime import datetime

TEST_NAME = "qft_adder2_random"
BASE_OUTPUT_DIRECTORY = f"../../test_results/{TEST_NAME}_comprehensive_test"
MODEL_PATH = f"../../test_results/{TEST_NAME}_results/{TEST_NAME}.joblib"

# Create base directory
os.makedirs(BASE_OUTPUT_DIRECTORY, exist_ok=True)

# Create a results summary dataframe
results_summary = []

# Number of bits per operand
n = 2

# Loop through all combinations of a and b
for a in range(4):  # 0-3
    for b in range(4):  # 0-3
        print(f"\n{'='*50}")
        print(f"Testing inputs: a={a}, b={b}")
        
        # Create specific output directory for this combination
        OUTPUT_DIRECTORY = f"{BASE_OUTPUT_DIRECTORY}/a{a}_b{b}_results"
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        
        # Create QFT adder circuit with specific input values
        circuit = QuantumCircuit(2*n+1, n+1)  # 5 qubits, 3 classical bits

        # Set input values based on a and b
        # Convert a to binary and apply X gates for 1s
        a_bin = format(a, f'0{n}b')
        for i in range(n):
            if a_bin[n-i-1] == '1':
                circuit.x(i)

        # Convert b to binary and apply X gates for 1s
        b_bin = format(b, f'0{n}b')
        for i in range(n):
            if b_bin[n-i-1] == '1':
                circuit.x(i+n)

        # Apply QFT to the target register (first number + carry qubit)
        qft = QFT(n+1, do_swaps=False)
        circuit.append(qft, [0, 1, 4])

        circuit.barrier()

        # Define target register qubits explicitly
        target_qubits = [0, 1, 4]  # First number + carry

        # Apply controlled phase rotations
        # For each qubit in the second operand
        for j in range(n):
            # Control qubit is from second operand (j+n)
            control = j + n
            
            # For each target qubit in the target register
            for k in range(n+1):
                # Skip if j > k as those phases would be too small
                if j <= k:
                    # Get the actual target qubit index from our target register
                    target = target_qubits[k]
                    
                    # Skip if control and target are the same qubit
                    if control != target:
                        # Calculate rotation angle: 2π/2^(k-j+1)
                        angle = 2 * np.pi / (2 ** (k-j+1))
                        circuit.cp(angle, control, target)

        circuit.barrier()

        # Apply inverse QFT to the result register
        inverse_qft = QFT(n+1, do_swaps=False).inverse()
        circuit.append(inverse_qft, [0, 1, 4])

        circuit.barrier()

        # Measure the result (3 bits to represent numbers 0-5)
        circuit.measure(0, 2)  # LSB of result
        circuit.measure(1, 1)
        circuit.measure(4, 0)  # MSB of result (carry)

        # Save circuit diagram
        circuit.draw(output='mpl', filename=f"{OUTPUT_DIRECTORY}/circuit_drawing.png")

        print(f"Expected classical result: {a} + {b} = {a+b}")

        # Run test with the specified inputs
        try:
            if a==0 and b==0:
                draw_circuit = True
            else:
                draw_circuit = False
            results = run_model_with_inputs(
                circuit_file=None,  # No file needed as we are passing the circuit directly
                circuit=circuit,    # Pass the circuit we created
                model_path=MODEL_PATH,
                input_values=[int(bit) for bit in a_bin + b_bin] + [0],  # Convert binary strings to list of ints
                error_rate=0.005,
                output_dir=OUTPUT_DIRECTORY,
                draw_circuit=draw_circuit,
            )

            # Helper function to interpret the bitstring as a decimal number
            def interpret_result(bitstring):
                # Extract the relevant bits
                if ' ' in bitstring:
                    relevant_bits = bitstring.split()[-1]
                else:
                    relevant_bits = bitstring[-3:]
                    
                # Reverse the bits to account for the measurement order
                reversed_bits = relevant_bits[::-1]
                return int(reversed_bits, 2)

            # Get top results
            clean_top = sorted(results['clean_counts'].items(), key=lambda x: x[1], reverse=True)[0][0]
            noisy_top = sorted(results['noisy_counts'].items(), key=lambda x: x[1], reverse=True)[0][0]
            mitigated_top = sorted(results['mitigated_counts'].items(), key=lambda x: x[1], reverse=True)[0][0]

            # Interpret results
            clean_result = interpret_result(clean_top)
            noisy_result = interpret_result(noisy_top)
            mitigated_result = interpret_result(mitigated_top)

            # Add to summary
            results_summary.append({
                'a': a,
                'b': b,
                'expected': a+b,
                'clean_result': clean_result,
                'clean_correct': clean_result == a+b,
                'noisy_result': noisy_result,
                'noisy_correct': noisy_result == a+b,
                'mitigated_result': mitigated_result,
                'mitigated_correct': mitigated_result == a+b,
                'noisy_mse': results['noisy_mse'],
                'mitigated_mse': results['mitigated_mse'],
                'improvement': (1 - results['mitigated_mse']/results['noisy_mse'])*100,
                'directory': OUTPUT_DIRECTORY
            })

            # Print results summary for this combination
            print(f"\nTest completed for a={a}, b={b}")
            print(f"Noisy MSE: {results['noisy_mse']:.6f}")
            print(f"Mitigated MSE: {results['mitigated_mse']:.6f}")
            print(f"Improvement: {(1 - results['mitigated_mse']/results['noisy_mse'])*100:.2f}%")
            print(f"Clean result: {clean_result} (Expected: {a+b})")
            print(f"Noisy result: {noisy_result} (Expected: {a+b})")
            print(f"Mitigated result: {mitigated_result} (Expected: {a+b})")
            
        except Exception as e:
            print(f"Error testing a={a}, b={b}: {str(e)}")
            results_summary.append({
                'a': a,
                'b': b,
                'expected': a+b,
                'error': str(e)
            })

# Create a DataFrame from the results summary
df = pd.DataFrame(results_summary)

# Calculate summary statistics
summary = {
    'total_tests': len(df),
    'clean_accuracy': df['clean_correct'].mean() if 'clean_correct' in df.columns else 'N/A',
    'noisy_accuracy': df['noisy_correct'].mean() if 'noisy_correct' in df.columns else 'N/A',
    'mitigated_accuracy': df['mitigated_correct'].mean() if 'mitigated_correct' in df.columns else 'N/A',
    'average_improvement': df['improvement'].mean() if 'improvement' in df.columns else 'N/A',
}

# Save results to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df.to_csv(f"{BASE_OUTPUT_DIRECTORY}/comprehensive_results_{timestamp}.csv", index=False)

# Print overall summary
print("\n" + "="*50)
print("OVERALL TESTING SUMMARY")
print("="*50)
print(f"Total tests: {summary['total_tests']}")
print(f"Clean accuracy: {summary['clean_accuracy']:.2%}")
print(f"Noisy accuracy: {summary['noisy_accuracy']:.2%}")
print(f"Mitigated accuracy: {summary['mitigated_accuracy']:.2%}")
print(f"Average improvement: {summary['average_improvement']:.2f}%")

print(f"\nDetailed results saved to: {BASE_OUTPUT_DIRECTORY}/comprehensive_results_{timestamp}.csv")