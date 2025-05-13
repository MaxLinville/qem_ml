from qiskit import QuantumCircuit
from qem_ml.run_model import test_model_with_inputs
import os

# Set input values for testing
a = 1
b = 0

TEST_NAME = "half_adder_random"
OUTPUT_DIRECTORY = f"../../test_results/{TEST_NAME}_test_{a}{b}_results"
MODEL_PATH = f"../../test_results/{TEST_NAME}_results/{TEST_NAME}.joblib"

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Create half adder circuit with specific input values
circuit = QuantumCircuit(4, 2)

# Set input qubits based on a and b values
if a == 1:
    circuit.x(0)  # Set first input
if b == 1:
    circuit.x(1)  # Set second input

# Core half adder logic remains the same
circuit.cx(0, 2)  # First XOR gate
circuit.cx(1, 2)  # Second XOR gate
circuit.ccx(0, 1, 3)  # AND gate for carry bit

# Measure the outputs
circuit.measure(2, 0)  # extract XOR value (sum bit)
circuit.measure(3, 1)  # extract AND value (carry bit)

# Save circuit diagram
circuit.draw(output='mpl', filename=f"{OUTPUT_DIRECTORY}/circuit_drawing.png")

print(f"Testing half adder model with inputs: a={a}, b={b}")
print(f"Expected classical result: sum={a ^ b}, carry={a & b}")

# Run test with the specified inputs
results = test_model_with_inputs(
    circuit_file=None,  # No file needed as we are passing the circuit directly
    circuit=circuit,    # Pass the circuit we created
    model_path=MODEL_PATH,
    input_values=[a, b, 0, 0],  # Not needed as we already set the inputs in the circuit
    error_rate=0.05,
    output_dir=OUTPUT_DIRECTORY,
)

# Print results summary
print(f"\nTest completed. Results saved to: {OUTPUT_DIRECTORY}")
print(f"Noisy MSE: {results['noisy_mse']:.6f}")
print(f"Mitigated MSE: {results['mitigated_mse']:.6f}")
print(f"Improvement: {(1 - results['mitigated_mse']/results['noisy_mse'])*100:.2f}%")

# Print the most probable outcomes
print("\nMost likely outcomes:")
for result_type in ['clean_counts', 'noisy_counts', 'mitigated_counts']:
    counts = results[result_type]
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{result_type.replace('_', ' ').title()}:")
    for i, (bitstring, count) in enumerate(sorted_counts[:2]):
        total = sum(counts.values())
        percentage = (count / total) * 100
        print(f"  {bitstring}: {percentage:.2f}%")