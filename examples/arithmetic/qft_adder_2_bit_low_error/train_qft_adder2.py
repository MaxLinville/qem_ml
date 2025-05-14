from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qem_ml.functions import end_to_end_error_mitigation
import numpy as np
import os

TEST_NAME = "qft_adder2_random"
OUTPUT_DIRECTORY = f"../../test_results/{TEST_NAME}_results"

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

n = 2  # Number of bits per operand
circuit = QuantumCircuit(2*n+1, n+1)  # Need extra qubit for carry

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

print("Running QFT adder training with randomized inputs...")

# Run error mitigation with randomized inputs
model, results = end_to_end_error_mitigation(
    circuit=circuit,
    output_dir=OUTPUT_DIRECTORY,
    model_name=TEST_NAME,
    error_range=(0.0001, 0.01),
    randomize_inputs=True,  # Enable randomized inputs for training
    verbose=False,  # Show more debugging information
    num_training_samples=2000,
)

print(f"Model training score: {results['training_score']}")
print(f"Test MSE: {results.get('test_mse', 'N/A')}")