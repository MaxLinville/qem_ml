from qiskit import QuantumCircuit
from qem_ml.functions import end_to_end_error_mitigation
import os


TEST_NAME = "half_adder_random"
OUTPUT_DIRECTORY = f"../../test_results/{TEST_NAME}_results"

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# implementation https://qmunity.thequantuminsider.com/2024/06/11/building-a-half-adder-circuit/


circuit = QuantumCircuit(4, 2)

# initialization random from training
circuit.cx(0, 2)  # First XOR gate
circuit.cx(1, 2)  # Second XOR gate
circuit.ccx(0, 1, 3)  # AND gate for carry bit

# Measure the outputs
circuit.measure(2, 0)  # extract XOR value (sum bit)
circuit.measure(3, 1)  # extract AND value (carry bit)

# Save circuit diagram
circuit.draw(output='mpl', filename=f"{OUTPUT_DIRECTORY}/circuit_drawing.png")

print("Training half adder model with randomized inputs")

# Run error mitigation with randomized inputs
model, results = end_to_end_error_mitigation(
    circuit=circuit,
    output_dir=OUTPUT_DIRECTORY,
    model_name=TEST_NAME,
    error_range=(0.01, 0.1),
    randomize_inputs=True,
    num_training_samples=100,
    verbose=False
)

print(f"Model training score: {results['training_score']}")
print(f"Test MSE: {results.get('test_mse', 'N/A')}")

print("input combinations (00, 01, 10, 11) for the half adder circuit.")
print(f"Model saved to: {OUTPUT_DIRECTORY}/{TEST_NAME}.joblib")